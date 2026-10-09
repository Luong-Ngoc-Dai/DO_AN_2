"""Pair by scene + optional numeric filename prefix + index; stream RGB crops."""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import re
import numpy as np
from PIL import Image
import tensorflow as tf
from .common import save_json, run_cli

_PATTERN = re.compile(
    r"^(?:(?P<prefix>\d+)_)?(?P<kind>NOISY|GT)_SRGB_(?P<index>\d+)\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Pair:
    scene: str
    index: str
    noisy: str
    gt: str
    prefix: str = ""  # Empty for legacy filenames/saved splits without a prefix.


def discover_pairs(root, expected_pairs=None):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"SIDD directory does not exist: {root}")
    entries = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = _PATTERN.fullmatch(path.name)
        if not match:
            if path.suffix.lower() == ".png" and "SRGB" in path.name.upper():
                raise ValueError(f"Unexpected SIDD filename: {path}")
            continue
        kind = match["kind"].upper()
        prefix = match["prefix"] or ""
        index = str(int(match["index"]))
        scene = path.parent.relative_to(root).as_posix()
        key = (scene, prefix, index)
        record = entries.setdefault(key, {})
        if kind in record:
            raise ValueError(f"Duplicate {kind} for scene/prefix/index {key}: {path}")
        record[kind] = path.relative_to(root).as_posix()
    pairs = []
    for (scene, prefix, index), record in sorted(entries.items()):
        if set(record) != {"NOISY", "GT"}:
            raise ValueError(f"Missing NOISY/GT partner for scene/prefix/index {(scene, prefix, index)}: {record}")
        pairs.append(Pair(scene, index, record["NOISY"], record["GT"], prefix))
    if not pairs:
        raise ValueError(f"No NOISY_SRGB/GT_SRGB pairs found in {root}; point to extracted sRGB Data directory")
    if expected_pairs is not None and len(pairs) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairs, found {len(pairs)}; inspect dataset extraction/root")
    return pairs


def pair_size(root, pair):
    sizes = []
    for rel in (pair.noisy, pair.gt):
        with Image.open(Path(root)/rel) as im:
            if im.mode != "RGB" or im.format != "PNG":
                raise ValueError(f"Expected 8-bit RGB PNG: {rel}, got {im.mode}/{im.format}")
            sizes.append(im.size)
    if sizes[0] != sizes[1]:
        raise ValueError(f"NOISY/GT dimensions differ: {pair.scene}/{pair.index}: {sizes}")
    return sizes[0]


def validate_pairs(root, pairs, patch_size):
    if patch_size < 8 or patch_size % 4:
        raise ValueError("patch_size must be >=8 and divisible by 4")
    for pair in pairs:
        if min(pair_size(root, pair)) < patch_size:
            raise ValueError(f"Image too small for {patch_size} patch: {pair}")


def split_pairs(pairs, val_fraction=0.2, seed=42):
    scenes = sorted({p.scene for p in pairs})
    if len(scenes) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two scene instances and validation fraction in (0,1)")
    random.Random(seed).shuffle(scenes)
    n = min(len(scenes)-1, max(1, round(len(scenes)*val_fraction)))
    val = set(scenes[:n])
    return [p for p in pairs if p.scene not in val], [p for p in pairs if p.scene in val]


def save_split(path, root, train, validation, seed):
    save_json(path, {"version": 2, "root": str(Path(root).resolve()), "seed": seed,
                     "protocol": "scene-instance split before crop; relative PNG paths",
                     "train": [asdict(p) for p in train], "validation": [asdict(p) for p in validation]})


def load_split(path, root=None):
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    train, val = ([Pair(**p) for p in data[k]] for k in ("train", "validation"))
    if not train or not val or {p.scene for p in train} & {p.scene for p in val}:
        raise ValueError("Invalid split: empty set or scene leakage")
    all_pairs = train + val
    keys = [(p.scene, p.prefix, p.index) for p in all_pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate pair in saved split")
    for p in all_pairs:
        for kind, rel in (("NOISY", p.noisy), ("GT", p.gt)):
            match = _PATTERN.fullmatch(Path(rel).name)
            if (not match or match["kind"].upper() != kind
                    or (match["prefix"] or "") != p.prefix
                    or str(int(match["index"])) != p.index):
                raise ValueError(f"Saved split scene/prefix/index does not match filename: {rel}")
            if Path(rel).is_absolute() or ".." in Path(rel).parts or Path(rel).parent.as_posix() != p.scene:
                raise ValueError(f"Invalid relative scene path in split: {rel}")
    return Path(root or data["root"]), train, val, data["seed"]


def check_memory(width, height, budget_mb, multiplier=24):
    estimate = width*height*multiplier
    if estimate > budget_mb*1024**2:
        raise MemoryError(f"Estimated image workspace {estimate/1024**2:.0f} MiB exceeds budget {budget_mb} MiB")


def read_rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path}")
        return np.asarray(im, dtype=np.float32)/255.0


def read_pair(root, pair, memory_budget_mb=1024):
    w, h = pair_size(root, pair)
    check_memory(w, h, memory_budget_mb, 32)
    return read_rgb(Path(root)/pair.noisy), read_rgb(Path(root)/pair.gt)


def read_crop(root, pair, box, memory_budget_mb=1024):
    w, h = pair_size(root, pair)
    check_memory(w, h, memory_budget_mb, 12)
    left, top, right, bottom = box
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError(f"Crop outside image: {box} / {(w,h)}")
    # Pillow decodes one full PNG at a time; convert only the cropped pixels to float32.
    crops = []
    for rel in (pair.noisy, pair.gt):
        with Image.open(Path(root)/rel) as im:
            crops.append(np.asarray(im.crop(box), dtype=np.float32)/255.0)
    return tuple(crops)


def make_dataset(root, pairs, patch_size=128, batch_size=8, training=False,
                 seed=42, patches_per_image=4, memory_budget_mb=1024):
    if batch_size < 1 or patches_per_image < 1 or not pairs:
        raise ValueError("Need nonempty pairs, batch_size and patches_per_image >=1")
    validate_pairs(root, pairs, patch_size)
    for pair in pairs:
        check_memory(*pair_size(root, pair), memory_budget_mb, 12)
    epoch = 0
    count = len(pairs)*(patches_per_image if training else 1)
    def generate():
        nonlocal epoch
        rng = np.random.default_rng(seed+epoch if training else seed)
        if training:
            epoch += 1
        for i in (rng.permutation(len(pairs)) if training else range(len(pairs))):
            pair = pairs[i]
            w, h = pair_size(root, pair)
            # Decode one uint8 pair per image, reuse for all crops; never cache the dataset.
            with Image.open(Path(root)/pair.noisy) as noisy, Image.open(Path(root)/pair.gt) as gt:
                noisy.load()
                gt.load()
                for _ in range(patches_per_image if training else 1):
                    left = int(rng.integers(w-patch_size+1)) if training else (w-patch_size)//2
                    top = int(rng.integers(h-patch_size+1)) if training else (h-patch_size)//2
                    box = (left, top, left+patch_size, top+patch_size)
                    x = np.asarray(noisy.crop(box), dtype=np.float32)/255.0
                    y = np.asarray(gt.crop(box), dtype=np.float32)/255.0
                    if training:
                        k = int(rng.integers(4))
                        x, y = np.rot90(x, k), np.rot90(y, k)
                        if rng.random() < .5:
                            x, y = x[:, ::-1], y[:, ::-1]
                    yield np.ascontiguousarray(x), np.ascontiguousarray(y)
    spec = tf.TensorSpec((patch_size, patch_size, 3), tf.float32)
    ds = tf.data.Dataset.from_generator(generate, output_signature=(spec, spec))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return ds.apply(tf.data.experimental.assert_cardinality(count)).batch(batch_size).with_options(options).prefetch(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="runs/split.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-pairs", type=int, default=320)
    args = p.parse_args()
    pairs = discover_pairs(args.data, args.expected_pairs)
    validate_pairs(args.data, pairs, 128)
    train, val = split_pairs(pairs, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_split(args.output, args.data, train, val, args.seed)
    print(f"{len(pairs)} pairs, {len({p.scene for p in pairs})} scenes; train={len(train)}, validation={len(val)}; no scene overlap")
    print(f"Saved split and seed to {args.output}")


if __name__ == "__main__":
    run_cli(main)
