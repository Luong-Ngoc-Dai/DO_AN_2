"""Train Model E on all SIDD Medium sRGB pairs without an internal validation split.

Save latest.keras every epoch and final.keras after the configured last epoch.
Optionally resume from latest.keras, restoring its optimizer state. Fixed training
crops can be monitored for diagnostic train PSNR/SSIM; these are NOT validation.
"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from .common import load_config, run_cli, save_json, setup_runtime
from .dataset import discover_pairs, make_dataset, pair_size, read_crop
from .metrics import image_metrics
from .model import build_model, print_summary


class TrainingQuality(tf.keras.callbacks.Callback):
    """Measure PSNR/SSIM on four fixed TRAIN crops without model selection."""

    def __init__(self, root, pairs, patch_size, memory_budget_mb, output, completed=0):
        super().__init__()
        self.root = root
        self.pairs = pairs[:4]
        self.patch_size = patch_size
        self.memory_budget_mb = memory_budget_mb
        self.path = output / "training_quality.csv"
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            # If interrupted during callback execution, drop any uncommitted rows.
            valid = [row for row in rows if int(row["epoch"]) <= completed]
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["epoch", "train_psnr", "train_ssim"])
                for row in valid:
                    writer.writerow([row["epoch"], row["train_psnr"], row["train_ssim"]])
        else:
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["epoch", "train_psnr", "train_ssim"])

    def on_epoch_end(self, epoch, logs=None):
        results = []
        for pair in self.pairs:
            width, height = pair_size(self.root, pair)
            left = (width - self.patch_size) // 2
            top = (height - self.patch_size) // 2
            noisy, gt = read_crop(
                self.root, pair,
                (left, top, left + self.patch_size, top + self.patch_size),
                self.memory_budget_mb,
            )
            prediction = np.clip(
                self.model(noisy[None], training=False).numpy()[0], 0.0, 1.0
            )
            results.append(image_metrics(gt, prediction))
        psnr = float(np.mean([r["PSNR"] for r in results]))
        ssim = float(np.mean([r["SSIM"] for r in results]))
        print(
            f" train_psnr={psnr:.4f} dB; train_ssim={ssim:.6f}"
            " (fixed TRAIN crops; not SIDD Validation)"
        )
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow([epoch + 1, psnr, ssim])


def _read_epochs(history_path):
    if not history_path.exists():
        raise ValueError(f"Cannot resume: missing training history at {history_path}")
    with history_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records or [int(row["epoch"]) for row in records] != list(range(len(records))):
        raise ValueError("Cannot resume: epoch history is empty or inconsistent")
    return len(records)


def train_full_sidd(cfg, smoke=False, resume=False):
    cfg = copy.deepcopy(cfg)
    if cfg["model"].get("architecture") != "two_skip_cae":
        raise ValueError("Full SIDD training supports only Model E (two_skip_cae)")
    if cfg["data"]["expected_pairs"] != 320 and not smoke:
        raise ValueError("Full-SIDD experiment requires exactly 320 pairs")
    if cfg["data"].get("split"):
        raise ValueError("Full-SIDD experiment must not use a train/validation split")
    if resume and smoke:
        raise ValueError("Resume for smoke tests is not supported")

    output = Path(cfg["training"]["output"])
    if smoke:
        output = output / "smoke"
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; use a new run directory")

    data = cfg["data"]
    training = cfg["training"]
    pairs = discover_pairs(data["root"], data["expected_pairs"])
    train_pairs = pairs[:min(len(pairs), 16)] if smoke else pairs
    if smoke:
        training["epochs"] = 1
        data["patches_per_image"] = 1

    runtime = setup_runtime(cfg["seed"])
    samples_per_epoch = len(train_pairs) * data["patches_per_image"]
    steps_per_epoch = (samples_per_epoch + training["batch_size"] - 1) // training["batch_size"]
    completed = 0

    if resume:
        if not (output / "config.yaml").exists() or not (output / "latest.keras").exists():
            raise ValueError("Cannot resume: missing config.yaml or latest.keras")
        previous = load_config(output / "config.yaml")
        comparable = copy.deepcopy(cfg)
        comparable["training"]["epochs"] = previous["training"]["epochs"]
        if comparable != previous:
            raise ValueError("Cannot resume: configuration differs (only epochs can change)")
        completed = _read_epochs(output / "training_history.csv")
        if completed >= training["epochs"]:
            raise ValueError("Already reached requested epochs")
        model = tf.keras.models.load_model(output / "latest.keras")
        if model.optimizer is None:
            raise ValueError("Cannot resume: latest.keras lacks optimizer")
        iterations = int(model.optimizer.iterations.numpy())
        if iterations != completed * steps_per_epoch:
            raise ValueError(
                f"Cannot resume: latest.keras has {iterations} optimizer steps "
                f"but history has {completed} epochs x {steps_per_epoch} steps"
            )
    else:
        model = build_model(cfg["model"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(training["learning_rate"]),
            loss="mse",
        )

    if model.count_params() != 40195:
        raise ValueError(f"Unexpected Model E parameter count: {model.count_params()}")
    dataset = make_dataset(
        data["root"], train_pairs, data["patch_size"], training["batch_size"],
        training=True, seed=cfg["seed"] + completed,
        patches_per_image=data["patches_per_image"],
        memory_budget_mb=data["memory_budget_mb"],
    )

    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        (output / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )
        with (output / "model_summary.txt").open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))
        save_json(output / "complexity.json", print_summary(model))
        save_json(output / "run_manifest.json", {
            "protocol": "all SIDD Medium sRGB pairs for training; no internal validation",
            "train_pair_count": len(train_pairs),
            "train_pairs": [
                dict(scene=p.scene, index=p.index, prefix=p.prefix,
                     noisy=p.noisy, gt=p.gt) for p in train_pairs
            ],
            "patches_per_epoch": samples_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": training["batch_size"],
            "patch_size": data["patch_size"],
            "model_selection": "final epoch; no validation-based selection",
            "runtime": runtime,
            "smoke": smoke,
        })

    print(
        f"Model E full-SIDD: {len(train_pairs)} pairs; {samples_per_epoch} patches, "
        f"{steps_per_epoch} batches/epoch; continuing after {completed} complete epochs"
    )
    # Keep CSVLogger's original header intact on resume (including its possible
    # "val_loss=NA" column). Training metrics live in a separate CSV.
    callbacks = [
        tf.keras.callbacks.CSVLogger(
            str(output / "training_history.csv"), append=resume
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(output / "latest.keras"), save_best_only=False
        ),
        TrainingQuality(
            data["root"], train_pairs, data["patch_size"],
            data["memory_budget_mb"], output, completed=completed
        ),
    ]
    model.fit(
        dataset, epochs=training["epochs"], initial_epoch=completed,
        shuffle=False, callbacks=callbacks,
    )
    model.save(output / "final.keras")
    with (output / "training_history.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    save_json(output / "training_summary.json", {
        "epochs_executed": len(rows),
        "train_pairs": len(train_pairs),
        "patches_per_epoch": samples_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": float(rows[-1]["loss"]),
        "checkpoint": "final.keras (last epoch; no validation-based selection)",
        "train_quality_csv": "training_quality.csv (TRAIN crops only)",
        "external_evaluation": "SIDD Validation sRGB 1280 blocks",
        "smoke": smoke,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_e_320pairs.yaml")
    parser.add_argument("--data", help="Path to extracted SIDD Medium sRGB/Data")
    parser.add_argument("--output", help="Run directory")
    parser.add_argument("--smoke", action="store_true", help="Short synthetic/real smoke test")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run using latest.keras and saved optimizer state",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["root"] = args.data
    if args.output:
        cfg["training"]["output"] = args.output
    print("Saved experiment:", train_full_sidd(
        cfg, smoke=args.smoke, resume=args.resume
    ))


if __name__ == "__main__":
    run_cli(main)
