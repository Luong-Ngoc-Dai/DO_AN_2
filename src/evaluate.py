"""Per-image validation or explicitly held-out evaluation, never mislabeled as test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .common import save_json, run_cli
from .dataset import discover_pairs, load_split, read_pair
from .inference import tiled_denoise
from .metrics import image_metrics


def evaluate(model, root, pairs, output, label="validation", tile_size=256, overlap=64, memory_budget_mb=2048):
    if not pairs:
        raise ValueError("No evaluation pairs")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    pair_fingerprint = hashlib.sha256(json.dumps(sorted((p.scene,p.index,p.noisy,p.gt) for p in pairs)).encode()).hexdigest()
    rows = []
    for i, pair in enumerate(pairs,1):
        noisy,gt = read_pair(root,pair,memory_budget_mb)
        prediction = tiled_denoise(model,noisy,tile_size,overlap,memory_budget_mb)
        baseline, scores = image_metrics(gt,noisy), image_metrics(gt,prediction)
        row = {"scene":pair.scene,"prefix":pair.prefix,"index":pair.index,"noisy_file":pair.noisy,"gt_file":pair.gt}
        row.update({f"{key.lower()}_{name}": value for name,metrics in [("noisy",baseline),("denoised",scores)] for key,value in metrics.items()})
        row["psnr_improvement"] = scores["PSNR"]-baseline["PSNR"]
        row["ssim_improvement"] = scores["SSIM"]-baseline["SSIM"]
        rows.append(row)
        print(f"{i}/{len(pairs)} {pair.scene}/{pair.index}: PSNR={scores['PSNR']:.4f} dB")
    with (output/"metrics_per_image.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [key for key in rows[0] if key.startswith(("psnr_","ssim_","uqi_"))]
    summary = {"split_label":label,"image_count":len(rows),"pair_list_sha256":pair_fingerprint,"data_range":1.0,
               "protocol":"full RGB images; tiled raw reconstruction blended then clipped [0,1]; equal weight per image",
               "tile_size":tile_size,"overlap":overlap,
               "means":{key:float(np.mean([r[key] for r in rows])) for key in keys},
               "note":"Validation is used for model selection, not an independent test. Nonfinite PSNR/improvement is serialized as a string."}
    save_json(output/"summary.json",summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--split",help="Saved split.json; evaluates validation only")
    source.add_argument("--heldout-data",help="Separate held-out dataset; user must ensure no overlap with training")
    p.add_argument("--data",help="Override dataset root when loading split")
    p.add_argument("--output",default="runs/evaluation")
    p.add_argument("--tile-size",type=int,default=256)
    p.add_argument("--overlap",type=int,default=64)
    p.add_argument("--memory-budget-mb",type=int,default=2048)
    args = p.parse_args()
    if args.split:
        root,_,pairs,_ = load_split(args.split,args.data)
        label = "validation"
    else:
        root,pairs,label = args.heldout_data,discover_pairs(args.heldout_data),"user_supplied_heldout"
    model = tf.keras.models.load_model(args.model,compile=False)
    summary = evaluate(model,root,pairs,args.output,label,args.tile_size,args.overlap,args.memory_budget_mb)
    print(summary["means"])


if __name__ == "__main__":
    run_cli(main)
