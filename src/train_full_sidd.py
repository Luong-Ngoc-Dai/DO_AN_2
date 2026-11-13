"""Train Model E on ALL 320 SIDD Medium sRGB pairs without internal hold-out.

This experiment uses a fixed number of epochs and saves final.keras. It does
NOT use SIDD Validation blocks for model selection, checkpoint selection, or LR
scheduling. Evaluate final.keras once with the separate 1280-block evaluator.
"""
import argparse
import csv
from pathlib import Path

import tensorflow as tf
import yaml

from .common import load_config, run_cli, save_json, setup_runtime
from .dataset import discover_pairs, make_dataset
from .model import build_model, print_summary


def train_full_sidd(cfg, smoke=False):
    cfg = yaml.safe_load(yaml.safe_dump(cfg))  # independent copy
    if cfg["model"].get("architecture") != "two_skip_cae":
        raise ValueError("Full-SIDD training is supported only for Model E (two_skip_cae)")
    if cfg["data"]["expected_pairs"] != 320 and not smoke:
        raise ValueError("Full-SIDD experiment requires exactly 320 pairs")
    if cfg["data"].get("split"):
        raise ValueError("Full-SIDD experiment must not use a train/validation split")
    if cfg["training"].get("resume"):
        raise ValueError("Resume is not supported; start a fresh run with a new output")
    output = Path(cfg["training"]["output"])
    if smoke:
        output = output / "smoke"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; use a new run directory")
    data = cfg["data"]
    training = cfg["training"]
    pairs = discover_pairs(data["root"], data["expected_pairs"])
    # All discovered NOISY/GT pairs are used for training; validation is external.
    train_pairs = pairs[:min(len(pairs), 16)] if smoke else pairs
    if smoke:
        training["epochs"] = 1
        data["patches_per_image"] = 1
    runtime = setup_runtime(cfg["seed"])
    dataset = make_dataset(
        data["root"], train_pairs, data["patch_size"], training["batch_size"],
        training=True, seed=cfg["seed"],
        patches_per_image=data["patches_per_image"],
        memory_budget_mb=data["memory_budget_mb"],
    )
    model = build_model(cfg["model"])
    if model.count_params() != 40195:
        raise ValueError(f"Unexpected Model E parameter count: {model.count_params()}")
    model.compile(optimizer=tf.keras.optimizers.Adam(training["learning_rate"]), loss="mse")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with (output / "model_summary.txt").open("w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))
    save_json(output / "complexity.json", print_summary(model))
    samples_per_epoch = len(train_pairs) * data["patches_per_image"]
    batch_size = training["batch_size"]
    steps_per_epoch = (samples_per_epoch + batch_size - 1) // batch_size
    save_json(output / "run_manifest.json", {
        "protocol": "all SIDD Medium sRGB pairs for training; external SIDD Validation is not used",
        "train_pair_count": len(train_pairs),
        "train_pairs": [dict(scene=p.scene, index=p.index, prefix=p.prefix,
                             noisy=p.noisy, gt=p.gt) for p in train_pairs],
        "patches_per_epoch": samples_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "patch_size": data["patch_size"],
        "model_selection": "final epoch; no validation-based selection",
        "runtime": runtime,
        "smoke": smoke,
    })
    print(f"Model E full-SIDD: {len(train_pairs)} pairs; {samples_per_epoch} "
          f"patches and {steps_per_epoch} batches per epoch; no validation/early stopping")
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(output / "training_history.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            str(output / "latest.keras"), save_best_only=False
        ),
    ]
    model.fit(dataset, epochs=training["epochs"], shuffle=False, callbacks=callbacks)
    # final.keras is the exact final-epoch model, not a best-val-PSNR checkpoint.
    model.save(output / "final.keras")
    with (output / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    save_json(output / "training_summary.json", {
        "epochs_executed": len(rows),
        "train_pairs": len(train_pairs),
        "patches_per_epoch": samples_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "final_train_loss": float(rows[-1]["loss"]),
        "checkpoint": "final.keras (final epoch, no validation metrics)",
        "external_evaluation": "SIDD Validation sRGB 1280 blocks; run after training",
        "smoke": smoke,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_e_320pairs.yaml")
    parser.add_argument("--data", help="Path to extracted SIDD Medium sRGB/Data directory")
    parser.add_argument("--output", help="Unique run directory; never reused")
    parser.add_argument("--smoke", action="store_true", help="Train 16 pairs for 1 epoch; not a real result")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["root"] = args.data
    if args.output:
        cfg["training"]["output"] = args.output
    print("Saved experiment:", train_full_sidd(cfg, smoke=args.smoke))


if __name__ == "__main__":
    run_cli(main)
