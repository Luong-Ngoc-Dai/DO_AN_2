"""Reproducible scene-split training and fixed-crop validation."""
import argparse
import copy
import csv
from dataclasses import asdict
from pathlib import Path
import numpy as np
import tensorflow as tf
import yaml
from .common import load_config, setup_runtime, save_json, run_cli
from .dataset import discover_pairs, split_pairs, save_split, load_split, make_dataset, read_crop, pair_size
from .metrics import image_metrics
from .model import build_cae, print_summary


class ValidationQuality(tf.keras.callbacks.Callback):
    """One deterministic center crop per validation image; no gradients/weight updates."""
    def __init__(self, root, pairs, patch_size, memory_budget_mb):
        super().__init__()
        self.root, self.pairs, self.size, self.budget = root, pairs, patch_size, memory_budget_mb

    def on_epoch_end(self, epoch, logs=None):
        rows = []
        clipped_mse = []
        for pair in self.pairs:
            w, h = pair_size(self.root, pair)
            left, top = (w-self.size)//2, (h-self.size)//2
            x, y = read_crop(self.root, pair, (left, top, left+self.size, top+self.size), self.budget)
            prediction = np.clip(self.model(x[None], training=False).numpy()[0], 0, 1)
            rows.append(image_metrics(y, prediction))
            clipped_mse.append(float(np.mean((y-prediction)**2)))
        logs["val_clipped_mse"] = float(np.mean(clipped_mse))
        logs["val_psnr"] = float(np.mean([r["PSNR"] for r in rows]))
        logs["val_ssim"] = float(np.mean([r["SSIM"] for r in rows]))
        logs["val_uqi"] = float(np.mean([r["UQI"] for r in rows]))
        print(f" validation clipped PSNR={logs['val_psnr']:.4f} dB; SSIM={logs['val_ssim']:.6f}")


def train_experiment(cfg, smoke=False, split_file=None):
    # 1. Đọc cấu hình và chia scene trước khi tạo patch.
    cfg = copy.deepcopy(cfg)
    runtime = setup_runtime(cfg["seed"])
    data_config = cfg["data"]
    train_config = cfg["training"]
    discovered = discover_pairs(data_config["root"], data_config["expected_pairs"])
    if split_file:
        _, train, val, saved_seed = load_split(split_file, data_config["root"])
        if saved_seed != cfg["seed"] or set(discovered) != set(train+val):
            raise ValueError("Saved split does not match dataset/seed")
    else:
        train, val = split_pairs(discovered, data_config["validation_fraction"], cfg["seed"])
    print(f"Dataset: {len(discovered)} pairs; {len({p.scene for p in discovered})} scenes; train {len(train)}, validation {len(val)}")
    output = Path(train_config["output"])
    if smoke:
        output = output/"smoke"
        # Real-data smoke uses the saved split, but only enough train images for two batches.
        train_run, val_run = train[:2*train_config["batch_size"]], val[:2]
        train_config["epochs"], data_config["patches_per_image"] = 1, 1
    else:
        train_run, val_run = train, val
    output.mkdir(parents=True, exist_ok=True)
    if (output/"config.yaml").exists():
        raise ValueError(f"Experiment directory already used: {output}; choose a new --output to avoid overwriting")
    # 2. Train crop ngẫu nhiên; validation crop cố định, không cập nhật weights.
    train_dataset = make_dataset(
        data_config["root"], train_run, data_config["patch_size"],
        train_config["batch_size"], training=True, seed=cfg["seed"],
        patches_per_image=data_config["patches_per_image"],
        memory_budget_mb=data_config["memory_budget_mb"],
    )
    validation_dataset = make_dataset(
        data_config["root"], val_run, data_config["patch_size"],
        train_config["batch_size"], training=False, seed=cfg["seed"],
        patches_per_image=1, memory_budget_mb=data_config["memory_budget_mb"],
    )
    # 3. Tạo CAE, in summary và tối ưu MSE trên đầu ra residual chưa clip.
    model = build_cae(cfg["model"]["filters"])
    save_json(output/"complexity.json", print_summary(model))
    model.compile(optimizer=tf.keras.optimizers.Adam(train_config["learning_rate"]), loss="mse")
    save_split(output/"split.json", data_config["root"], train, val, cfg["seed"])
    save_json(output/"run_manifest.json", {"smoke": smoke, "runtime": runtime,
              "train_used": [asdict(p) for p in train_run], "validation_used": [asdict(p) for p in val_run]})
    (output/"config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    # 4. Tính val_psnr trước để checkpoint và scheduler đọc được metric này.
    callbacks = [
        ValidationQuality(data_config["root"], val_run, data_config["patch_size"],
                          data_config["memory_budget_mb"]),
        tf.keras.callbacks.ModelCheckpoint(
            str(output/"best.keras"), monitor="val_psnr", mode="max", save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_psnr", mode="max",
            patience=train_config["early_stopping_patience"], restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_psnr", mode="max", factor=.5,
            patience=train_config["reduce_lr_patience"], min_lr=train_config["min_learning_rate"],
        ),
        tf.keras.callbacks.CSVLogger(str(output/"training_history.csv")),
    ]
    history = model.fit(
        train_dataset, validation_data=validation_dataset,
        epochs=train_config["epochs"], shuffle=False, callbacks=callbacks,
    )
    # 5. Lưu model và lịch sử để đánh giá/so sánh sau training.
    model.save(output/"restored_best.keras")
    save_json(output/"history.json", history.history)
    save_json(output/"training_summary.json", {"epochs_executed": len(history.epoch),
              "best_validation_psnr": max(history.history["val_psnr"]), "smoke": smoke,
              "loss_domain": "unclipped input + residual", "metric_domain": "clipped RGB [0,1]",
              "validation_protocol": "one fixed center crop per image; not independent test"})
    plot_history(output/"training_history.csv", output/"quality_history.png")
    return output


def plot_history(csv_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, key, label in zip(axes, ["val_psnr", "val_ssim"], ["Validation PSNR (dB)", "Validation SSIM"]):
        ax.plot([int(r["epoch"])+1 for r in rows], [float(r[key]) for r in rows], marker=".")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--data")
    p.add_argument("--output")
    p.add_argument("--split")
    p.add_argument("--smoke", action="store_true", help="A few real training batches, 1 epoch; not full SIDD")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["root"] = args.data
    if args.output:
        cfg["training"]["output"] = args.output
    print("Saved experiment:", train_experiment(cfg, args.smoke, args.split))


if __name__ == "__main__":
    run_cli(main)
