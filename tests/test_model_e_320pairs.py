"""Regression tests for full 320-pair training mode (synthetic smoke only)."""
import csv
import json
import pytest
import tensorflow as tf
from src.common import load_config
from src.train_full_sidd import train_full_sidd


def test_model_e_320_config_keeps_architecture():
    old = load_config("configs/model_e.yaml")
    new = load_config("configs/model_e_320pairs.yaml")
    assert old["model"] == new["model"]
    assert new["data"]["expected_pairs"] == 320
    assert "split" not in new["data"]
    assert new["training"]["output"] != old["training"]["output"]


def test_full_sidd_smoke_no_validation(tmp_path):
    from test_pipeline import fake_sidd
    root = tmp_path / "images"
    fake_sidd(root)
    cfg = load_config("configs/model_e_320pairs.yaml")
    cfg["data"].update(root=str(root), expected_pairs=10, patch_size=16)
    cfg["training"].update(output=str(tmp_path / "full"), epochs=2, batch_size=2)
    with pytest.raises(ValueError, match="exactly 320"):
        train_full_sidd(cfg)
    result = train_full_sidd(cfg, smoke=True)
    manifest = json.loads((result / "run_manifest.json").read_text())
    assert manifest["train_pair_count"] == 10
    assert manifest["steps_per_epoch"] == 5
    assert manifest["model_selection"].startswith("final epoch")
    rows = list(csv.DictReader((result / "training_history.csv").open()))
    assert len(rows) == 1
    assert "val_loss" not in rows[0] and "val_psnr" not in rows[0]
    model = tf.keras.models.load_model(result / "final.keras", compile=False)
    assert model.count_params() == 40195
    assert (result / "latest.keras").exists()
    with pytest.raises(ValueError, match="not empty"):
        train_full_sidd(cfg, smoke=True)
