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
    # Keras CSVLogger may emit an empty validation column as "NA", even
    # when model.fit() was called without validation_data.
    assert "val_psnr" not in rows[0]
    assert rows[0].get("val_loss", "NA") in ("NA", "", "None")
    model = tf.keras.models.load_model(result / "final.keras", compile=False)
    assert model.count_params() == 40195
    assert (result / "latest.keras").exists()
    with pytest.raises(ValueError, match="not empty"):
        train_full_sidd(cfg, smoke=True)

def test_full_sidd_resume_preserves_epoch_and_optimizer(tmp_path):
    from test_pipeline import fake_sidd
    root = tmp_path / "images"
    fake_sidd(root)
    cfg = load_config("configs/model_e_320pairs.yaml")
    cfg["data"].update(root=str(root), expected_pairs=10, patch_size=16,
                       patches_per_image=1)
    cfg["training"].update(output=str(tmp_path / "resume_run"), epochs=1,
                           batch_size=2)
    with pytest.raises(ValueError, match="exactly 320"):
        train_full_sidd(cfg)
    # Synthetic smoke cannot be resumed by design; test a 10-pair run by
    # patching only the pair count guard through a temporary 320-pair listing.
    from src import train_full_sidd as full_module
    actual_discover = full_module.discover_pairs
    full_module.discover_pairs = lambda root, count: actual_discover(root, 10)
    try:
        cfg["data"]["expected_pairs"] = 320
        output = train_full_sidd(cfg)
        assert (output / "latest.keras").exists()
        cfg["training"]["epochs"] = 2
        output = train_full_sidd(cfg, resume=True)
        rows = list(csv.DictReader((output / "training_history.csv").open()))
        assert len(rows) == 2
        checkpoint = tf.keras.models.load_model(output / "final.keras")
        assert int(checkpoint.optimizer.iterations.numpy()) == 10
        diagnostics = list(csv.DictReader((output / "training_quality.csv").open()))
        assert len(diagnostics) == 2
        assert all(float(x["train_psnr"]) > 0 for x in diagnostics)
    finally:
        full_module.discover_pairs = actual_discover
