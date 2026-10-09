"""Compare the readable declaration with architecture/metrics frozen before refactor."""
import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from src.model import build_cae, complexity
from src.metrics import image_metrics

FIXTURES = Path(__file__).parent/'fixtures'


def test_model_matches_original_with_same_weights(tmp_path):
    # JSON chỉ là cấu trúc trước refactor, không chứa pretrained/trained weights.
    tf.keras.utils.set_random_seed(23)
    original = tf.keras.models.model_from_json(
        (FIXTURES/'cae_before_refactor.json').read_text(encoding='utf-8'))
    readable = build_cae()
    # JSON khôi phục shape thành list; mạng vừa tạo dùng tuple. Chuẩn hóa kiểu,
    # vẫn so toàn bộ cấu hình lớp và kết nối, không bỏ trường kiến trúc nào.
    assert json.loads(json.dumps(original.get_config())) == json.loads(json.dumps(readable.get_config()))
    assert complexity(original) == complexity(readable)
    readable.set_weights(original.get_weights())
    for old, new in zip(original.get_weights(), readable.get_weights()):
        np.testing.assert_array_equal(old, new)
    rng = np.random.default_rng(23)
    for size in (128, 256):
        image = rng.random((1, size, size, 3), dtype=np.float32)
        old_output = original(image, training=False).numpy()
        new_output = readable(image, training=False).numpy()
        np.testing.assert_array_equal(old_output, new_output)
        old_metrics = image_metrics(image[0], np.clip(old_output[0], 0, 1))
        new_metrics = image_metrics(image[0], np.clip(new_output[0], 0, 1))
        assert old_metrics == new_metrics
    # Checkpoint lưu bằng cấu trúc cũ vẫn nạp weights vào cấu trúc mới.
    checkpoint = tmp_path/'old.weights.h5'
    original.save_weights(checkpoint)
    restored = build_cae()
    restored.load_weights(checkpoint)
    np.testing.assert_array_equal(original(image).numpy(), restored(image).numpy())


def test_configured_filter_counts_still_supported():
    filters = [16, 16, 32, 32, 48, 32, 32, 16, 16, 3]
    model = build_cae(filters)
    assert [layer.filters for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)] == filters
    assert model(np.zeros((1, 128, 128, 3), dtype=np.float32)).shape == (1, 128, 128, 3)


def test_metrics_match_values_before_refactor():
    # Expected values computed from the old metric implementation, on synthetic pixels.
    rng = np.random.default_rng(17)
    reference = rng.uniform(.1, .9, (24, 28, 3)).astype(np.float32)
    prediction = np.clip(reference + rng.normal(0, .05, reference.shape), 0, 1)
    expected = json.loads((FIXTURES/'metrics_before_refactor.json').read_text())
    assert image_metrics(reference, prediction) == pytest.approx(expected, rel=1e-12, abs=1e-12)
