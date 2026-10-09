"""Configuration, reproducibility, and readable resource errors."""
import json
import math
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")
    if not 1 <= cfg["training"]["epochs"] <= 100:
        raise ValueError("epochs must be in [1,100]")
    return cfg


def setup_runtime(seed):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__}: {len(gpus)} GPU(s) detected" if gpus else
          f"TensorFlow {tf.__version__}: no GPU detected; running on CPU. Use --smoke first.")
    return {"tensorflow": tf.__version__, "gpu_count": len(gpus)}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, data):
    Path(path).write_text(json.dumps(json_safe(data), indent=2, allow_nan=False), encoding="utf-8")


def run_cli(main):
    try:
        main()
    except (MemoryError, tf.errors.ResourceExhaustedError) as exc:
        raise SystemExit("RAM/VRAM exhausted. Reduce batch_size, tile_size, or image memory budget; "
                         "close other applications. Details: " + str(exc)) from exc
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise SystemExit("Setup/data error: " + str(exc)) from exc
