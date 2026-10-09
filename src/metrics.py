"""Per-image full-reference metrics for RGB images normalized to [0,1]."""
import numpy as np
from skimage.metrics import structural_similarity
from sewar.full_ref import uqi as sewar_uqi


def image_metrics(reference, prediction):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 8:
        raise ValueError("Expected matching HxWx3 RGB images, at least 8x8")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1:
        raise ValueError("Metrics require finite pixels in [0,1]")
    mse = np.mean((a-b)**2)
    return {"PSNR": float('inf') if mse == 0 else float(10*np.log10(1/mse)),
            "SSIM": float(structural_similarity(a, b, channel_axis=-1, data_range=1.0, win_size=7)),
            "UQI": float(sewar_uqi(a, b, ws=8))}
