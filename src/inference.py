"""Overlap-add tiled inference, padding, clipped RGB output and visual reports."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from .common import run_cli
from .dataset import read_rgb, check_memory
from .metrics import image_metrics


def validate_rgb(image):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) < 1 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("Expected finite HxWx3 RGB pixels in [0,1]")
    return x


def denoise_array(model, image):
    x = validate_rgb(image)
    h, w = x.shape[:2]
    padded = np.pad(x, ((0,(-h)%4),(0,(-w)%4),(0,0)), mode="edge")
    return np.clip(model(padded[None], training=False).numpy()[0,:h,:w], 0, 1)


def _positions(length, tile, stride):
    positions = list(range(0, length-tile+1, stride))
    if positions[-1] != length-tile:
        positions.append(length-tile)
    return positions


def tiled_denoise(model, image, tile_size=256, overlap=64, memory_budget_mb=2048):
    x = validate_rgb(image)
    if tile_size < 8 or tile_size % 4 or overlap < 0 or overlap >= tile_size or overlap % 4:
        raise ValueError("tile_size >=8 and overlap must be multiples of 4; 0 <= overlap < tile_size")
    h, w = x.shape[:2]
    ph, pw = max(tile_size, (h+3)//4*4), max(tile_size, (w+3)//4*4)
    check_memory(pw, ph, memory_budget_mb, 48)
    x = np.pad(x, ((0,ph-h),(0,pw-w),(0,0)), mode="edge")
    stride = tile_size-overlap
    # Positive Hann window avoids zero-division at image edges.
    window = np.maximum(np.hanning(tile_size), 1e-3).astype(np.float32)
    weight = (window[:,None]*window[None,:])[...,None]
    accum = np.zeros_like(x)
    weights = np.zeros((ph,pw,1), dtype=np.float32)
    for top in _positions(ph,tile_size,stride):
        for left in _positions(pw,tile_size,stride):
            raw = model(x[None,top:top+tile_size,left:left+tile_size], training=False).numpy()[0]
            accum[top:top+tile_size,left:left+tile_size] += raw*weight
            weights[top:top+tile_size,left:left+tile_size] += weight
    # Blend raw residual reconstructions first, then clip once, consistent with full-image metrics.
    return np.clip(accum[:h,:w]/weights[:h,:w], 0, 1)


def save_rgb(path, array):
    Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8)).save(path)


def comparison_report(noisy, denoised, gt, output, crop=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h,w = noisy.shape[:2]
    if crop is None:
        size = min(128,h,w)
        crop = ((w-size)//2,(h-size)//2,size,size)
    left,top,cw,ch = crop
    if not (0 <= left < left+cw <= w and 0 <= top < top+ch <= h):
        raise ValueError("Comparison crop falls outside image")
    images = [("NOISY",noisy),("DENOISED",denoised)] + ([] if gt is None else [("GT",gt)])
    fig,axes = plt.subplots(2,len(images),figsize=(5*len(images),8),squeeze=False)
    for i,(title,image) in enumerate(images):
        axes[0,i].imshow(image)
        axes[0,i].set_title(title)
        axes[1,i].imshow(image[top:top+ch,left:left+cw])
        axes[1,i].set_title(f"Same crop: x={left}, y={top}, {cw}x{ch}")
        if gt is not None:
            m = image_metrics(gt,image)
            axes[1,i].set_xlabel(f"Full image PSNR={m['PSNR']:.3f} dB\nSSIM={m['SSIM']:.5f}; UQI={m['UQI']:.5f}")
        for ax in axes[:,i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(Path(output)/"comparison.png",dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--gt")
    p.add_argument("--output", default="runs/inference")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--memory-budget-mb", type=int, default=2048)
    p.add_argument("--crop", nargs=4, type=int, metavar=("X","Y","WIDTH","HEIGHT"))
    args = p.parse_args()
    with Image.open(args.input) as im:
        check_memory(*im.size, args.memory_budget_mb, 48)
    noisy = read_rgb(args.input)
    gt = read_rgb(args.gt) if args.gt else None
    if gt is not None and noisy.shape != gt.shape:
        raise ValueError("GT shape differs from NOISY; comparison requires aligned images")
    model = tf.keras.models.load_model(args.model,compile=False)
    result = tiled_denoise(model,noisy,args.tile_size,args.overlap,args.memory_budget_mb)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    save_rgb(output/"NOISY.png",noisy)
    save_rgb(output/"DENOISED.png",result)
    if gt is not None:
        save_rgb(output/"GT.png",gt)
    comparison_report(noisy,result,gt,output,args.crop)
    print(f"Saved {result.shape[1]}x{result.shape[0]} image and comparison to {output}")


if __name__ == "__main__":
    run_cli(main)
