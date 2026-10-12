# Model B — Skip Residual CAE

Model A `build_cae()` and its layer names/configuration are unchanged. Legacy YAML
without `model.architecture` selects Model A. Old full `.keras` models still load
with `tf.keras.models.load_model`; no custom objects or checkpoint migration.

Model B adds `conv_7 + conv_3` at H/2 × W/2 × 64 and `conv_9 + conv_1` at
H × W × 32, before the next convolution. It retains `noisy_rgb + conv_10`, linear
RGB correction, nearest upsampling, ten 3×3 convolutions and all baseline filters.
Inputs must be divisible by four; tiled inference already pads arbitrary images.
Custom filters must match channels at each skip; no projection convolutions are
added. Outputs are unbounded during training; evaluation clips as before.

## Independent experiment

```bash
python -m src.architecture_report --output runs/architecture_comparison
python -m src.model --config configs/skip_cae.yaml
python -m src.train --config configs/skip_cae.yaml --data /path/to/SIDD/Data --smoke
# Run full training only when explicitly intended:
python -m src.train --config configs/skip_cae.yaml --data /path/to/SIDD/Data
python -m src.train --config configs/skip_cae.yaml --data /path/to/SIDD/Data --resume
```

Create `runs/sidd_split.json` with the existing dataset CLI if it is not already
available; never regenerate an existing split to compare models. Model B defaults
to reading it, verifies seed/pair membership and requires 256 training / 64
validation pairs for the 320-pair dataset. Batch 8, patch 128, seed 42 and all
other baseline settings are retained. Baseline outputs are separate. Synthetic
smoke tests demonstrate the pipeline, not SIDD performance.

`best.keras` selects highest validation-crop PSNR. `recovery/current` points to an
atomic generation containing `latest.keras` (model plus Adam slots/iterations/LR),
callback best/wait/cooldown counters, best-model snapshot and CSV history. Resume
requires the same configuration and split (only total epochs may change), starts
after the last committed epoch and restores CSV to that boundary. Partial epochs
are replayed; this is not mid-batch recovery. Crop seed advances by committed
epoch count. Bitwise reproducibility across different hardware/TF versions is not
promised. Do not run concurrent writers in one experiment directory. Recovery
is committed after LR scheduling and CSV logging and before best weights are
restored for final export. Resume is supported for runs created by this version;
old `best.keras` alone is an inference/best-weight checkpoint, not training recovery.

The existing PSNR/SSIM/UQI formulas and PNG evaluator are unchanged. Evaluate A
and B using the same saved split, tile size and overlap, then use `src.compare`.
This checkout has no official SIDD Validation MAT/1,280-block evaluator or dataset;
that protocol has not been implemented or validated by this change. Do not report
the existing center-crop or PNG validation as that benchmark.

## Hardware comparison

Both models: 241,827 parameters, 1,066,401,792 convolution MACs and 2,132,803,584
convolution FLOPs at 128×128 RGB. The report separately counts scalar additions
(including the global residual); convolution costs exclude biases, activations,
upsampling and clipping. Model B adds 786,432 scalar additions per patch. Add
avoids wider decoder convolutions but requires retained encoder feature buffers:
conv_3 has 262,144 elements and conv_1 has 524,288 elements per 128×128 patch.
At FP32 these require 1 MiB and 2 MiB respectively, before buffer reuse/tiling.
Actual BRAM, latency, throughput and quantization quality require synthesis and
quantized evaluation; equal convolution MACs do not establish equal FPGA cost.

No trained SIDD quality or FPGA synthesis results are claimed.
