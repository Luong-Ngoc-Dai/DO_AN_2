# Model E validation

Executed on Linux CPU with **Python 3.13.5 and TensorFlow 2.20.0**, the pinned
requirements, and `TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2`.

```bash
uv pip check --python /workspace/model-e-venv/bin/python
/workspace/model-e-venv/bin/python -m pytest tests/test_model_e.py -q -rs
/workspace/model-e-venv/bin/python -m pytest -q -rs
/workspace/model-e-venv/bin/python -m src.report_model_e --output /tmp/model-e-report
```

- Dependency check: all 56 installed packages compatible.
- Model E suite: **10 passed**, 10.51 seconds.
- Full A/B/C/D/E suite: **62 passed, 1 skipped**, 40.01 seconds, exit code 0.
  Existing Keras/NumPy and sewar/SciPy deprecation warnings remain.
- Skipped: real SIDD 320-pair dataset test; the real dataset is unavailable.
- Graph: exactly seven Conv2D, two nearest upsampling layers and three Adds;
  no prohibited layers. Conv configs and seed-42 initial weights match D.
- Half/full skip sources, consumers and tensor additions verified at 128×128
  and 128×256. Zero correction exactly returns noisy input.
- Independent arithmetic confirms 40,195 trainable parameters, 160,432,128 Conv
  MACs and 320,864,256 Conv FLOPs. The programmatic graph report counts 393,216
  extra scalar additions and per-map FP32/INT8 storage.
- `.keras` save/load, exact NPY kernel/bias round-trip and output equivalence pass.
  Export includes checksums and the two Add source/consumer positions; rejects D
  checkpoints and an E graph with altered interpolation but unchanged params.
- Synthetic training smoke at patch 128, batch 8 passes. It produces best,
  restored-best, recovery, CSV/history/summary, UTF-8 model summary and plot.
  Metrics/loss are finite; no NaN/Inf seen in the tested forwards/smoke. This is
  not evidence of convergence or superior quality on real SIDD.
- Resume test interrupts after the first committed epoch, confirms exact weights
  and Adam state restoration, then completes epoch two with no duplicated history.
  That small mechanics test uses 16×16 crops and one patch per pair. The shared
  split remains byte-identical. No learning-rate/loss/patience change was made.
- Existing A/B/C/D and legacy-checkpoint fixture tests pass. The frozen historical
  graph fixtures are tested; actual trained user checkpoints are not available.
- Windows-default CP1252 summary behavior is simulated in both training and report
  tests; UTF-8 writes pass. Native Windows itself has not been executed here.

## Actual CLI checks using synthetic images

Generated a separate 320-pair synthetic PNG fixture under `/tmp`, scene-disjoint
256/64 split, seed 42, and ran the original E YAML through `python -m src.train`
with only data/split/output overrides and `--smoke`. Smoke completed one epoch,
two batches of eight 128×128 crops and two validation crops, and saved the full
256/64 split. Smoke uses one patch per image; full config remains four patches
per image and up to 100 epochs. No real split file was created or overwritten.

CLI export, evaluation and full-image inference also exited 0:

```bash
python -m src.export_model_e --model /tmp/model-e-cli-smoke/smoke/best.keras --output /tmp/model-e-cli-export
python -m src.evaluate --model /tmp/model-e-cli-smoke/smoke/best.keras --split /tmp/model-e-cli-synthetic/sidd_split.json --data /tmp/model-e-cli-synthetic/Data --output /tmp/model-e-cli-evaluation
python -m src.inference --model /tmp/model-e-cli-smoke/smoke/best.keras --input /tmp/model-e-cli-synthetic/Data/0000_scene/NOISY_SRGB_010.PNG --gt /tmp/model-e-cli-synthetic/Data/0000_scene/GT_SRGB_010.PNG --output /tmp/model-e-cli-inference
```

Evaluation completed all 64 synthetic validation images with finite PSNR/SSIM/UQI.
Inference preserved the original 136×132 image size and generated `comparison.png`.
Export produced seven FP32 kernel/bias pairs, two skip descriptors and global
residual metadata. These checks verify the software path, not real SIDD quality.

## Remaining checks

Real SIDD training, validation quality and 100-epoch execution were not run.
`evaluate_sidd_blocks.py` is absent: the exact script/flags used for A/B/C/D and
real SIDD Validation sRGB 1,280-block data must be supplied before this benchmark
can be verified. No replacement evaluator was implemented. Quantization W8A8,
`.mem`, RTL, FPGA synthesis and KV260 board measurements are outside this change.
No dataset, venv, `runs/`, checkpoint or exported weights are tracked in Git.
