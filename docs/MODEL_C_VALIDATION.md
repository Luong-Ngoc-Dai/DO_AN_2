# Model C validation

Executed on Python 3.12 / TensorFlow 2.20, CPU, with the pinned requirements:

```bash
/workspace/cae-venv/bin/python -m pip check
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /workspace/cae-venv/bin/python -m pytest -q -rs
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /workspace/cae-venv/bin/python -m src.architecture_report --output /tmp/cae-abc-report
```

- Dependency check: no broken requirements.
- Full suite: **40 passed, 1 skipped**, 50.33 seconds. Deprecation warnings from
  Keras/NumPy and sewar/SciPy remain; no failed tests.
- Skipped: `test_real_sidd_320_pairs`; real SIDD data is unavailable.
- Model C: graph/config, 128×128 / 128×256 / 256×256 forward, zero-correction
  identity, parameter/MAC/FLOP arithmetic and shape/padding/storage estimates pass.
- FP32 export: every kernel/bias round-trips through NPY and CSV exactly; source
  checkpoint hash stays unchanged. An independent NumPy SAME cross-correlation,
  nearest upsample and global Add reference matches every model layer on an
  8×12 synthetic input. Incompatible A and small-skip-B graphs are rejected.
- Model C smoke: synthetic 10-pair dataset, shared saved split, 128×128 patches,
  batch 8, one epoch/one training batch, two validation images. Saves best and
  recovery checkpoints, PSNR/SSIM history and exported checkpoint weights.
  Smoke intentionally uses one patch/pair, unlike full config's four.
- Model C interruption/resume: separate synthetic run with 16×16 crops and one
  patch/pair, interrupted after epoch-one recovery commit; resumes epoch two,
  restores optimizer iterations and produces nonduplicated history. Shared split
  bytes remain unchanged. Missing split and wrong 320-pair split counts fail.
- Existing Model A/B tests, old graph/output/metric equivalence and legacy full
  `.keras` fixture loading all pass. Historical trained `best.keras` files are
  not present; compatibility checks use the frozen legacy architecture fixture.
- Architecture reports generated for A/B/C and committed under
  `docs/architecture_comparison/`. INT16/INT8 storage numbers are hypothetical.

**Unrun:** real SIDD training/evaluation and PSNR/SSIM quality comparison, GPU
execution, calibrated quantization, RTL simulation, synthesis/place-and-route,
KV260 memory/latency/timing measurements. No 100-epoch training was launched.
