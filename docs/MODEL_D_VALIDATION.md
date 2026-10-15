# Model D validation

Executed on Linux CPU, Python 3.12 / TensorFlow 2.20, pinned requirements.

```bash
/workspace/cae-venv/bin/python -m pip check
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /workspace/cae-venv/bin/python -m pytest -q -rs
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /workspace/cae-venv/bin/python -m src.report_model_d --output /tmp/model-d-report
```

- Dependency check: no broken requirements.
- Full suite: **52 passed, 1 skipped**, 76.17 seconds, exit code 0.
  Existing Keras/NumPy and sewar/SciPy deprecation warnings remain.
- Model D suite alone: **12 passed**. Tests cover the supplied graph, input shapes
  including rectangular images, zero-correction identity, independent parameter
  and MAC arithmetic, fixed filters, full `.keras` loading, exact NPY round-trip,
  SHA-256 provenance, rejection of same-cost altered graphs, UTF-8 summary writes,
  synthetic training, post-training evaluation and interrupted training/resume.
- All existing Model A/B/C, metric and legacy-checkpoint fixture tests passed.
- Skipped: real SIDD 320-pair test; the actual dataset is unavailable.
- Computed by `src.report_model_d`: 7 Conv2D, 40,195 parameters,
  160,432,128 convolution MACs, 320,864,256 convolution FLOPs at RGB 128×128.
  Saved reports: `docs/model_d_architecture/`.

## Actual CLI smoke and evaluation

Generated a separate **synthetic** 320-pair PNG fixture under `/tmp`, split into
256/64 scene-disjoint pairs with seed 42. Used the unchanged supplied YAML and
only CLI data/split/output overrides:

```bash
python -m src.train --config configs/model_d.yaml --data /tmp/model-d-cli-synthetic/Data --split /tmp/model-d-cli-synthetic/sidd_split.json --output /tmp/model-d-cli-smoke --smoke
python -m src.export_model_d --model /tmp/model-d-cli-smoke/smoke/best.keras --output /tmp/model-d-cli-export
python -m src.evaluate --model /tmp/model-d-cli-smoke/smoke/best.keras --split /tmp/model-d-cli-synthetic/sidd_split.json --data /tmp/model-d-cli-synthetic/Data --output /tmp/model-d-cli-evaluation
```

All three commands exited 0. Smoke ran one epoch, two batches of eight 128×128
training crops, two validation crops, and saved the full 256/64 split. Export
produced seven kernel/bias pairs and an unquantized FP32 manifest. Evaluation
completed all 64 synthetic validation images with the existing PNG evaluator.
Synthetic quality metrics are not reported as SIDD performance.

Interruption/resume is tested separately with ten synthetic pairs, 16×16 crops,
one patch/pair and two epochs, interrupting after the first recovery commit.
Adam variables restore exactly, optimizer iterations advance to two, history has
epochs 0 and 1 without duplicates, and shared split bytes stay unchanged. Smoke
uses one patch/pair; full YAML remains four patches/pair/epoch and up to 100 epochs.

## Source provenance and remaining checks

Byte-for-byte comparison with the supplied ZIP passed for `src/model_d.py` and
`configs/model_d.yaml`. Model D source SHA-256:
`ffae42b95c6a9013be1fe57453e7c150ace468bcbe205c9f652f19f088b49106`.
The other addon files were integrated with the compatibility/validation changes
listed in `docs/MODEL_D.md`; the addon installer was not run or committed.

**Unrun:** real SIDD training/evaluation, native Windows execution (CP1252 summary
fallback is simulated in tests), GPU, and 100-epoch training. Historical trained
checkpoints are absent; backward-compatibility tests use existing frozen graph
fixtures. No Verilog, quantization or FPGA implementation was added. Dataset,
venv, checkpoint, `runs/` and exported/trained weights are not tracked in Git.
