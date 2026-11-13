# Model E with all 320 SIDD Medium sRGB training pairs

This is a **new independent experiment** from `configs/model_e.yaml` (256/64).
The original split, checkpoints and training script are not changed.

Why a separate trainer? The original `src/train.py` requires the 256/64 saved
scene split and selects `best.keras` via validation PSNR. Using all 320 for
training leaves no internal held-out scene; falsely reporting a validation
PSNR would leak the evaluation set. `src/train_full_sidd.py` therefore trains
for a fixed 100 epochs with **no early stopping, no validation callbacks, and
no validation-based model selection**. It saves `final.keras` from the last
epoch and logs training loss plus diagnostic training-crop PSNR/SSIM (no validation).

## Run on Windows / PowerShell

```powershell
cd "C:\Nam_4_HK1\Do_an_2\DO_AN_2-feat-model-e-two-skip"
$py = ".\.venv\Scripts\python.exe"
$siddRoot = "C:\Nam_4_HK1\Do_an_2\dataset\SIDD_Medium_Srgb\mnt\d\SIDD_Medium_Srgb\Data"

# Sanity check first (16 training pairs, 1 epoch):
& $py -m src.train_full_sidd --config configs\model_e_320pairs.yaml --data "$siddRoot" --smoke

# All 320 pairs / 100 epochs. Use a unique output each time:
& $py -m src.train_full_sidd --config configs\model_e_320pairs.yaml --data "$siddRoot" --output runs\model_e_320pairs
```

The script checks that exactly 320 complete NOISY/GT PNG pairs exist. With
`patches_per_image=4` and `batch_size=8`, it processes 1280 randomly
cropped/augmented samples and 160 batches in each epoch. It draws fresh random
crops each epoch; this does **not** create a permanent cache of 96000 patches
like the reported FASNet preparation stage.

Output: `final.keras`, `latest.keras`, `training_history.csv`,
`training_summary.json`, `run_manifest.json`, `complexity.json`,
`model_summary.txt`, `config.yaml`. The filename `final.keras` is deliberate:
there is no external-validation checkpoint selection, so it is NOT
`best.keras`.

## Evaluation and comparison

Use the existing, separately maintained `evaluate_sidd_blocks` script that
produced the Model E 1280-block score. The external .mat files
`ValidationNoisyBlocksSrgb.mat` and `ValidationGtBlocksSrgb.mat` must
**never** appear in training, LR scheduling, or checkpoint selection.

Run the identical SIDD Validation sRGB evaluator on the new `final.keras`
and on the old `best.keras`. Record both the 256/64 vs full-320 training
difference and the checkpoint-selection difference in the report; this is
not a strict single-variable ablation because the selection protocol changed.

Before treating the 1280-block score as an independent final test, be aware
that comparing many configurations or selecting the best architecture based
on this set also creates evaluation-set selection bias. For a clean final
benchmark, use the separate official SIDD Benchmark submission or a genuinely
untouched external holdout.


## Resume an interrupted full-320 run

A later update adds `--resume`, using `latest.keras` (which Keras saves
after each completed epoch) and `training_history.csv` to infer the starting
epoch. The optimizer state is loaded from the complete Keras model checkpoint.
The loader refuses a resume if the optimizer iteration counter does not equal
completed epochs × steps/epoch or if configuration fields differ (apart from
the requested total epoch count). The training patches are freshly sampled
again, so resumed execution is NOT guaranteed bitwise identical to an
uninterrupted TensorFlow run.

You can stop the PowerShell process with Ctrl+C after an epoch has finished.
First back up `runs/model_e_320pairs/` and update the local source. Resume:

```powershell
& $py -m src.train_full_sidd --config configs\model_e_320pairs.yaml --data "$siddRoot" --output runs\model_e_320pairs --resume
```

This DOES NOT resume an in-progress batch or partial epoch, and it cannot
resume if the checkpoint and history disagree. Do not delete old checkpoints.
A full run that reaches 100 epochs already has `final.keras` and should
not be resumed unless the total number of epochs is explicitly raised, subject
to `src/common.py` currently enforcing `epochs <= 100`.

## Diagnostic quality monitoring

New runs print `train_psnr` and `train_ssim` at the end of each epoch.
These are calculated on four **fixed, center-cropped TRAIN images**, NOT on
held-out validation images, and saved to `training_quality.csv`.
The external `ValidationNoisyBlocksSrgb.mat` and
`ValidationGtBlocksSrgb.mat` are not used to monitor progress or choose
checkpoints. Existing runs may begin diagnostic logging at the resumed epoch,
so earlier epochs will not have these diagnostic scores.

Existing active Python processes use the code loaded when they started.
Finish/interrupt the current process before starting a separate resume
command; do not update local files mid-run.
