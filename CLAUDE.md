# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BirdCLEF+ 2026 Kaggle competition** — predict per-species presence probabilities for 234 species across 5-second windows of Pantanal wetland field recordings. Metric is macro-averaged ROC-AUC. Submission is a CPU-only Kaggle notebook with a hard **90-minute runtime cap**. Competition deadline: **June 3, 2026**.

Read `BirdCLEF_2026_Master_Plan.md` for the full strategy, decision branches, and Claude Code prompt templates. That document is the authoritative reference for this project.

## Data Layout

```
birdclef-2026/
  train.csv                   # ~35k clips: primary_label, secondary_labels, type, lat/lon, author, rating, filename, collection
  taxonomy.csv                # 234 target species: primary_label (iNat taxon ID), scientific_name, common_name, class_name
  sample_submission.csv       # row_id = {soundscape_id}_{end_time_s}; one column per species in fixed order
  train_audio/                # labeled single-species clips (Xeno-Canto + iNat style)
  train_soundscapes/          # field recordings, some expert-annotated; used for validation and pseudo-labeling
  train_soundscapes_labels.csv
  test_soundscapes/           # hidden during training; populated only during Kaggle evaluation
  recording_location.txt      # Pantanal, MS, Brazil; lat -16.5 to -21.6, lon -55.9 to -57.6
```

The `filename` column in `train.csv` is relative to `train_audio/`. Species are identified by integer iNat taxon IDs (e.g., `1161364`). Some `primary_label` values have sub-labels like `47158son01` — match `sample_submission.csv` column order exactly.

## Planned Code Structure

The codebase is built via the prompts in Part 6 of the master plan:

```
src/                  # Python package
  dataset.py          # BirdCLEFDataset: returns (image, label_vector, loss_mask)
  model.py            # timm backbone + GEM pooling + dropout + linear(234)
  train.py            # training loop with checkpoint soup, OOF prediction storage
  pseudo_label.py     # teacher inference → parquet pseudo-label files
  ensemble.py         # mean blend, correlation matrix, temporal smoothing
  metrics.py          # macro_roc_auc() skipping zero-positive classes
  export.py           # ONNX export + parity verification
configs/              # YAML configs loaded via a config dataclass
  baseline.yaml
  pseudo_label.yaml
notebooks/            # Kaggle submission notebook lives here
  inference.ipynb
data/
  folds.csv           # FROZEN grouped 5-fold split — never regenerate
  oof/                # per-model out-of-fold predictions (.npy or .parquet)
  pseudo_labels/      # per-window pseudo-label probability vectors (.parquet)
eda/                  # plots from EDA script
checkpoints/          # model weights + ONNX exports
experiments.csv       # config hash, CV mean/std, runtime
```

## Audio Preprocessing (canonical — must match training and inference exactly)

1. Load at **32,000 Hz, mono**
2. **5-second window** — random crop in training, fixed non-overlapping windows in inference
3. Zero-pad short clips, signal centered; std-normalize waveform
4. Log-mel spectrogram: `n_mels=128`, `n_fft=2048`, `hop_length=512`, `f_min=20`, `f_max=16000`
5. Resize to configurable input size (224×224 default)
6. Stack into **3 identical channels** (RGB-style for ImageNet-pretrained backbones)

Preprocessing parameters must be **byte-identical** between training and the inference notebook. This is the single most common source of silent score drops.

## Model Architecture

- **Tier 1 (student/final):** `efficientvit_b0`, `tf_efficientnet_b0_ns`, `mnasnet_100` — ONNX-deployable, fit in 90-min budget
- **Tier 2 (teacher only):** `efficientvit_b1` (288×288), `tf_efficientnetv2_s` — GPU-only for pseudo-label generation
- Head: GEM pooling → dropout → `nn.Linear(features, 234)`
- Loss: average of `BCEWithLogitsLoss` + multi-label focal loss, applied with `loss_mask` (secondary labels masked out)
- Optimizer: AdamW + cosine annealing with short warmup
- **Checkpoint soup:** average weights of checkpoints that improved validation CV (preferred over early stopping)

## Key Decisions Already Made

- **CV strategy:** grouped 5-fold by `site > recordist > date_block`; fold assignments go to `data/folds.csv` and are frozen forever
- **Class balancing:** cap each species at 500 clips, upsample rare classes to a minimum per fold
- **Secondary labels:** mask their loss (do not backpropagate); reliable ~+0.01 improvement
- **Augmentation (light):** time-shift ±1s, additive Mixup (label = elementwise max), mild gain, light SpecAugment — avoid heavy aug
- **Pseudo-labeling:** 25–45% batch mix, label combination = `max(original, pseudo_prob)`, amplitude scaling `10^uniform(amp_min, amp_max)`
- **Ensemble:** simple mean of sigmoid outputs; check inter-model OOF correlation before adding; diversity beats count
- **Temporal smoothing:** convolve per-window predictions with `[0.1, 0.2, 0.4, 0.2, 0.1]` — nearly always safe
- **Inference:** ONNX export for all final models; parallelize audio→mel with thread pool; 85-minute safety timer

## What to Measure and Trust

- **Trust:** out-of-fold (OOF) macro ROC-AUC — computed on `data/folds.csv` splits
- **Cross-check only:** Kaggle public leaderboard (partial, noisy; chasing it causes overfitting)
- All experiments logged to `experiments.csv` with config hash, CV mean ± std, and wall-clock runtime
- One change at a time; if an idea does not improve OOF in one attempt, discard it

## Common Commands

Once the project skeleton exists, typical commands will be:

```bash
# EDA
python src/eda.py --train birdclef-2026/train.csv --taxonomy birdclef-2026/taxonomy.csv

# Build frozen folds (run once only)
python src/folds.py --train birdclef-2026/train.csv --out data/folds.csv

# Training (1 fold for fast iteration, all 5 for final)
python src/train.py --config configs/baseline.yaml --fold 0
python src/train.py --config configs/baseline.yaml  # all folds

# Pseudo-label generation
python src/pseudo_label.py --config configs/pseudo_label.yaml --soundscapes birdclef-2026/train_soundscapes/

# Ensemble evaluation
python src/ensemble.py --oof-dir data/oof/ --folds data/folds.csv

# ONNX export + verification
python src/export.py --checkpoint checkpoints/<name>.pth --out checkpoints/<name>.onnx
```

## Competition Constraints Checklist

- [ ] Folds frozen in `data/folds.csv` — never regenerated mid-competition
- [ ] Preprocessing parameters identical between training and inference notebook
- [ ] All final models export to ONNX and outputs verified within tolerance vs PyTorch
- [ ] Inference notebook has an 85-minute safety timer with a fallback path
- [ ] Submission CSV column order matches `sample_submission.csv` exactly
- [ ] Two submissions prepared: conservative (proven blend) and bold (experimental blend)
