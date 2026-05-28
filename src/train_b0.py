"""EfficientNet-B0 second model — thin wrapper over train.py with B0-specific overrides.

Diversity axis vs. M1 (B2):
  - Smaller backbone (B0 vs B2): different inductive bias, less overfitting to clean train_audio
  - Heavier MixUp (alpha=0.6), no CutMix: more regularization for the lower-capacity model
  - Broader SpecAugment freq masking (max_f=30 vs 20): encourages frequency-robust features
  - Higher LR (1.5e-3) and longer training (40 epochs) to compensate for lower capacity
  - Outputs to outputs/b0/ to keep checkpoints separate from M1
"""

from __future__ import annotations

import argparse
import dataclasses
import numpy as np

from src.dataset import BirdCLEFDataset
from src.train import TrainConfig, train_fold


class _B0Dataset(BirdCLEFDataset):
    """BirdCLEFDataset with wider frequency masking for the B0 diversity run."""

    @staticmethod
    def _apply_specaugment(mel: np.ndarray) -> np.ndarray:
        _, freq_bins, time_frames = mel.shape

        for _ in range(2):
            width = np.random.randint(0, 31)
            if width > 0 and time_frames > width:
                start = np.random.randint(0, time_frames - width + 1)
                mel[:, :, start : start + width] = 0.0

        # freq_max raised to 30 bins (vs default 20) for more aggressive masking.
        for _ in range(2):
            height = np.random.randint(0, 31)
            if height > 0 and freq_bins > height:
                start = np.random.randint(0, freq_bins - height + 1)
                mel[:, start : start + height, :] = 0.0

        return mel


def _b0_train_fold(config: TrainConfig) -> dict:
    """Wrap train_fold, patching the dataset class with the B0 subclass."""
    import src.train as _train_module

    original_cls = _train_module.BirdCLEFDataset
    _train_module.BirdCLEFDataset = _B0Dataset
    try:
        return train_fold(config)
    finally:
        _train_module.BirdCLEFDataset = original_cls


B0_DEFAULTS = dict(
    backbone="tf_efficientnet_b0",
    epochs=40,
    lr=1.5e-3,
    mixup_alpha=0.6,
    cutmix_p=0.0,
    dropout=0.4,
    output_dir="outputs/b0",
)


def build_b0_config(**overrides) -> TrainConfig:
    """Return a TrainConfig with B0 defaults applied, then any caller overrides."""
    base = dataclasses.asdict(TrainConfig())
    base.update(B0_DEFAULTS)
    base.update(overrides)
    return TrainConfig(**base)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 diversity model")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=B0_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=B0_DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pseudo-label-csv", default=None)
    parser.add_argument("--mel-cache-root", default="data/mel_cache")
    parser.add_argument("--folds-csv", default="data/train_folds.csv")
    parser.add_argument("--sample-submission-csv", default="birdclef-2026/sample_submission.csv")
    parser.add_argument("--taxonomy-csv", default="birdclef-2026/taxonomy.csv")
    parser.add_argument("--output-dir", default=B0_DEFAULTS["output_dir"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return build_b0_config(
        val_fold=args.val_fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pseudo_label_csv=args.pseudo_label_csv,
        mel_cache_root=args.mel_cache_root,
        folds_csv=args.folds_csv,
        sample_submission_csv=args.sample_submission_csv,
        taxonomy_csv=args.taxonomy_csv,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print("B0 config:")
    for k, v in dataclasses.asdict(cfg).items():
        print(f"  {k}: {v}")
    result = _b0_train_fold(cfg)
    print("B0 training complete:", result)
