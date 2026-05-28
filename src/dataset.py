"""Dataset and augmentation utilities for BirdCLEF+ 2026."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_secondary_labels(value: object) -> list[str]:
    """Parse semicolon-separated secondary labels into a clean list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        return []
    labels = [item.strip() for item in value.split(";")]
    return [item for item in labels if item]


class BirdCLEFDataset(Dataset):
    """Load cached mel spectrograms and build smoothed multi-label targets."""

    def __init__(
        self,
        manifest_df,
        mel_cache_root,
        species_list,
        mode: str = "train",
        pseudo_label_weight: float = 0.5,
    ):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.mel_cache_root = Path(mel_cache_root)
        self.species_list = list(species_list)
        self.mode = mode
        self.pseudo_label_weight = float(pseudo_label_weight)
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_list)}
        self.num_classes = len(self.species_list)

        if self.num_classes != 234:
            warnings.warn(
                f"Expected 234 species columns, got {self.num_classes}.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_npy_path(self, row) -> Path:
        filepath = row.get("filepath")
        if isinstance(filepath, str) and filepath:
            path = Path(filepath)
            if path.is_absolute():
                return path
            return self.mel_cache_root / path

        filename = row.get("filename")
        primary_label = row.get("primary_label")
        if isinstance(filename, str) and primary_label is not None:
            return self.mel_cache_root / str(primary_label) / f"{Path(filename).stem}.npy"

        return self.mel_cache_root / "__missing__.npy"

    def _load_mel(self, npy_path: Path) -> np.ndarray:
        if not npy_path.exists():
            warnings.warn(f"Missing mel cache file: {npy_path}", stacklevel=2)
            return np.zeros((3, 224, 224), dtype=np.float32)

        try:
            mel = np.load(npy_path)
            mel = mel.astype(np.float32, copy=False)
            if mel.shape != (3, 224, 224):
                warnings.warn(
                    f"Unexpected mel shape {mel.shape} for {npy_path}, returning zeros.",
                    stacklevel=2,
                )
                return np.zeros((3, 224, 224), dtype=np.float32)
            return mel
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.warn(f"Failed loading {npy_path}: {exc}", stacklevel=2)
            return np.zeros((3, 224, 224), dtype=np.float32)

    def _build_label_vector(self, row) -> np.ndarray:
        # Label smoothing baseline: negatives=0.05 for all classes.
        label = np.full((self.num_classes,), 0.05, dtype=np.float32)

        primary = row.get("primary_label")
        if primary is not None:
            primary_code = str(primary)
            if primary_code in self.species_to_idx:
                label[self.species_to_idx[primary_code]] = 0.95

        # Secondary labels remain at 0.05 (masked / not treated as positives).
        # We parse them to validate known labels and avoid silent malformed strings.
        secondaries = _parse_secondary_labels(row.get("secondary_labels"))
        for sec in secondaries:
            if sec not in self.species_to_idx:
                continue
            # Explicit no-op to document behavior.
            label[self.species_to_idx[sec]] = 0.05

        return label

    @staticmethod
    def _apply_specaugment(mel: np.ndarray) -> np.ndarray:
        """Apply time/frequency masking in-place."""
        _, freq_bins, time_frames = mel.shape

        for _ in range(2):
            width = np.random.randint(0, 31)
            if width > 0 and time_frames > width:
                start = np.random.randint(0, time_frames - width + 1)
                mel[:, :, start : start + width] = 0.0

        for _ in range(2):
            height = np.random.randint(0, 21)
            if height > 0 and freq_bins > height:
                start = np.random.randint(0, freq_bins - height + 1)
                mel[:, start : start + height, :] = 0.0

        return mel

    @staticmethod
    def _apply_horizontal_flip(mel: np.ndarray, p: float = 0.5) -> np.ndarray:
        if np.random.rand() < p:
            mel = mel[:, :, ::-1].copy()
        return mel

    @staticmethod
    def _apply_gain_jitter(mel: np.ndarray) -> np.ndarray:
        # Equivalent to mel += Uniform(-0.25, 0.25) in normalized log-mel space.
        delta = (np.random.rand() - 0.5) * 0.5
        mel = mel + np.float32(delta)
        return np.clip(mel, -1.0, 1.0)

    def _augment(self, mel: np.ndarray) -> np.ndarray:
        mel = self._apply_specaugment(mel)
        mel = self._apply_horizontal_flip(mel, p=0.5)
        mel = self._apply_gain_jitter(mel)
        return mel

    def _sample_weight(self, row) -> float:
        weight = float(row.get("sample_weight", 1.0))
        is_pseudo = bool(row.get("is_pseudo_label", False))
        if is_pseudo:
            weight *= self.pseudo_label_weight
        return weight

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        npy_path = self._resolve_npy_path(row)
        mel = self._load_mel(npy_path)
        label = self._build_label_vector(row)
        weight = self._sample_weight(row)

        if self.mode == "train":
            mel = self._augment(mel)

        mel_tensor = torch.from_numpy(mel).float()
        label_tensor = torch.from_numpy(label).float()
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        return mel_tensor, label_tensor, weight_tensor


def collate_fn(batch):
    """Collate into float32 tensors: (mel_batch, label_batch, weight_batch)."""
    mels, labels, weights = zip(*batch)
    mel_batch = torch.stack(mels, dim=0).float()
    label_batch = torch.stack(labels, dim=0).float()
    weight_batch = torch.stack(weights, dim=0).float()
    return mel_batch, label_batch, weight_batch
