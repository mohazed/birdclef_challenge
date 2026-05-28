"""Soundscape-domain validation for BirdCLEF+ 2026.

Evaluates model checkpoints against the 66 expert-annotated soundscape files,
which are the gold-standard proxy for test-set distribution (closer than OOF AUC
on train_audio due to the 8.7 dB SNR domain gap).
"""

from __future__ import annotations

import sys as _sys
import os as _os
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

import warnings
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.model import BirdCLEFModel


# ──────────────────────────────────────────────────────────────────────────────
# Mel spectrogram (must be byte-identical to training)
# ──────────────────────────────────────────────────────────────────────────────

SR = 32_000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16_000
INPUT_SIZE = 224


def _compute_mel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Return a (3, INPUT_SIZE, INPUT_SIZE) float32 array from a 5-second waveform."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [-1, 1]
    mel_db = mel_db / 80.0  # power_to_db range is roughly [-80, 0]
    mel_db = np.clip(mel_db, -1.0, 1.0)
    # Resize to INPUT_SIZE × INPUT_SIZE
    mel_resized = cv2.resize(mel_db, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Stack 3 identical channels
    return np.stack([mel_resized, mel_resized, mel_resized], axis=0).astype(np.float32)


def _parse_time(t: str) -> int:
    """Convert 'HH:MM:SS' to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ──────────────────────────────────────────────────────────────────────────────
# Main validation function
# ──────────────────────────────────────────────────────────────────────────────

def run_soundscape_validation(
    model_paths: list[str | Path],
    soundscape_dir: str | Path,
    annotation_csv: str | Path,
    species_list: list[str],
    output_dir: str | Path,
    taxonomy_csv: str | Path | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    max_files: int | None = None,
) -> dict:
    """
    Evaluate an ensemble of PyTorch checkpoints against expert-annotated soundscapes.

    Parameters
    ----------
    model_paths:     Paths to .pt checkpoint files (one per fold = teacher ensemble).
    soundscape_dir:  Directory containing .ogg soundscape files.
    annotation_csv:  Path to train_soundscapes_labels.csv.
    species_list:    Ordered 234-element list from sample_submission.columns[1:].
    output_dir:      Directory to write soundscape_val_auc.csv and summary.
    taxonomy_csv:    Optional path to taxonomy.csv; used to split bird vs. non-bird AUC.
    batch_size:      Mel windows per inference batch.
    device:          'cpu' or 'cuda'.

    Returns
    -------
    dict with keys: macro_auc, bird_auc, nonbird_auc, per_class_auc (Series)
    """
    soundscape_dir = Path(soundscape_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    n_classes = len(species_list)

    # ── Load taxonomy for bird / non-bird split ──────────────────────────────
    bird_mask = np.ones(n_classes, dtype=bool)  # default: treat all as birds
    if taxonomy_csv is not None:
        tax = pd.read_csv(taxonomy_csv)
        tax["primary_label"] = tax["primary_label"].astype(str)
        bird_set = set(tax.loc[tax["class_name"] == "Aves", "primary_label"].tolist())
        for i, sp in enumerate(species_list):
            bird_mask[i] = sp in bird_set

    # ── Load annotations ─────────────────────────────────────────────────────
    annot = pd.read_csv(annotation_csv)
    annot["primary_label"] = annot["primary_label"].astype(str)
    annot["start_sec"] = annot["start"].apply(_parse_time)
    annot["end_sec"] = annot["end"].apply(_parse_time)

    n_windows = len(annot)
    gt = np.zeros((n_windows, n_classes), dtype=np.float32)
    for row_i, row in annot.iterrows():
        for sp in row["primary_label"].split(";"):
            sp = sp.strip()
            if sp in species_to_idx:
                gt[row_i, species_to_idx[sp]] = 1.0

    # ── Load models ───────────────────────────────────────────────────────────
    models: list[BirdCLEFModel] = []
    for ckpt_path in model_paths:
        m = BirdCLEFModel.load_from_checkpoint(ckpt_path, device=device)
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} model(s).")

    # ── Run inference per annotated file ─────────────────────────────────────
    # Build a lookup: filename -> list of (row_index, start_sec, end_sec)
    file_windows: dict[str, list[tuple[int, int, int]]] = {}
    for row_i, row in annot.iterrows():
        fname = row["filename"]
        file_windows.setdefault(fname, []).append((row_i, row["start_sec"], row["end_sec"]))

    if max_files is not None:
        file_windows = dict(list(file_windows.items())[:max_files])

    preds = np.zeros((n_windows, n_classes), dtype=np.float32)
    missing_files = 0

    for fname, windows in file_windows.items():
        audio_path = soundscape_dir / fname
        if not audio_path.exists():
            warnings.warn(f"Missing soundscape file: {audio_path}")
            missing_files += 1
            continue

        audio, _ = librosa.load(audio_path, sr=SR, mono=True)

        # Collect mel arrays for this file's windows
        mel_arrays: list[np.ndarray] = []
        row_indices: list[int] = []
        for row_i, start_sec, end_sec in windows:
            seg_start = int(start_sec * SR)
            seg_end = int(end_sec * SR)
            seg = audio[seg_start:seg_end]
            # Pad if shorter than 5 s (shouldn't happen with expert annotations)
            target_len = 5 * SR
            if len(seg) < target_len:
                pad = target_len - len(seg)
                seg = np.pad(seg, (pad // 2, pad - pad // 2))
            elif len(seg) > target_len:
                seg = seg[:target_len]
            mel_arrays.append(_compute_mel(seg))
            row_indices.append(row_i)

        # Batch inference
        for batch_start in range(0, len(mel_arrays), batch_size):
            batch_mels = mel_arrays[batch_start : batch_start + batch_size]
            batch_rows = row_indices[batch_start : batch_start + batch_size]
            x = torch.from_numpy(np.stack(batch_mels)).to(device)  # (B, 3, 224, 224)

            batch_probs = np.zeros((len(batch_mels), n_classes), dtype=np.float32)
            with torch.no_grad():
                for m in models:
                    logits = m(x)
                    batch_probs += torch.sigmoid(logits).cpu().numpy()
            batch_probs /= len(models)

            for local_i, row_i in enumerate(batch_rows):
                preds[row_i] = batch_probs[local_i]

    if missing_files:
        print(f"Warning: {missing_files} soundscape file(s) not found; their windows have zero predictions.")

    # ── Per-class AUC ─────────────────────────────────────────────────────────
    per_class_auc: dict[str, float] = {}
    for i, sp in enumerate(species_list):
        y_true = gt[:, i]
        y_score = preds[:, i]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class_auc[sp] = float("nan")  # can't compute AUC
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_class_auc[sp] = float(roc_auc_score(y_true, y_score))

    auc_series = pd.Series(per_class_auc, name="auc")
    valid_mask = auc_series.notna()

    macro_auc = float(auc_series[valid_mask].mean())
    bird_idx = [i for i, sp in enumerate(species_list) if bird_mask[i]]
    nonbird_idx = [i for i, sp in enumerate(species_list) if not bird_mask[i]]

    bird_aucs = auc_series.iloc[bird_idx].dropna()
    nonbird_aucs = auc_series.iloc[nonbird_idx].dropna()
    bird_auc = float(bird_aucs.mean()) if len(bird_aucs) > 0 else float("nan")
    nonbird_auc = float(nonbird_aucs.mean()) if len(nonbird_aucs) > 0 else float("nan")

    # ── Save per-class CSV ────────────────────────────────────────────────────
    if taxonomy_csv is not None:
        tax = pd.read_csv(taxonomy_csv)
        tax["primary_label"] = tax["primary_label"].astype(str)
        tax_map = tax.set_index("primary_label")[["common_name", "class_name", "scientific_name"]]
        auc_df = auc_series.reset_index()
        auc_df.columns = ["species_code", "auc"]
        auc_df = auc_df.join(tax_map, on="species_code")
    else:
        auc_df = auc_series.reset_index()
        auc_df.columns = ["species_code", "auc"]

    # Count ground-truth positives per species
    gt_counts = pd.Series(gt.sum(axis=0).astype(int), index=species_list, name="gt_positives")
    auc_df = auc_df.join(gt_counts, on="species_code")
    auc_df.to_csv(output_dir / "soundscape_val_auc.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    n_valid = valid_mask.sum()
    n_total = len(species_list)
    print("\n" + "=" * 60)
    print("SOUNDSCAPE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Windows evaluated : {n_windows}")
    print(f"Classes with AUC  : {n_valid} / {n_total}")
    print(f"Macro AUC (all)   : {macro_auc:.4f}")
    print(f"Bird AUC          : {bird_auc:.4f}  ({len(bird_aucs)} classes)")
    print(f"Non-bird AUC      : {nonbird_auc:.4f}  ({len(nonbird_aucs)} classes)")

    top10_worst = auc_series.dropna().nsmallest(10)
    top10_best = auc_series.dropna().nlargest(10)

    print("\nTop-10 WORST classes:")
    _print_auc_table(top10_worst, auc_df)

    print("\nTop-10 BEST classes:")
    _print_auc_table(top10_best, auc_df)
    print("=" * 60 + "\n")

    return {
        "macro_auc": macro_auc,
        "bird_auc": bird_auc,
        "nonbird_auc": nonbird_auc,
        "per_class_auc": auc_series,
    }


def _print_auc_table(auc_subset: pd.Series, auc_df: pd.DataFrame) -> None:
    for sp, auc_val in auc_subset.items():
        row = auc_df[auc_df["species_code"] == sp]
        name = row["common_name"].values[0] if "common_name" in auc_df.columns and len(row) else sp
        cls = row["class_name"].values[0] if "class_name" in auc_df.columns and len(row) else "?"
        gt_n = int(row["gt_positives"].values[0]) if "gt_positives" in auc_df.columns and len(row) else -1
        print(f"  {sp:<16}  AUC={auc_val:.4f}  [{cls:<9}]  gt={gt_n:>4}  {name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate models on annotated soundscapes.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to .pt checkpoint files")
    parser.add_argument("--soundscape-dir", required=True, help="Directory with .ogg soundscapes")
    parser.add_argument("--annotation-csv", required=True, help="train_soundscapes_labels.csv")
    parser.add_argument("--sample-submission", required=True, help="sample_submission.csv for species order")
    parser.add_argument("--output-dir", default="outputs/soundscape_val", help="Where to save results")
    parser.add_argument("--taxonomy-csv", default=None, help="taxonomy.csv for bird/non-bird split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of soundscapes processed (for dry-run)")
    args = parser.parse_args()

    sub = pd.read_csv(args.sample_submission)
    species_list = list(sub.columns[1:])
    assert len(species_list) == 234, f"Expected 234 species, got {len(species_list)}"

    run_soundscape_validation(
        model_paths=args.checkpoints,
        soundscape_dir=args.soundscape_dir,
        annotation_csv=args.annotation_csv,
        species_list=species_list,
        output_dir=args.output_dir,
        taxonomy_csv=args.taxonomy_csv,
        batch_size=args.batch_size,
        device=args.device,
        max_files=args.max_files,
    )
