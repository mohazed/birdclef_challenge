"""Data preparation pipeline for BirdCLEF+ 2026."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Optional

import cv2
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


def build_folds(train_csv_path: str | Path, out_path: str | Path = "data/train_folds.csv") -> pd.DataFrame:
    """Build 5-fold split and compute sample weights; saves to out_path."""
    train = pd.read_csv(train_csv_path)

    # groups: author, but Unknown rows use lat/lon bucket
    groups = train["author"].copy()
    unknown_mask = groups == "Unknown"
    groups[unknown_mask] = (
        train.loc[unknown_mask, "latitude"].fillna(0).astype(int).astype(str)
        + "_"
        + train.loc[unknown_mask, "longitude"].fillna(0).astype(int).astype(str)
    )

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(train, train["primary_label"], groups)):
        train.loc[val_idx, "fold"] = fold_idx

    def _weight(rating: float) -> float:
        if rating == 0.0:
            w = 0.50
        elif rating < 3.0:
            w = max(rating, 0.5) / 5.0
        else:
            w = rating / 5.0
        return max(w, 0.30)

    train["sample_weight"] = train["rating"].apply(_weight)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_path, index=False)
    print(f"Saved folds to {out_path}  ({len(train)} rows, fold distribution:\n{train['fold'].value_counts().sort_index()})")
    return train


# ---------------------------------------------------------------------------
# Mel cache helpers
# ---------------------------------------------------------------------------

_CACHE_ROOT: Optional[Path] = None
_AUDIO_ROOT: Optional[Path] = None
_FAILED_LOG: Optional[Path] = None
_RNG_SEED = 42


def _process_clip(row: tuple) -> Optional[str]:
    """Worker function: build one mel and save as float16 .npy. Returns path on success, error str on failure."""
    filename, primary_label = row
    audio_path = _AUDIO_ROOT / filename
    out_path = _CACHE_ROOT / primary_label / (Path(filename).stem + ".npy")

    if out_path.exists():
        return None  # already cached

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        y, _ = librosa.load(str(audio_path), sr=32000, mono=True)
    except Exception as e:
        return f"{filename}\t{e}"

    # 5-second window (160 000 samples at 32 kHz)
    target_len = 5 * 32000
    if len(y) >= target_len:
        rng = np.random.default_rng(_RNG_SEED + hash(filename) % (2**31))
        start = rng.integers(0, len(y) - target_len + 1)
        y = y[start : start + target_len]
    else:
        # zero-pad centered
        pad_total = target_len - len(y)
        pad_left = pad_total // 2
        y = np.pad(y, (pad_left, pad_total - pad_left))

    # mel spectrogram → dB → normalize
    mel = librosa.feature.melspectrogram(
        y=y, sr=32000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=16000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db / 80.0).clip(-1.0, 1.0)  # typical dB range ~80

    # resize to (128, 128) then stack 3 channels → (3, 224, 224)
    mel_128 = cv2.resize(mel_norm, (128, 128), interpolation=cv2.INTER_LINEAR)
    mel_224 = cv2.resize(mel_128, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = np.stack([mel_224, mel_224, mel_224], axis=0).astype(np.float16)

    np.save(str(out_path), img)
    return None


def _worker_init(cache_root: Path, audio_root: Path, failed_log: Path) -> None:
    global _CACHE_ROOT, _AUDIO_ROOT, _FAILED_LOG
    _CACHE_ROOT = cache_root
    _AUDIO_ROOT = audio_root
    _FAILED_LOG = failed_log


def build_mel_cache(
    train_folds_csv: str | Path,
    audio_root: str | Path,
    cache_root: str | Path,
    num_workers: int = 8,
) -> None:
    """Compute and cache mel spectrograms for all clips in train_folds_csv."""
    train = pd.read_csv(train_folds_csv)
    audio_root = Path(audio_root)
    cache_root = Path(cache_root)
    failed_log = Path("data/failed_clips.txt")
    failed_log.parent.mkdir(parents=True, exist_ok=True)

    rows = list(zip(train["filename"], train["primary_label"].astype(str)))

    # filter already-done
    pending = [
        (fn, lbl)
        for fn, lbl in rows
        if not (cache_root / lbl / (Path(fn).stem + ".npy")).exists()
    ]
    print(f"Clips to process: {len(pending)} / {len(rows)}")

    errors: list[str] = []

    def _init():
        _worker_init(cache_root, audio_root, failed_log)

    # patch globals in main process too (needed if num_workers=0)
    _worker_init(cache_root, audio_root, failed_log)

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(cache_root, audio_root, failed_log),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_clip, pending, chunksize=32),
            total=len(pending),
            desc="Building mel cache",
        ):
            if result is not None:
                errors.append(result)

    if errors:
        with open(failed_log, "w") as f:
            f.write("\n".join(errors) + "\n")
        print(f"  {len(errors)} failed clips logged to {failed_log}")
    else:
        print("  All clips processed successfully.")


# ---------------------------------------------------------------------------
# Submission columns
# ---------------------------------------------------------------------------


def extract_soundscape_annotation_windows(
    annotation_csv: str | Path,
    soundscape_dir: str | Path,
    out_path: str | Path = "data/soundscape_train_windows.csv",
) -> pd.DataFrame:
    """Extract annotated 5-second windows from labeled soundscapes.

    Reads train_soundscapes_labels.csv and produces a training manifest
    with one row per annotated window, compatible with BirdCLEFDataset.
    """
    ann = pd.read_csv(annotation_csv)

    def _parse_time(t: object) -> int:
        s = str(t)
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return int(float(s))

    soundscape_dir = Path(soundscape_dir)
    rows = []

    for _, row in ann.iterrows():
        fname = str(row["filename"])
        primary = str(row["primary_label"])
        start_sec = _parse_time(row["start"]) if "start" in ann.columns else 0

        rows.append({
            "filename": fname,
            "filepath": str(soundscape_dir / fname),
            "primary_label": primary.split(";")[0].strip(),
            "secondary_labels": ";".join(primary.split(";")[1:]).strip(),
            "start_sec": start_sec,
            "fold": -1,
            "sample_weight": 1.0,
            "is_soundscape_window": True,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} soundscape annotation windows to {out_path}")
    return out_df


def get_submission_columns(sample_submission_path: str | Path) -> list[str]:
    """Return the 234 species columns in submission order (excludes row_id)."""
    sub = pd.read_csv(sample_submission_path, nrows=1)
    return sub.columns.tolist()[1:]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 data preparation")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_folds = subparsers.add_parser("folds", help="Build train_folds.csv")
    p_folds.add_argument("--train", default="birdclef-2026/train.csv")
    p_folds.add_argument("--out", default="data/train_folds.csv")

    p_mel = subparsers.add_parser("mel", help="Build mel cache")
    p_mel.add_argument("--folds", default="data/train_folds.csv")
    p_mel.add_argument("--audio-root", default="birdclef-2026/train_audio")
    p_mel.add_argument("--cache-root", default="data/mel_cache")
    p_mel.add_argument("--workers", type=int, default=8)

    p_sub = subparsers.add_parser("submission-cols", help="Print submission column order")
    p_sub.add_argument("--sample-submission", default="birdclef-2026/sample_submission.csv")

    args = parser.parse_args()

    if args.cmd == "folds":
        build_folds(args.train, args.out)
    elif args.cmd == "mel":
        build_mel_cache(args.folds, args.audio_root, args.cache_root, args.workers)
    elif args.cmd == "submission-cols":
        cols = get_submission_columns(args.sample_submission)
        print(f"{len(cols)} species columns")
        print(cols[:5], "...")
