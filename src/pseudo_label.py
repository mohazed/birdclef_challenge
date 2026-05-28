"""Pseudo-label generation pipeline for BirdCLEF+ 2026.

Teacher models infer on unannotated soundscapes, and windows that exceed
per-species confidence thresholds are saved as training data for the next round.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model import BirdCLEFModel

# ---------------------------------------------------------------------------
# Per-species confidence thresholds
# ---------------------------------------------------------------------------
BIRD_THRESHOLD = 0.50
FROG_THRESHOLD = 0.40
INSECT_SON_THRESHOLD = 0.30
OTHER_NONBIRD_THRESHOLD = 0.40

SR = 32_000
WINDOW_SEC = 5
WINDOWS_PER_FILE = 12  # 60 s / 5 s


def get_threshold(species_code: str, taxonomy_df: pd.DataFrame) -> float:
    """Return confidence threshold for a species code."""
    if species_code.startswith("47158son"):
        return INSECT_SON_THRESHOLD
    if species_code[0].isdigit():
        rows = taxonomy_df[taxonomy_df["primary_label"] == species_code]
        if len(rows) > 0 and rows["class_name"].values[0] == "Amphibia":
            return FROG_THRESHOLD
        return OTHER_NONBIRD_THRESHOLD
    return BIRD_THRESHOLD


def _build_threshold_vector(species_list: list[str], taxonomy_df: pd.DataFrame) -> np.ndarray:
    """Pre-build a (234,) threshold array for vectorised filtering."""
    return np.array(
        [get_threshold(s, taxonomy_df) for s in species_list], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Mel spectrogram helper — must be byte-identical to data_prep.py
# ---------------------------------------------------------------------------

def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Convert a 5-second (160 000-sample) mono waveform to (3, 224, 224) float32."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=16000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise to [-1, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max > mel_min:
        mel_norm = 2.0 * (mel_db - mel_min) / (mel_max - mel_min) - 1.0
    else:
        mel_norm = np.zeros_like(mel_db)
    mel_128 = cv2.resize(mel_norm.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR)
    mel_224 = cv2.resize(mel_128, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.stack([mel_224, mel_224, mel_224], axis=0).astype(np.float32)


def _extract_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Split a 60-second audio array into 12 × 5-second non-overlapping windows."""
    target_len = WINDOW_SEC * SR
    windows: list[np.ndarray] = []
    for i in range(WINDOWS_PER_FILE):
        start = i * target_len
        end = start + target_len
        chunk = audio[start:end]
        if len(chunk) < target_len:
            pad = np.zeros(target_len - len(chunk), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        windows.append(chunk)
    return windows


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def _load_models(model_checkpoints: Sequence[str | Path], device: str) -> list[BirdCLEFModel]:
    models = []
    for ckpt in model_checkpoints:
        m = BirdCLEFModel.load_from_checkpoint(ckpt, device=device)
        m.to(device)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def _infer_batch(models: list[BirdCLEFModel], mel_batch: np.ndarray, device: str) -> np.ndarray:
    """Run mel_batch through each model and return averaged sigmoid probabilities.

    mel_batch: (B, 3, 224, 224) float32
    returns: (B, 234) float32
    """
    tensor = torch.from_numpy(mel_batch).to(device)
    all_probs = []
    for model in models:
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.mean(all_probs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    model_checkpoints: Sequence[str | Path],
    soundscape_dir: str | Path,
    excluded_files: Sequence[str],
    species_list: list[str],
    taxonomy_csv: str | Path,
    output_csv: str | Path,
    round_num: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """Generate and save pseudo-labels for unannotated soundscapes.

    Processes soundscapes in batches of 48 windows (4 files × 12 windows) to
    keep memory usage bounded.  Saves accepted mel spectrograms to
    ``data/pseudo_mel_cache_r{round_num}/`` alongside the output CSV.

    Parameters
    ----------
    model_checkpoints:
        Paths to .pt checkpoint files (one per fold = teacher ensemble).
    soundscape_dir:
        Directory containing .ogg soundscape files.
    excluded_files:
        Filenames to skip (the 66 expert-annotated soundscapes).
    species_list:
        Ordered list of 234 species codes matching sample_submission columns.
    taxonomy_csv:
        Path to taxonomy.csv for threshold lookup.
    output_csv:
        Destination path for the pseudo-label manifest CSV.
    round_num:
        Pseudo-label round number (1, 2, …). Used for cache directory naming.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    DataFrame with the pseudo-label manifest.
    """
    soundscape_dir = Path(soundscape_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    taxonomy_df = pd.read_csv(taxonomy_csv)
    threshold_vec = _build_threshold_vector(species_list, taxonomy_df)

    # Mel cache directory for this round
    mel_cache_dir = output_csv.parent / f"pseudo_mel_cache_r{round_num}"
    mel_cache_dir.mkdir(parents=True, exist_ok=True)

    excluded_set = {Path(f).name for f in excluded_files}

    all_ogg = sorted(soundscape_dir.glob("*.ogg"))
    candidate_files = [f for f in all_ogg if f.name not in excluded_set]
    n_files = len(candidate_files)
    print(
        f"[Round {round_num}] {n_files} candidate soundscapes "
        f"({len(excluded_set)} annotated files excluded)"
    )
    if n_files == 0:
        print("No candidate files found — check soundscape_dir and excluded_files.")
        return pd.DataFrame()

    print(f"Loading {len(model_checkpoints)} teacher model(s) on {device} ...")
    models = _load_models(model_checkpoints, device)

    records: list[dict] = []
    batch_files: list[Path] = []
    batch_mels: list[np.ndarray] = []   # accumulate windows
    batch_meta: list[tuple[str, int]] = []  # (filename, window_start_sec)

    def _flush_batch() -> None:
        if not batch_mels:
            return
        mel_arr = np.stack(batch_mels, axis=0)  # (B, 3, 224, 224)
        probs = _infer_batch(models, mel_arr, device)  # (B, 234)
        for i, (fname, start_sec) in enumerate(batch_meta):
            p = probs[i]
            mask = p >= threshold_vec
            if not mask.any():
                continue
            labels = [species_list[j] for j in np.where(mask)[0]]
            npy_path = mel_cache_dir / f"{Path(fname).stem}_{start_sec}.npy"
            np.save(npy_path, mel_arr[i].astype(np.float16))
            records.append(
                {
                    "soundscape_filename": fname,
                    "window_start_sec": start_sec,
                    "npy_cache_path": str(npy_path),
                    "pseudo_labels": ";".join(labels),
                    "max_confidence": float(p.max()),
                    "round_num": round_num,
                }
            )
        batch_mels.clear()
        batch_meta.clear()

    milestone_pcts = {0.10, 0.25, 0.50}
    start_time = time.time()

    for file_idx, ogg_path in enumerate(tqdm(candidate_files, desc=f"Pseudo-label R{round_num}")):
        # Progress ETAs at milestone percentages
        pct_done = (file_idx + 1) / n_files
        for mp in sorted(milestone_pcts):
            if pct_done >= mp:
                elapsed = time.time() - start_time
                estimated_total = elapsed / pct_done
                remaining = estimated_total - elapsed
                print(
                    f"  {int(mp*100):3d}% complete | "
                    f"elapsed {elapsed/60:.1f}m | "
                    f"est remaining {remaining/60:.1f}m | "
                    f"accepted windows so far: {len(records)}"
                )
                milestone_pcts.discard(mp)

        try:
            audio, _ = librosa.load(ogg_path, sr=SR, mono=True, duration=60.0)
        except Exception as exc:
            tqdm.write(f"  WARN: failed to load {ogg_path.name}: {exc}")
            continue

        windows = _extract_windows(audio)
        for win_idx, chunk in enumerate(windows):
            start_sec = win_idx * WINDOW_SEC
            mel = compute_mel(chunk)
            batch_mels.append(mel)
            batch_meta.append((ogg_path.name, start_sec))

            # Flush when batch reaches 48 windows (≈ 4 files)
            if len(batch_mels) >= 48:
                _flush_batch()

    _flush_batch()  # final partial batch

    elapsed_total = time.time() - start_time
    print(
        f"\n[Round {round_num}] Done in {elapsed_total/60:.1f}m. "
        f"Accepted {len(records)} windows from {n_files} files."
    )

    if not records:
        print("WARNING: zero windows accepted — all predictions below threshold.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved pseudo-label manifest to {output_csv}")
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_stats(pseudo_label_csv: str | Path, taxonomy_csv: str | Path) -> None:
    """Print a diagnostic summary of a pseudo-label round.

    Reports:
    - Total windows accepted.
    - Per-species pseudo-label counts (top-20 and bottom-20).
    - Fraction of zero-clip species that now have at least one pseudo-label.
    """
    pl_df = pd.read_csv(pseudo_label_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)

    total_windows = len(pl_df)
    print(f"\n{'='*60}")
    print(f"Pseudo-label summary  (round {pl_df['round_num'].iloc[0]})")
    print(f"{'='*60}")
    print(f"Total accepted windows : {total_windows:,}")
    print(f"Unique soundscapes     : {pl_df['soundscape_filename'].nunique():,}")

    # Per-species counts
    species_counts: dict[str, int] = {}
    for labels_str in pl_df["pseudo_labels"]:
        for sp in str(labels_str).split(";"):
            sp = sp.strip()
            if sp:
                species_counts[sp] = species_counts.get(sp, 0) + 1

    counts_series = pd.Series(species_counts).sort_values(ascending=False)
    print(f"\nTop-20 species by pseudo-label count:")
    print(counts_series.head(20).to_string())
    print(f"\nBottom-20 species (lowest coverage):")
    print(counts_series.tail(20).to_string())

    # Zero-clip species coverage check
    # Species are zero-clip if they have no rows in train.csv — we approximate
    # by checking which taxonomy entries have no pseudo-label at all.
    pl_species = set(species_counts.keys())
    all_species = set(taxonomy_df["primary_label"].astype(str))
    uncovered = all_species - pl_species
    print(f"\nSpecies with zero pseudo-labels : {len(uncovered):,} / {len(all_species)}")
    if uncovered:
        uncov_df = taxonomy_df[taxonomy_df["primary_label"].astype(str).isin(uncovered)]
        class_breakdown = uncov_df["class_name"].value_counts()
        print("  Class breakdown of uncovered species:")
        print(class_breakdown.to_string(header=False))

    # Insect sonotype coverage
    son_covered = [s for s in species_counts if s.startswith("47158son")]
    son_total = sum(1 for s in all_species if s.startswith("47158son"))
    print(f"\nInsect sonotypes covered : {len(son_covered)} / {son_total}")
    for s in sorted(s for s in all_species if s.startswith("47158son")):
        cnt = species_counts.get(s, 0)
        print(f"  {s:20s} : {cnt:,}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 pseudo-label generation")
    subparsers = parser.add_subparsers(dest="command")

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Run pseudo-label inference")
    gen.add_argument("--checkpoints", nargs="+", required=True, help="Model checkpoint paths")
    gen.add_argument("--soundscape-dir", required=True)
    gen.add_argument("--annotation-csv", required=True, help="train_soundscapes_labels.csv (to exclude annotated files)")
    gen.add_argument("--sample-submission", required=True, help="sample_submission.csv for species order")
    gen.add_argument("--taxonomy-csv", required=True)
    gen.add_argument("--output-csv", required=True)
    gen.add_argument("--round", type=int, required=True, dest="round_num")
    gen.add_argument("--device", default="cpu")

    # --- stats ---
    stats = subparsers.add_parser("stats", help="Print summary stats for a pseudo-label CSV")
    stats.add_argument("--pseudo-label-csv", required=True)
    stats.add_argument("--taxonomy-csv", required=True)

    args = parser.parse_args()

    if args.command == "generate":
        annot_df = pd.read_csv(args.annotation_csv)
        excluded = annot_df["filename"].unique().tolist()

        sub_df = pd.read_csv(args.sample_submission)
        species_list = list(sub_df.columns[1:])
        assert len(species_list) == 234, f"Expected 234 species, got {len(species_list)}"

        generate_pseudo_labels(
            model_checkpoints=args.checkpoints,
            soundscape_dir=args.soundscape_dir,
            excluded_files=excluded,
            species_list=species_list,
            taxonomy_csv=args.taxonomy_csv,
            output_csv=args.output_csv,
            round_num=args.round_num,
            device=args.device,
        )

    elif args.command == "stats":
        summary_stats(args.pseudo_label_csv, args.taxonomy_csv)

    else:
        parser.print_help()
