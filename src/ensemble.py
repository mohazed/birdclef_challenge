"""
Ensemble and post-processing module for BirdCLEF+ 2026.
All numpy + onnxruntime only — no PyTorch.
"""

import time
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import cv2


# ---------------------------------------------------------------------------
# Audio / mel helpers
# ---------------------------------------------------------------------------

def compute_mel(audio_5s: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Convert a 5-second mono waveform to a (3, 224, 224) float32 mel tensor.
    Must match training preprocessing exactly.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_5s, sr=sr,
        n_fft=2048, hop_length=512, n_mels=128,
        fmin=20, fmax=16000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # normalize to [-1, 1]
    mel_db = mel_db / 80.0  # librosa power_to_db range is roughly [-80, 0]
    mel_db = np.clip(mel_db, -1.0, 1.0)
    # resize to (128, 224) then stack 3 channels
    mel_resized = cv2.resize(mel_db, (224, 224))
    mel_3ch = np.stack([mel_resized, mel_resized, mel_resized], axis=0)  # (3, 224, 224)
    return mel_3ch.astype(np.float32)


def _extract_windows(audio: np.ndarray, sr: int = 32000, window_sec: int = 5) -> list:
    """Split audio into non-overlapping 5-second windows."""
    window_samples = sr * window_sec
    total_windows = int(np.floor(len(audio) / window_samples))
    windows = []
    for i in range(total_windows):
        start = i * window_samples
        end = start + window_samples
        windows.append(audio[start:end])
    # handle short file — pad last window if remaining audio is > 0
    remainder = audio[total_windows * window_samples:]
    if len(remainder) > 0:
        padded = np.zeros(window_samples, dtype=audio.dtype)
        padded[:len(remainder)] = remainder
        windows.append(padded)
    return windows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_onnx_models(model_paths: list) -> list:
    """
    Load a list of ONNX model paths; return InferenceSession objects (CPU).
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sessions = []
    for p in model_paths:
        sess = ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
        sessions.append(sess)
    return sessions


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def infer_batch(sessions: list, mel_batch: np.ndarray) -> np.ndarray:
    """
    Run mel_batch through each session, average sigmoid outputs.

    mel_batch : (B, 3, 224, 224) float32 numpy
    returns   : (B, 234) float32 numpy probabilities
    """
    all_probs = []
    for sess in sessions:
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: mel_batch})[0]  # (B, 234)
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.append(probs)
    return np.mean(all_probs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def temporal_smooth(preds_per_file: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply a rolling mean of size `window` along the time axis (axis=0).

    preds_per_file : (n_windows, 234)
    returns        : (n_windows, 234) smoothed
    """
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=preds_per_file,
    )
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Soundscape-level prediction
# ---------------------------------------------------------------------------

def predict_soundscape(
    sessions: list,
    audio_path: str,
    species_list: list,
    smooth: bool = True,
    batch_size: int = 16,
) -> dict:
    """
    Predict species probabilities for every 5-second window in a soundscape.

    Returns dict: row_id (str) -> 234-dim probability array
    Row ID format: "{stem}_{start_sec}"
    """
    audio, sr = librosa.load(audio_path, sr=32000, mono=True)
    windows = _extract_windows(audio, sr=sr, window_sec=5)
    n_windows = len(windows)

    mel_list = [compute_mel(w, sr=sr) for w in windows]

    all_probs = []
    for i in range(0, n_windows, batch_size):
        batch = np.stack(mel_list[i : i + batch_size], axis=0)  # (B, 3, 224, 224)
        probs = infer_batch(sessions, batch)  # (B, 234)
        all_probs.append(probs)

    preds = np.concatenate(all_probs, axis=0)  # (n_windows, 234)

    if smooth and n_windows > 1:
        preds = temporal_smooth(preds, window=3)

    stem = Path(audio_path).stem
    result = {}
    for i, prob_vec in enumerate(preds):
        start_sec = i * 5
        row_id = f"{stem}_{start_sec}"
        result[row_id] = prob_vec

    return result


# ---------------------------------------------------------------------------
# Full submission builder
# ---------------------------------------------------------------------------

def build_submission(
    sessions: list,
    test_soundscape_dir: str,
    species_list: list,
    output_path: str,
    smooth: bool = True,
    batch_size: int = 16,
    sample_submission_path: str = None,
) -> pd.DataFrame:
    """
    Process all test soundscapes and build a submission CSV.

    Verifies column order against sample_submission.csv when provided.
    """
    test_dir = Path(test_soundscape_dir)
    ogg_files = sorted(test_dir.glob("*.ogg"))
    n_files = len(ogg_files)

    if n_files == 0:
        raise FileNotFoundError(f"No .ogg files found in {test_soundscape_dir}")

    # Timing estimate after first 5 files
    rows = {}
    t_start = time.time()

    for file_idx, ogg_path in enumerate(ogg_files):
        file_preds = predict_soundscape(
            sessions, str(ogg_path), species_list,
            smooth=smooth, batch_size=batch_size,
        )
        rows.update(file_preds)

        if file_idx == 4:
            elapsed = time.time() - t_start
            est_total = elapsed / 5 * n_files
            print(f"[timing] 5/{n_files} files done in {elapsed:.1f}s — "
                  f"estimated total {est_total/60:.1f} min")
            if est_total > 75 * 60:
                print("[WARNING] Projected time exceeds 75 min — consider reducing models")

        if (file_idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / (file_idx + 1) * (n_files - file_idx - 1)
            print(f"[progress] {file_idx + 1}/{n_files} | "
                  f"elapsed {elapsed/60:.1f}m | est remaining {remaining/60:.1f}m")

    # Build DataFrame
    sub = pd.DataFrame.from_dict(rows, orient="index", columns=species_list)
    sub.index.name = "row_id"
    sub = sub.reset_index()
    sub = sub.sort_values("row_id").reset_index(drop=True)

    # Verify column order against sample_submission if provided
    if sample_submission_path is not None:
        ref = pd.read_csv(sample_submission_path, nrows=0)
        ref_species = list(ref.columns[1:])
        assert ref_species == species_list, (
            "Column order mismatch with sample_submission.csv — check species_list source"
        )

    # Sanity checks
    assert sub.isna().sum().sum() == 0, "NaN values found in submission"
    probs = sub[species_list].values
    assert probs.min() >= 0.0 and probs.max() <= 1.0, "Probabilities out of [0, 1] range"

    sub.to_csv(output_path, index=False)

    elapsed_total = time.time() - t_start
    print(f"\n[done] {len(sub)} rows | shape {sub.shape} | "
          f"total time {elapsed_total/60:.1f} min")
    print(f"[stats] mean prob: {probs.mean():.4f} | "
          f"max prob: {probs.max():.4f} | min prob: {probs.min():.4f}")
    top_species = sub[species_list].mean().nlargest(10)
    print("[top-10 most active species by mean prediction]")
    print(top_species.to_string())

    return sub


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BirdCLEF+ 2026 ensemble inference")
    parser.add_argument("--models", nargs="+", required=True, help="ONNX model paths")
    parser.add_argument("--test-dir", required=True, help="Directory with test .ogg files")
    parser.add_argument("--sample-submission", required=True, help="sample_submission.csv path")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--no-smooth", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ref = pd.read_csv(args.sample_submission, nrows=0)
    species = list(ref.columns[1:])
    assert len(species) == 234, f"Expected 234 species, got {len(species)}"

    print(f"Loading {len(args.models)} ONNX model(s)...")
    sessions = load_onnx_models(args.models)

    # Warm-up run
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    _ = infer_batch(sessions, dummy)
    print("Models loaded and warmed up.")

    build_submission(
        sessions=sessions,
        test_soundscape_dir=args.test_dir,
        species_list=species,
        output_path=args.output,
        smooth=not args.no_smooth,
        batch_size=args.batch_size,
        sample_submission_path=args.sample_submission,
    )
