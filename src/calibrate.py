"""Post-hoc calibration and diagnostics for BirdCLEF+ 2026."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def analyze_per_species_performance(
    soundscape_val_auc_csv: str | Path,
    taxonomy_csv: str | Path,
    output_dir: str | Path,
) -> dict:
    """Analyze per-species AUC distributions and failure buckets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    auc_df = pd.read_csv(soundscape_val_auc_csv)
    tax = pd.read_csv(taxonomy_csv)
    tax["primary_label"] = tax["primary_label"].astype(str)
    auc_df["species_code"] = auc_df["species_code"].astype(str)

    merged = auc_df.merge(
        tax[["primary_label", "class_name", "common_name"]],
        left_on="species_code",
        right_on="primary_label",
        how="left",
        suffixes=("", "_tax"),
    )
    if "class_name" not in merged.columns and "class_name_tax" in merged.columns:
        merged["class_name"] = merged["class_name_tax"]
    if "common_name" not in merged.columns and "common_name_tax" in merged.columns:
        merged["common_name"] = merged["common_name_tax"]
    merged["group"] = np.where(
        merged["species_code"].str.startswith("47158son"),
        "Insect sonotype",
        np.where(merged["class_name"] == "Aves", "Bird", "Non-bird"),
    )

    failed = merged[merged["auc"].fillna(0.0) < 0.55].copy()
    zero_clip_proxy = merged[merged["gt_positives"].fillna(0).astype(int) == 0].copy()

    n_failed_nonbird = len(failed[failed["group"] != "Bird"])
    nonbird_drag = 0.5 * (n_failed_nonbird / 234.0) * 0.5

    summary = {
        "n_species": int(len(merged)),
        "n_failed_auc_lt_055": int(len(failed)),
        "n_zero_clip_proxy": int(len(zero_clip_proxy)),
        "n_failed_nonbird": int(n_failed_nonbird),
        "nonbird_drag": float(nonbird_drag),
        "group_auc_means": merged.groupby("group")["auc"].mean().to_dict(),
    }

    failed_cols = ["species_code", "auc", "group", "class_name", "common_name", "gt_positives"]
    failed[failed_cols].sort_values("auc").to_csv(output_dir / "failed_species_auc_lt_055.csv", index=False)
    zero_clip_proxy[failed_cols].sort_values("species_code").to_csv(
        output_dir / "zero_clip_proxy_species.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(output_dir / "calibration_summary.csv", index=False)
    return summary


def compute_species_priors(annotation_csv: str | Path) -> dict[str, float]:
    """Compute per-species prevalence from annotated soundscape windows."""
    ann = pd.read_csv(annotation_csv)
    total_windows = len(ann)
    counts: dict[str, int] = {}
    for labels in ann["primary_label"].astype(str):
        for sp in labels.split(";"):
            sp = sp.strip()
            if not sp:
                continue
            counts[sp] = counts.get(sp, 0) + 1
    return {sp: cnt / total_windows for sp, cnt in counts.items()}


def apply_prior_fallback(
    submission_df: pd.DataFrame,
    failed_species_list: Iterable[str],
    priors_dict: dict[str, float],
) -> pd.DataFrame:
    """Replace predictions for failed species with fixed class priors."""
    out = submission_df.copy()
    for sp in failed_species_list:
        if sp in out.columns:
            out[sp] = float(priors_dict.get(sp, 0.0))
    return out


def run_isotonic_calibration(
    oof_preds_df: pd.DataFrame,
    oof_labels_df: pd.DataFrame,
    species_list: list[str],
    output_dir: str | Path = "outputs/calibrators",
) -> dict[str, str]:
    """Fit per-species isotonic regressors when positives >= 10."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for sp in species_list:
        if sp not in oof_preds_df.columns or sp not in oof_labels_df.columns:
            continue
        y_pred = oof_preds_df[sp].astype(float).to_numpy()
        y_true = oof_labels_df[sp].astype(int).to_numpy()
        if y_true.sum() < 10:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_pred, y_true)
        path = output_dir / f"{sp}.pkl"
        joblib.dump(iso, path)
        saved[sp] = str(path)
    return saved

