"""Training loop for BirdCLEF+ 2026."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data_prep import get_submission_columns
from src.dataset import BirdCLEFDataset, collate_fn
from src.model import BirdCLEFModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    backbone: str = "tf_efficientnet_b2"
    val_fold: int = 0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    pseudo_label_csv: str | None = None
    mel_cache_root: str = "data/mel_cache"
    folds_csv: str = "data/train_folds.csv"
    sample_submission_csv: str = "birdclef-2026/sample_submission.csv"
    taxonomy_csv: str = "birdclef-2026/taxonomy.csv"
    output_dir: str = "outputs"
    num_workers: int = 4
    patience: int = 5
    seed: int = 42
    mixup_p: float = 0.5
    mixup_alpha: float = 0.4
    cutmix_p: float = 0.3
    dropout: float = 0.3


def build_species_split(species_list: list[str], taxonomy_csv: str | Path) -> tuple[list[int], list[int]]:
    taxonomy = pd.read_csv(taxonomy_csv)
    species_col = "primary_label" if "primary_label" in taxonomy.columns else "species_code"
    class_lookup = dict(zip(taxonomy[species_col].astype(str), taxonomy["class_name"].astype(str)))

    bird_idxs, nonbird_idxs = [], []
    for idx, species in enumerate(species_list):
        if class_lookup.get(species) == "Aves":
            bird_idxs.append(idx)
        else:
            nonbird_idxs.append(idx)
    return bird_idxs, nonbird_idxs


def _rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_mix = lam * y + (1.0 - lam) * y[idx]
    return x_mix, y_mix


def apply_cutmix(x: torch.Tensor, y: torch.Tensor):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0), device=x.device)
    b, _, h, w = x.shape
    x1, y1, x2, y2 = _rand_bbox(w, h, lam)
    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    area = (x2 - x1) * (y2 - y1)
    lam_adj = 1.0 - (area / float(w * h))
    y_cut = lam_adj * y + (1.0 - lam_adj) * y[idx]
    return x_cut, y_cut


def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, class_indices: list[int] | None = None) -> float:
    if class_indices is None:
        class_indices = list(range(y_true.shape[1]))
    aucs = []
    for idx in class_indices:
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
) -> float:
    model.train()
    running_loss = 0.0

    for x, y, w in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        if np.random.rand() < config.mixup_p:
            x, y = apply_mixup(x, y, alpha=config.mixup_alpha)
        elif np.random.rand() < config.cutmix_p:
            x, y = apply_cutmix(x, y)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type in ("cuda", "mps"))):
            logits = model(x)
            loss_matrix = criterion(logits, y)
            sample_loss = loss_matrix.mean(dim=1)
            loss = (sample_loss * w).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    bird_idxs: list[int],
    nonbird_idxs: list[int],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Returns (metrics_dict, y_prob, y_true)."""
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []

    for x, y, w in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        logits = model(x)
        loss_matrix = criterion(logits, y)
        sample_loss = loss_matrix.mean(dim=1)
        loss = (sample_loss * w).mean()
        running_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets)

    y_prob = np.concatenate(all_probs, axis=0)
    # Convert smoothed labels back to hard binary targets for ROC-AUC computation.
    y_true = (np.concatenate(all_targets, axis=0) > 0.5).astype(np.int32)
    result = {
        "val_loss": running_loss / max(len(loader.dataset), 1),
        "val_macro_auc": compute_macro_auc(y_true, y_prob),
        "bird_macro_auc": compute_macro_auc(y_true, y_prob, bird_idxs),
        "nonbird_macro_auc": compute_macro_auc(y_true, y_prob, nonbird_idxs),
    }
    return result, y_prob, y_true


def _prepare_manifests(config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(config.folds_csv)
    train_df = manifest[manifest["fold"] != config.val_fold].copy()
    val_df = manifest[manifest["fold"] == config.val_fold].copy()
    train_df["is_pseudo_label"] = False
    val_df["is_pseudo_label"] = False

    if config.pseudo_label_csv:
        pseudo_df = pd.read_csv(config.pseudo_label_csv)
        if "is_pseudo_label" not in pseudo_df.columns:
            pseudo_df["is_pseudo_label"] = True
        if "sample_weight" not in pseudo_df.columns:
            pseudo_df["sample_weight"] = 1.0
        pseudo_df["sample_weight"] = pseudo_df["sample_weight"].astype(float) * 0.5
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True, sort=False)

    if "filepath" not in train_df.columns and "filename" in train_df.columns:
        train_df["filepath"] = train_df.apply(
            lambda r: str(Path(str(r["primary_label"])) / f"{Path(r['filename']).stem}.npy"),
            axis=1,
        )
    if "filepath" not in val_df.columns and "filename" in val_df.columns:
        val_df["filepath"] = val_df.apply(
            lambda r: str(Path(str(r["primary_label"])) / f"{Path(r['filename']).stem}.npy"),
            axis=1,
        )
    return train_df, val_df


def train_fold(config: TrainConfig) -> dict:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    print(f"Using device: {device}")

    species_list = get_submission_columns(config.sample_submission_csv)
    bird_idxs, nonbird_idxs = build_species_split(species_list, config.taxonomy_csv)
    train_df, val_df = _prepare_manifests(config)

    train_ds = BirdCLEFDataset(
        train_df,
        mel_cache_root=config.mel_cache_root,
        species_list=species_list,
        mode="train",
        pseudo_label_weight=1.0,
    )
    val_ds = BirdCLEFDataset(
        val_df,
        mel_cache_root=config.mel_cache_root,
        species_list=species_list,
        mode="val",
        pseudo_label_weight=1.0,
    )

    sampler = WeightedRandomSampler(
        weights=train_df["sample_weight"].fillna(1.0).astype(float).values,
        num_samples=len(train_df),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = BirdCLEFModel(
        backbone_name=config.backbone,
        num_classes=234,
        pretrained=True,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))  # MPS: float32 only

    best_auc = -np.inf
    best_epoch = -1
    best_y_prob: np.ndarray | None = None
    best_y_true: np.ndarray | None = None
    no_improve = 0
    logs = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_metrics, y_prob, y_true = validate_one_epoch(model, val_loader, criterion, device, bird_idxs, nonbird_idxs)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        logs.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
            f"| val_auc={val_metrics['val_macro_auc']:.4f} "
            f"| bird_auc={val_metrics['bird_macro_auc']:.4f} "
            f"| nonbird_auc={val_metrics['nonbird_macro_auc']:.4f}"
        )

        if val_metrics["val_macro_auc"] > best_auc:
            best_auc = val_metrics["val_macro_auc"]
            best_epoch = epoch
            best_y_prob = y_prob
            best_y_true = y_true
            no_improve = 0
            best_path = output_dir / f"fold{config.val_fold}_best.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_kwargs": {
                        "backbone_name": config.backbone,
                        "num_classes": 234,
                        "pretrained": False,
                        "dropout": config.dropout,
                    },
                    "config": asdict(config),
                    "epoch": epoch,
                    "best_auc": best_auc,
                },
                best_path,
            )
        else:
            no_improve += 1

        if no_improve >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch={best_epoch}.")
            break

    last_path = output_dir / f"fold{config.val_fold}_last.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_kwargs": {
                "backbone_name": config.backbone,
                "num_classes": 234,
                "pretrained": False,
                "dropout": config.dropout,
            },
            "config": asdict(config),
            "epoch": logs[-1]["epoch"] if logs else 0,
            "best_auc": best_auc,
        },
        last_path,
    )

    log_df = pd.DataFrame(logs)
    log_path = output_dir / f"fold{config.val_fold}_train_log.csv"
    log_df.to_csv(log_path, index=False)

    # Save OOF predictions from the best epoch for ensemble calibration.
    oof_dir = Path("data/oof")
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_probs_path = oof_dir / f"fold{config.val_fold}_probs.npy"
    oof_targets_path = oof_dir / f"fold{config.val_fold}_targets.npy"
    if best_y_prob is not None:
        np.save(oof_probs_path, best_y_prob.astype(np.float32))
        np.save(oof_targets_path, best_y_true.astype(np.float32))
        print(f"OOF predictions saved: {oof_probs_path} {best_y_prob.shape}")

    return {
        "best_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "best_path": str(output_dir / f"fold{config.val_fold}_best.pt"),
        "last_path": str(last_path),
        "log_path": str(log_path),
        "oof_probs_path": str(oof_probs_path),
        "oof_targets_path": str(oof_targets_path),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train BirdCLEF fold model")
    parser.add_argument("--backbone", default="tf_efficientnet_b2")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pseudo-label-csv", default=None)
    parser.add_argument("--mel-cache-root", default="data/mel_cache")
    parser.add_argument("--folds-csv", default="data/train_folds.csv")
    parser.add_argument("--sample-submission-csv", default="birdclef-2026/sample_submission.csv")
    parser.add_argument("--taxonomy-csv", default="birdclef-2026/taxonomy.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup-p", type=float, default=0.5)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-p", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    return TrainConfig(
        backbone=args.backbone,
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
        mixup_p=args.mixup_p,
        mixup_alpha=args.mixup_alpha,
        cutmix_p=args.cutmix_p,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = train_fold(cfg)
    print("Training complete:", result)
