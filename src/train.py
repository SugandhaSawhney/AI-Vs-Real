"""
train.py
--------
Full training loop for CIFAKEDetector with:
  - Two-phase training (frozen backbone -> fine-tune)
  - OneCycleLR scheduler
  - Label smoothing cross-entropy
  - Mixed precision (AMP)
  - Best-model checkpointing
  - CSV + console logging
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from dataset import build_dataloaders, CLASS_NAMES
from model import CIFAKEDetector
# Make src importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))


class TrainConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    batch_size: int = 64
    num_workers: int = 0
    phase1_epochs: int = 5     # frozen backbone
    phase2_epochs: int = 15    # fine-tune top blocks
    head_lr: float = 1e-3
    backbone_lr: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    unfreeze_from_block: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: CIFAKEDetector,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: torch.device,
    phase: str,
    grad_clip: float,
) -> tuple[float, float]:
    """
    Run one forward (+ optionally backward) pass over a DataLoader.

    Returns
    -------
    avg_loss : float
    avg_acc  : float
    """
    is_train = phase == "train"
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  {phase:>5}", leave=False, ncols=90)

    with torch.set_grad_enabled(is_train):
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type,
                          enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            batch_acc = accuracy(logits.detach(), labels)
            total_loss += loss.item()
            total_acc += batch_acc
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_acc:.4f}",
            )

    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def train_phase(
    model: CIFAKEDetector,
    train_loader,
    val_loader,
    epochs: int,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    label_smoothing: float,
    grad_clip: float,
    device: torch.device,
    output_dir: Path,
    csv_writer,
    phase_name: str,
    best_val_acc: list,          # mutable container so caller sees update
) -> None:
    param_groups = model.param_groups(
        backbone_lr=backbone_lr, head_lr=head_lr
    )
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in param_groups],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(
            f"\n[{phase_name}] Epoch {epoch}/{epochs}"
        )

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, scaler,
            device, "train", grad_clip,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion,
            None, None, scaler,
            device, "val", grad_clip,
        )

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[-1]["lr"]

        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f}  val_acc={val_acc:.4f} "
            f"| lr={lr_now:.2e}  [{elapsed:.0f}s]"
        )

        csv_writer.writerow({
            "phase":      phase_name,
            "epoch":      epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc":  f"{train_acc:.6f}",
            "val_loss":   f"{val_loss:.6f}",
            "val_acc":    f"{val_acc:.6f}",
            "lr":         f"{lr_now:.2e}",
        })

        if val_acc > best_val_acc[0]:
            best_val_acc[0] = val_acc
            ckpt_path = output_dir / "best_model.pth"
            model.save(ckpt_path)
            print(f"  [+] New best val_acc={val_acc:.4f} -> saved {ckpt_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()
    output_dir = Path(cfg.output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" CIFAKE Detector -- Training")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Data dir     : {cfg.data_dir}")
    print(f"  Batch size   : {cfg.batch_size}")
    print(f"  Phase 1 ep   : {cfg.phase1_epochs}")
    print(f"  Phase 2 ep   : {cfg.phase2_epochs}")
    print("=" * 60)

    # ---- Data -------------------------------------------------------
    train_loader, val_loader = build_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_weighted_sampler=True,
    )
    print(
        f"  Train batches: {len(train_loader)} "
        f"| Val batches: {len(val_loader)}"
    )

    # ---- Model ------------------------------------------------------
    model = CIFAKEDetector(
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    stats = model.count_parameters()
    print(
        f"  Params total={stats['total']:,} "
        f"trainable={stats['trainable']:,}"
    )

    # ---- CSV log ----------------------------------------------------
    log_path = output_dir / "results" / "training_log.csv"
    fieldnames = [
        "phase", "epoch",
        "train_loss", "train_acc",
        "val_loss", "val_acc", "lr",
    ]
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    best_val_acc = [0.0]   # mutable container

    # ---- Phase 1: train head only -----------------------------------
    print("\n[*] Phase 1 -- Backbone frozen, training head only")
    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.phase1_epochs,
        head_lr=cfg.head_lr,
        backbone_lr=0.0,
        weight_decay=cfg.weight_decay,
        label_smoothing=cfg.label_smoothing,
        grad_clip=cfg.grad_clip,
        device=device,
        output_dir=output_dir,
        csv_writer=csv_writer,
        phase_name="phase1",
        best_val_acc=best_val_acc,
    )

    # ---- Phase 2: fine-tune top blocks ------------------------------
    model.unfreeze_backbone(
        unfreeze_from_block=cfg.unfreeze_from_block
    )
    stats = model.count_parameters()
    print(
        f"\n[*] Phase 2 -- Partial unfreeze (block>={cfg.unfreeze_from_block})"
        f"  trainable={stats['trainable']:,}"
    )
    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.phase2_epochs,
        head_lr=cfg.head_lr,
        backbone_lr=cfg.backbone_lr,
        weight_decay=cfg.weight_decay,
        label_smoothing=cfg.label_smoothing,
        grad_clip=cfg.grad_clip,
        device=device,
        output_dir=output_dir,
        csv_writer=csv_writer,
        phase_name="phase2",
        best_val_acc=best_val_acc,
    )

    log_file.close()

    print("\n" + "=" * 60)
    print(f" Training complete. Best val_acc = {best_val_acc[0]:.4f}")
    print(f" Checkpoint : {output_dir / 'best_model.pth'}")
    print(f" Log        : {log_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CIFAKE Real vs AI-Generated Detector"
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Root data directory (default: data/)"
    )
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Output directory for weights and logs (default: outputs/)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64
    )
    parser.add_argument(
        "--num_workers", type=int, default=4
    )
    parser.add_argument(
        "--phase1_epochs", type=int, default=5
    )
    parser.add_argument(
        "--phase2_epochs", type=int, default=15
    )
    parser.add_argument(
        "--head_lr", type=float, default=1e-3
    )
    parser.add_argument(
        "--backbone_lr", type=float, default=5e-5
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.phase1_epochs = args.phase1_epochs
    cfg.phase2_epochs = args.phase2_epochs
    cfg.head_lr = args.head_lr
    cfg.backbone_lr = args.backbone_lr
    cfg.seed = args.seed

    main(cfg)
