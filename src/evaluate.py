"""
evaluate.py
-----------
Full test-set evaluation for CIFAKEDetector.
Produces:
  - Classification report (precision / recall / F1)
  - ROC-AUC score
  - Confusion matrix plot
  - ROC curve plot
  - Per-class accuracy bar chart
All artefacts are saved to outputs/results/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm
from dataset import CIFAKEDataset, build_val_transform, CLASS_NAMES
from model import CIFAKEDetector

sys.path.insert(0, str(Path(__file__).parent))


matplotlib.use("Agg")
# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":      "#0f0f0f",
    "surface": "#1a1a1a",
    "text":    "#e8e8e8",
    "accent":  "#00d4aa",
    "danger":  "#ff4d6d",
    "muted":   "#555555",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["muted"],
    "grid.alpha":        0.3,
    "font.family":       "monospace",
    "figure.dpi":        150,
})


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: CIFAKEDetector,
    test_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over the entire test split.

    Returns
    -------
    labels     : (N,) int array
    preds      : (N,) int array
    probs_pos  : (N,) float array -- P(REAL)
    """
    from torch.utils.data import DataLoader

    ds = CIFAKEDataset(
        root=test_dir,
        transform=build_val_transform(),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    for imgs, labels in tqdm(loader, desc="  Evaluating", ncols=80):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs[:, 1].cpu().numpy())   # P(REAL)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ---------------------------------------------------------------------------
# Plot: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        cm, interpolation="nearest",
        cmap="Greens",
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > thresh else PALETTE["text"],
            )

    ax.set_ylabel("True label", labelpad=10)
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=14, pad=12,
                 color=PALETTE["accent"])

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [+] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot: ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    labels: np.ndarray,
    probs_pos: np.ndarray,
    auc: float,
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs_pos)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        color=PALETTE["accent"],
        lw=2.0,
        label=f"ROC  (AUC = {auc:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        "--",
        color=PALETTE["muted"],
        lw=1.2,
        label="Random classifier",
    )

    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["accent"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate", labelpad=8)
    ax.set_title("Receiver Operating Characteristic", fontsize=13,
                 pad=12, color=PALETTE["accent"])
    ax.legend(loc="lower right", fontsize=10,
              framealpha=0.2, edgecolor=PALETTE["muted"])
    ax.grid(True, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [+] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot: Per-class accuracy
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
) -> None:
    accs = []
    for cls_idx in range(len(CLASS_NAMES)):
        mask = labels == cls_idx
        acc = (preds[mask] == labels[mask]).mean() if mask.sum() > 0 else 0.0
        accs.append(float(acc))

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [PALETTE["accent"], PALETTE["danger"]]
    bars = ax.bar(CLASS_NAMES, accs, color=colors, width=0.45, zorder=3)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", labelpad=8)
    ax.set_title("Per-class Accuracy", fontsize=13,
                 pad=12, color=PALETTE["accent"])
    ax.grid(axis="y", linestyle="--", zorder=0)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [+] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_dir: str = "data",
    output_dir: str = "outputs",
    batch_size: int = 64,
    num_workers: int = 4,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    results_path = output_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    ckpt = output_path / "best_model.pth"
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("        Run train.py first.")
        sys.exit(1)

    print("=" * 60)
    print(" CIFAKE Detector -- Evaluation")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {ckpt}")

    model = CIFAKEDetector.load(ckpt, device=device, pretrained=False)

    labels, preds, probs_pos = collect_predictions(
        model=model,
        test_dir=Path(data_dir) / "test",
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    # ---- Metrics ----------------------------------------------------
    print("\n[Classification Report]")
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report)

    report_path = results_path / "classification_report.txt"
    report_path.write_text(report)
    print(f"  [+] Saved: {report_path}")

    auc = roc_auc_score(labels, probs_pos)
    print(f"  ROC-AUC    : {auc:.6f}")

    overall_acc = (preds == labels).mean()
    print(f"  Overall Acc: {overall_acc:.6f}")

    # ---- Plots ------------------------------------------------------
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm,    results_path / "confusion_matrix.png")
    plot_roc_curve(labels, probs_pos, auc, results_path / "roc_curve.png")
    plot_per_class_accuracy(labels, preds, results_path / "per_class_acc.png")

    print("\n  All artefacts saved to:", results_path)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAKE Detector on test split"
    )
    parser.add_argument("--data_dir",    default="data")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
