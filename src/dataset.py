"""
dataset.py
----------
Data loading, preprocessing, and augmentation pipeline
for the CIFAKE Real vs AI-Generated image classification task.
"""

from pathlib import Path
from typing import Tuple, Optional


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics
STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224   # EfficientNetB0 native input resolution

CLASS_NAMES = ["FAKE", "REAL"]   # sorted alphabetically -> index 0 / 1


# ---------------------------------------------------------------------------
# Augmentation factories
# ---------------------------------------------------------------------------

def build_train_transform() -> transforms.Compose:
    """
    Aggressive augmentation suite for training.
    Designed to improve generalisation on unseen generative artefacts.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
        ),
    ])


def build_val_transform() -> transforms.Compose:
    """
    Deterministic preprocessing for validation and test splits.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def build_predict_transform() -> transforms.Compose:
    """
    Preprocessing for single-image inference.
    Identical to validation but returned separately for clarity.
    """
    return build_val_transform()


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CIFAKEDataset(Dataset):
    """
    ImageFolder-style dataset for CIFAKE.

    Directory layout expected:
        root/
          FAKE/  <- label 0
          REAL/  <- label 1

    Parameters
    ----------
    root : str | Path
        Path to the split directory (e.g. data/train).
    transform : transforms.Compose, optional
        Torchvision transform pipeline.
    max_samples : int, optional
        Cap on total samples (useful for quick smoke-tests).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        self._load_samples(max_samples)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self, max_samples: Optional[int]) -> None:
        """Walk class sub-directories and collect (path, label) pairs."""
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected class directory not found: {cls_dir}"
                )
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in valid_ext:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                "Verify that FAKE/ and REAL/ sub-folders are populated."
            )

        if max_samples is not None:
            # Stratified truncation: keep equal ratio from each class
            from collections import defaultdict
            import random

            buckets: dict[int, list] = defaultdict(list)
            for item in self.samples:
                buckets[item[1]].append(item)

            per_class = max_samples // len(CLASS_NAMES)
            self.samples = []
            for items in buckets.values():
                random.shuffle(items)
                self.samples.extend(items[:per_class])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def class_counts(self) -> dict[str, int]:
        """Return per-class sample counts."""
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        return {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}

    def class_weights(self) -> list[float]:
        """
        Inverse-frequency weights for WeightedRandomSampler.
        Returns a weight per sample in the same order as self.samples.
        """
        counts = self.class_counts()
        total = len(self.samples)
        freq = {cls: cnt / total for cls, cnt in counts.items()}
        weights = [
            1.0 / freq[CLASS_NAMES[lbl]]
            for _, lbl in self.samples
        ]
        return weights


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct train and validation DataLoaders.

    Parameters
    ----------
    data_dir : str | Path
        Root data directory containing train/ and test/ sub-folders.
    batch_size : int
        Mini-batch size for both loaders.
    num_workers : int
        Parallel workers for data loading.
    max_train_samples : int, optional
        Truncate training set (for quick iteration).
    max_val_samples : int, optional
        Truncate validation set.
    use_weighted_sampler : bool
        If True, use WeightedRandomSampler to handle class imbalance.

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    data_dir = Path(data_dir)

    train_ds = CIFAKEDataset(
        root=data_dir / "train",
        transform=build_train_transform(),
        max_samples=max_train_samples,
    )
    val_ds = CIFAKEDataset(
        root=data_dir / "test",
        transform=build_val_transform(),
        max_samples=max_val_samples,
    )

    if use_weighted_sampler:
        weights = torch.DoubleTensor(train_ds.class_weights())
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_path = Path("data")
    if not data_path.exists():
        print("[ERROR] 'data/' directory not found. Download the dataset.")
        sys.exit(1)

    train_loader, val_loader = build_dataloaders(
        data_dir=data_path,
        batch_size=32,
        num_workers=0,
    )

    imgs, labels = next(iter(train_loader))
    print(f"[dataset.py] Train batches : {len(train_loader)}")
    print(f"[dataset.py] Val   batches : {len(val_loader)}")
    print(f"[dataset.py] Batch shape   : {imgs.shape}")
    print(f"[dataset.py] Label sample  : {labels[:8].tolist()}")
