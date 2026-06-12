"""
model.py
--------
EfficientNetB0 backbone with a custom classification head
for binary Real vs AI-Generated image detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ---------------------------------------------------------------------------
# Custom head
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """
    Replacement for EfficientNet's default linear head.

    Architecture:
        AdaptiveAvgPool -> Flatten
        -> BN -> Dropout -> Linear(1280, 512) -> GELU
        -> BN -> Dropout -> Linear(512, 128)  -> GELU
        -> Linear(128, num_classes)

    Parameters
    ----------
    in_features : int
        Feature dimension from the backbone (1280 for B0).
    num_classes : int
        Output dimension (2 for binary, but we keep it flexible).
    dropout_rate : float
        Dropout probability applied before each linear layer.
    """

    def __init__(
        self,
        in_features: int = 1280,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512, bias=False),
            nn.GELU(),

            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128, bias=False),
            nn.GELU(),

            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CIFAKEDetector(nn.Module):
    """
    EfficientNetB0 + custom head for CIFAKE binary classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 2).
    pretrained : bool
        If True, load ImageNet weights for the backbone.
    freeze_backbone : bool
        If True, freeze all backbone parameters during phase-1 training.
    dropout_rate : float
        Dropout rate passed to ClassifierHead.
    """

    BACKBONE_OUT_FEATURES = 1280

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)

        # Strip the original classifier; keep only the feature extractor
        self.backbone = efficientnet.features

        self.classifier = ClassifierHead(
            in_features=self.BACKBONE_OUT_FEATURES,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        if freeze_backbone:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)        # (B, 1280, 7, 7)
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (phase-1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_from_block: int = 4) -> None:
        """
        Progressively unfreeze backbone starting from a given MBConv block.

        Parameters
        ----------
        unfreeze_from_block : int
            All MBConv blocks at index >= this value will be unfrozen.
            Blocks are indexed 0-8 in EfficientNetB0.features.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        for idx in range(unfreeze_from_block, len(self.backbone)):
            for param in self.backbone[idx].parameters():
                param.requires_grad = True

    def param_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
    ) -> list[dict]:
        """
        Return parameter groups with separate learning rates
        for use with optimizers that support per-group LR.
        """
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = list(self.classifier.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        groups.append({"params": head_params, "lr": head_lr})
        return groups

    def count_parameters(self) -> dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {"total": total, "trainable": trainable}

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model state dict."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "CIFAKEDetector":
        """
        Instantiate model and load saved weights.

        Parameters
        ----------
        path : str | Path
            Path to .pth checkpoint.
        device : torch.device, optional
            Target device. Defaults to CPU.
        **kwargs
            Extra keyword arguments forwarded to __init__.
        """
        if device is None:
            device = torch.device("cpu")
        model = cls(**kwargs)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAKEDetector(pretrained=True).to(device)

    dummy = torch.randn(4, 3, 224, 224).to(device)
    out = model(dummy)

    stats = model.count_parameters()
    print(f"[model.py] Output shape    : {out.shape}")
    print(f"[model.py] Total params    : {stats['total']:,}")
    print(f"[model.py] Trainable params: {stats['trainable']:,}")
    print(f"[model.py] Device          : {device}")
