"""
predict.py
----------
Single-image inference for CIFAKEDetector.
Can be used as a module or run directly from the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import build_predict_transform, CLASS_NAMES
from model import CIFAKEDetector

sys.path.insert(0, str(Path(__file__).parent))


class Predictor:
    """
    Thin wrapper around CIFAKEDetector for inference.

    Parameters
    ----------
    checkpoint : str | Path
        Path to best_model.pth.
    device : str | torch.device, optional
        'cpu', 'cuda', or 'mps'. Auto-detected if None.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.device = torch.device(device)
        self.transform = build_predict_transform()

        self.model = CIFAKEDetector.load(
            path=checkpoint,
            device=self.device,
            pretrained=False,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.

        Parameters
        ----------
        image : PIL.Image
            Input image (any mode; converted to RGB internally).

        Returns
        -------
        dict with keys:
            label       : str   -- predicted class name
            label_index : int   -- 0 or 1
            confidence  : float -- confidence in predicted class [0, 1]
            prob_fake   : float -- probability of FAKE
            prob_real   : float -- probability of REAL
        """
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        prob_fake = float(probs[CLASS_NAMES.index("FAKE")])
        prob_real = float(probs[CLASS_NAMES.index("REAL")])
        idx = int(probs.argmax())

        return {
            "label":       CLASS_NAMES[idx],
            "label_index": idx,
            "confidence":  float(probs[idx]),
            "prob_fake":   prob_fake,
            "prob_real":   prob_real,
        }

    @torch.no_grad()
    def predict_path(self, image_path: str | Path) -> dict:
        """
        Run inference given a file path.

        Parameters
        ----------
        image_path : str | Path

        Returns
        -------
        Same dict as predict_pil, with an additional 'path' key.
        """
        path = Path(image_path)
        result = self.predict_pil(Image.open(path))
        result["path"] = str(path)
        return result

    @torch.no_grad()
    def predict_batch(
        self,
        paths: list[str | Path],
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Run inference over a list of file paths.

        Parameters
        ----------
        paths : list[str | Path]
        batch_size : int

        Returns
        -------
        List of result dicts (same structure as predict_path).
        """
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        class _InlineDataset(TorchDataset):
            def __init__(self, file_paths, transform):
                self.paths = [Path(p) for p in file_paths]
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img), str(self.paths[idx])

        ds = _InlineDataset(paths, self.transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        results = []
        for tensors, file_paths in loader:
            tensors = tensors.to(self.device)
            logits = self.model(tensors)
            probs = F.softmax(logits, dim=1).cpu()

            for i, fp in enumerate(file_paths):
                p = probs[i]
                idx = int(p.argmax())
                results.append({
                    "path":        fp,
                    "label":       CLASS_NAMES[idx],
                    "label_index": idx,
                    "confidence":  float(p[idx]),
                    "prob_fake":   float(p[CLASS_NAMES.index("FAKE")]),
                    "prob_real":   float(p[CLASS_NAMES.index("REAL")]),
                })

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: dict) -> None:
    bar_len = 30
    conf_bar = "#" * int(result["confidence"] * bar_len)
    conf_bar = conf_bar.ljust(bar_len, "-")

    label = result["label"]
    icon = "[REAL]" if label == "REAL" else "[FAKE]"

    print("-" * 50)
    print(f"  File       : {result.get('path', 'PIL image')}")
    print(f"  Verdict    : {icon}  {label}")
    print(f"  Confidence : [{conf_bar}] {result['confidence'] * 100:.2f}%")
    print(f"  P(FAKE)    : {result['prob_fake'] * 100:.2f}%")
    print(f"  P(REAL)    : {result['prob_real'] * 100:.2f}%")
    print("-" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CIFAKE Detector on one or more images"
    )
    parser.add_argument(
        "images",
        nargs="+",
        metavar="IMAGE",
        help="Path(s) to input image file(s)",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/best_model.pth",
        help="Path to model checkpoint (default: outputs/best_model.pth)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: cpu | cuda | mps (auto-detected if omitted)",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        print("        Run src/train.py first.")
        sys.exit(1)

    predictor = Predictor(checkpoint=ckpt, device=args.device)
    print(f"[predict.py] Loaded checkpoint from {ckpt}")
    print(f"[predict.py] Running on {predictor.device}")

    if len(args.images) == 1:
        result = predictor.predict_path(args.images[0])
        _print_result(result)
    else:
        results = predictor.predict_batch(args.images)
        for r in results:
            _print_result(r)


if __name__ == "__main__":
    main()
