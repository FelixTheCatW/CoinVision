"""Ensembling and flip-free test-time augmentation for evaluation.

No training happens here: several checkpoints vote on the same images and each image is seen
through a few deterministic views. A horizontal flip is deliberately absent - mirroring turns the
date into text the model never saw - so the views are the frame, a tighter centre crop and a
brighter copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import v2

from .dataset import IMAGENET_MEAN, IMAGENET_STD, read_manifest
from .infer import CoinPredictor
from .train import last_digit_report


class _Photometric(nn.Module):
    """Brightness/contrast jitter applied as a fixed view (no randomness)."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = v2.functional.adjust_brightness(tensor, 1.15)
        return v2.functional.adjust_contrast(tensor, 1.08)


def build_eval_views(image_size: tuple[int, int], count: int = 3) -> list[v2.Compose]:
    """Deterministic views; ``count`` of 1 means plain evaluation."""
    if count < 1:
        raise ValueError("at least one view is required")
    width, height = int(image_size[0]), int(image_size[1])
    resize = v2.Resize((height, width), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    views = [v2.Compose([v2.ToImage(), resize, to_float, normalize])]
    if count >= 2:
        # central 85% of the frame, same output size: catches the coin when it fills the photo
        views.append(
            v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize((int(height * 1.18), int(width * 1.18)), antialias=True),
                    v2.CenterCrop((height, width)),
                    to_float,
                    normalize,
                ]
            )
        )
    if count >= 3:
        views.append(
            v2.Compose([v2.ToImage(), resize, to_float, _Photometric(), normalize])
        )
    return views[:count]


def _probs_for_view(
    models: Sequence[nn.Module],
    paths: Sequence[Path],
    transform: v2.Compose,
    device: torch.device,
    image_size: tuple[int, int],
    batch_size: int = 16,
) -> torch.Tensor:
    """Mean softmax over models for one view; returns [N, C] on CPU."""
    from PIL import Image

    summed: torch.Tensor | None = None
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        tensors = []
        for path in chunk:
            with Image.open(path) as handle:
                tensors.append(transform(handle.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        batch_probs: torch.Tensor | None = None
        for model in models:
            with torch.no_grad():
                logits = model(batch)[:, -1]
            probs = logits.softmax(dim=-1).cpu()
            batch_probs = probs if batch_probs is None else batch_probs + probs
        assert batch_probs is not None
        summed = batch_probs if summed is None else torch.cat([summed, batch_probs])

    if summed is None:
        raise ValueError("no images to evaluate")
    return summed / len(models)


def evaluate_ensemble(
    checkpoints: Sequence[str | Path],
    manifest: str | Path,
    image_size: tuple[int, int] = (448, 224),
    views: int = 1,
    device_spec: str = "auto",
    num_classes: int = 10,
) -> dict:
    """Last-digit report for the average of several checkpoints over several views."""
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")

    predictors = [CoinPredictor.from_checkpoint(path, device_spec) for path in checkpoints]
    device = predictors[0].device
    models = [predictor.model for predictor in predictors]

    rows = read_manifest(manifest)
    paths = [path for path, _ in rows]
    targets = [int(year[3]) for _, year in rows]

    summed: torch.Tensor | None = None
    for transform in build_eval_views(image_size, views):
        probs = _probs_for_view(models, paths, transform, device, image_size)
        summed = probs if summed is None else summed + probs
    assert summed is not None

    report = last_digit_report(targets, summed.argmax(dim=-1).tolist(), num_classes)
    report["checkpoints"] = [str(Path(path)) for path in checkpoints]
    report["views"] = views
    return report
