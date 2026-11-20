from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .cache import ImageCache

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Colour-mode specific statistics. The grey values are the luma-weighted equivalents of the
# ImageNet numbers above; the binary mode only ever produces 0 and 1, so it maps them to +-1.
GRAY_MEAN = (0.449, 0.449, 0.449)
GRAY_STD = (0.226, 0.226, 0.226)
BINARY_MEAN = (0.5, 0.5, 0.5)
BINARY_STD = (0.5, 0.5, 0.5)

COLOR_MODES = ("rgb", "gray", "binary")


class Binarize(torch.nn.Module):
    """Otsu threshold of a single-channel float image, expanded back to three identical channels.

    The threshold is computed per image, because exposure varies a lot between listings: a fixed
    cut-off blows out the highlights on the bright coins and turns the dark ones into one blob.
    """

    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        self.bins = bins

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=0, keepdim=True) if image.shape[0] > 1 else image
        threshold = _otsu_threshold(gray, self.bins)
        mask = (gray >= threshold).to(image.dtype)
        return mask.expand(3, *mask.shape[-2:])


def _otsu_threshold(gray: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Between-class variance maximisation over a histogram of the values in [0, 1]."""
    values = gray.flatten().to(torch.float32)
    histogram = torch.histc(values, bins=bins, min=0.0, max=1.0)
    centers = (torch.arange(bins, dtype=histogram.dtype) + 0.5) / bins
    cumulative_weight = torch.cumsum(histogram, 0)
    total_weight = cumulative_weight[-1]
    if float(total_weight) <= 0.0:
        return torch.tensor(0.5, dtype=torch.float32)

    cumulative_energy = torch.cumsum(histogram * centers, 0)
    total_energy = cumulative_energy[-1]
    background_weight = cumulative_weight
    foreground_weight = total_weight - cumulative_weight
    # Pixels at or below the bin, pixels above it: the standard two-class split.
    background_mean = cumulative_energy / background_weight.clamp_min(1e-6)
    foreground_mean = (total_energy - cumulative_energy) / foreground_weight.clamp_min(1e-6)
    between = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
    valid = (background_weight > 0) & (foreground_weight > 0)
    between = torch.where(valid, between, torch.zeros_like(between))
    best = int(torch.argmax(between))
    # The upper edge of the winning bin, so pixels that voted for the background bin stay in it:
    # using the bin centre leaves values just above the centre on the bright side of the cut.
    return torch.tensor(float(best + 1) / bins, dtype=torch.float32)


def normalization_for(color_mode: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if color_mode == "rgb":
        return IMAGENET_MEAN, IMAGENET_STD
    if color_mode == "gray":
        return GRAY_MEAN, GRAY_STD
    if color_mode == "binary":
        return BINARY_MEAN, BINARY_STD
    raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}")


def _color_ops(color_mode: str) -> list:
    """Channels stay at three so the ImageNet-pretrained stem keeps working unchanged."""
    if color_mode == "rgb":
        return []
    if color_mode == "gray":
        return [v2.Grayscale(num_output_channels=3)]
    if color_mode == "binary":
        return [v2.Grayscale(num_output_channels=1), Binarize()]
    raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {color_mode!r}")



def year_to_digits(year: str | int, num_digits: int = 4) -> list[int]:
    """Turn a year label such as 1832 into per-position digit targets [1, 8, 3, 2]."""
    text = str(year).strip()
    if len(text) != num_digits or not text.isdigit():
        raise ValueError(f"year must be exactly {num_digits} digits, got {year!r}")
    return [int(char) for char in text]


def digits_to_year(digits: Sequence[int]) -> str:
    return "".join(str(int(digit)) for digit in digits)


def _augment_ops() -> list:
    """Training-time jitter. No horizontal flip: mirroring would turn the date into text the
    digit heads never see at inference time."""
    return [
        v2.RandomAffine(degrees=(-10.0, 10.0), translate=(0.04, 0.06), scale=(0.9, 1.1), shear=3.0),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.03),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.25),
    ]


def build_transform(
    image_size: tuple[int, int], augment: bool = False, color_mode: str = "rgb"
) -> v2.Compose:
    """image_size is (width, height) as in the config; torchvision wants (height, width)."""
    width, height = image_size
    ops: list = [v2.ToImage(), v2.Resize((int(height), int(width)), antialias=True)]
    if augment:
        ops.extend(_augment_ops())
    ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.extend(_color_ops(color_mode))
    mean, std = normalization_for(color_mode)
    ops.append(v2.Normalize(mean, std))
    return v2.Compose(ops)


def build_cached_transform(augment: bool = False, color_mode: str = "rgb") -> v2.Compose:
    """Same pipeline as build_transform, for pixels ImageCache already decoded and resized."""
    ops: list = [] if not augment else list(_augment_ops())
    ops.append(v2.ToDtype(torch.float32, scale=True))
    ops.extend(_color_ops(color_mode))
    mean, std = normalization_for(color_mode)
    ops.append(v2.Normalize(mean, std))
    return v2.Compose(ops)


def read_manifest(path: str | Path) -> list[tuple[Path, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [(Path(row["image"]), row["year"]) for row in csv.DictReader(fh)]


class CoinDataset(Dataset):
    """Coin images with one digit label per position of the year."""

    def __init__(
        self,
        manifest: str | Path,
        image_size: tuple[int, int] = (448, 224),
        num_digits: int = 4,
        transform: v2.Compose | None = None,
        augment: bool = False,
        color_mode: str = "rgb",
    ) -> None:
        self.manifest = Path(manifest)
        self.num_digits = num_digits
        self.transform = (
            transform
            if transform is not None
            else build_transform(image_size, augment=augment, color_mode=color_mode)
        )
        self.samples = [
            (path, year_to_digits(year, num_digits)) for path, year in read_manifest(self.manifest)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, digits = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(digits, dtype=torch.long)


class CachedCoinDataset(Dataset):
    """Same samples as CoinDataset, but the pixels come from an ImageCache instead of a decode.

    The cache is preloaded once per process, so every epoch and every split that shares an image
    set reuses the same tensors: the loader turns into little more than a dict lookup.
    """

    def __init__(
        self,
        manifest: str | Path,
        cache: ImageCache,
        num_digits: int = 4,
        transform: v2.Compose | None = None,
        augment: bool = False,
        color_mode: str = "rgb",
    ) -> None:
        self.manifest = Path(manifest)
        self.cache = cache
        self.num_digits = num_digits
        self.transform = (
            transform
            if transform is not None
            else build_cached_transform(augment=augment, color_mode=color_mode)
        )
        self.samples = [
            (path, year_to_digits(year, num_digits)) for path, year in read_manifest(self.manifest)
        ]
        self.cache.preload(str(path) for path, _ in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, digits = self.samples[index]
        return self.transform(self.cache.get(path)), torch.tensor(digits, dtype=torch.long)
