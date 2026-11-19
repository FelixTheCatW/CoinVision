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


def build_transform(image_size: tuple[int, int], augment: bool = False) -> v2.Compose:
    """image_size is (width, height) as in the config; torchvision wants (height, width)."""
    width, height = image_size
    ops: list = [v2.ToImage(), v2.Resize((int(height), int(width)), antialias=True)]
    if augment:
        ops.extend(_augment_ops())
    ops.extend([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return v2.Compose(ops)


def build_cached_transform(augment: bool = False) -> v2.Compose:
    """Same pipeline as build_transform, for pixels ImageCache already decoded and resized."""
    ops: list = [] if not augment else list(_augment_ops())
    ops.extend([v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return v2.Compose(ops)


STACK_VIEWS = ("full", "label", "date")
"""Views concatenated along the channels: the whole frame, the slab label (top-left) and the
engraved date (bottom-right). A single forward pass sees all of them, so the network learns which
view carries the year for a given photo instead of us guessing the photo type."""

VIEW_SPECS: dict[str, tuple[float, str]] = {
    "full": (1.0, "centre"),
    "label": (0.35, "top_left"),
    "date": (0.35, "bottom_right"),
}


def view_window(
    width: int, height: int, aspect: float, scale: float, anchor: str
) -> tuple[int, int, int, int]:
    """Deterministic (left, top, width, height) window with the requested aspect."""
    crop_h = max(8, int(round(height * scale)))
    crop_w = max(8, int(round(crop_h * aspect)))
    if crop_w > width:
        crop_w = width
        crop_h = max(8, int(round(crop_w / aspect)))
    if crop_h > height:
        crop_h = height
        crop_w = max(8, int(round(crop_h * aspect)))

    if anchor == "top_left":
        left, top = 0, 0
    elif anchor == "bottom_right":
        left, top = width - crop_w, height - crop_h
    else:
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return left, top, crop_w, crop_h


class StackedViewsDataset(Dataset):
    """One sample = several deterministic views of the photo, concatenated along the channels."""

    def __init__(
        self,
        manifest: str | Path,
        cache: ImageCache,
        view_size: tuple[int, int] = (448, 224),
        num_digits: int = 4,
        views: tuple[str, ...] = STACK_VIEWS,
        transform: v2.Compose | None = None,
    ) -> None:
        unknown = set(views) - set(VIEW_SPECS)
        if unknown:
            raise ValueError(f"unknown views: {sorted(unknown)}")
        self.manifest = Path(manifest)
        self.cache = cache
        self.views = views
        self.num_digits = num_digits
        self.out_width, self.out_height = int(view_size[0]), int(view_size[1])
        self.aspect = self.out_width / self.out_height
        self.transform = transform if transform is not None else build_cached_transform()
        self.samples = [
            (path, year_to_digits(year, num_digits)) for path, year in read_manifest(self.manifest)
        ]
        self.cache.preload(str(path) for path, _ in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, digits = self.samples[index]
        image = self.cache.get(path)
        _, height, width = image.shape
        pieces = []
        for view in self.views:
            scale, anchor = VIEW_SPECS[view]
            left, top, crop_w, crop_h = view_window(width, height, self.aspect, scale, anchor)
            window = v2.functional.crop(image, top, left, crop_h, crop_w)
            window = v2.functional.resize(
                window, [self.out_height, self.out_width], antialias=True
            )
            pieces.append(self.transform(window))
        return torch.cat(pieces, dim=0), torch.tensor(digits, dtype=torch.long)


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
    ) -> None:
        self.manifest = Path(manifest)
        self.num_digits = num_digits
        self.transform = (
            transform
            if transform is not None
            else build_transform(image_size, augment=augment)
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
    ) -> None:
        self.manifest = Path(manifest)
        self.cache = cache
        self.num_digits = num_digits
        self.transform = (
            transform if transform is not None else build_cached_transform(augment=augment)
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
