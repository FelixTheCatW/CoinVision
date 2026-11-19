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


CROP_ANCHORS = ("uniform", "date", "label")
"""Where a 2:1 window is taken from: anywhere, the engraved date (bottom-right of the reverse) or
the slab label (top-left). The dataset mixes them so one model covers both photo types."""

ANCHOR_WEIGHTS = (0.5, 0.25, 0.25)
ANCHOR_EVAL_SCALE = {"uniform": 0.9, "date": 0.35, "label": 0.35}


def crop_box(
    width: int,
    height: int,
    aspect: float,
    scale: float,
    anchor: str,
    jitter: float,
    roll: float,
    roll2: float,
) -> tuple[int, int, int, int]:
    """Window (left, top, width, height) with the requested aspect, kept inside the image."""
    crop_h = max(8, int(round(height * scale)))
    crop_w = max(8, int(round(crop_h * aspect)))
    if crop_w > width:
        crop_w = width
        crop_h = max(8, int(round(crop_w / aspect)))
    if crop_h > height:
        crop_h = height
        crop_w = max(8, int(round(crop_h * aspect)))

    if anchor == "date":
        left = width - crop_w - int(round(jitter * width * roll))
        top = height - crop_h - int(round(jitter * height * roll2))
    elif anchor == "label":
        left = int(round(jitter * width * roll))
        top = int(round(jitter * height * roll2))
    else:
        left = int(round(roll * (width - crop_w)))
        top = int(round(roll2 * (height - crop_h)))
    return max(0, left), max(0, top), crop_w, crop_h


class CroppedCoinDataset(Dataset):
    """Feeds 2:1 windows cropped out of the *native* image, resized to the model input size.

    At 448x224 for the whole photo the engraved date is a few pixels tall and the network can only
    guess; cropping keeps the date at a readable size. ``variant="random"`` mixes the anchors for
    training, a named anchor gives a reproducible view for evaluation and TTA.
    """

    def __init__(
        self,
        manifest: str | Path,
        cache: ImageCache,
        crop_size: tuple[int, int] = (448, 224),
        num_digits: int = 4,
        transform: v2.Compose | None = None,
        variant: str = "random",
        min_scale: float = 0.30,
        max_scale: float = 0.90,
        jitter: float = 0.10,
        augment: bool = False,
    ) -> None:
        if variant != "random" and variant not in CROP_ANCHORS:
            raise ValueError(f"unknown crop variant {variant!r}")
        self.manifest = Path(manifest)
        self.cache = cache
        self.num_digits = num_digits
        self.variant = variant
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.jitter = jitter
        self.out_width, self.out_height = int(crop_size[0]), int(crop_size[1])
        self.aspect = self.out_width / self.out_height
        self.transform = (
            transform if transform is not None else build_cached_transform(augment=augment)
        )
        self.samples = [
            (path, year_to_digits(year, num_digits)) for path, year in read_manifest(self.manifest)
        ]
        self.cache.preload(str(path) for path, _ in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _view(self) -> tuple[str, float, float, float]:
        if self.variant == "random":
            anchor = CROP_ANCHORS[
                int(torch.multinomial(torch.tensor(ANCHOR_WEIGHTS), 1).item())
            ]
            scale = float(torch.empty(1).uniform_(self.min_scale, self.max_scale).item())
            rolls = (float(torch.rand(1).item()), float(torch.rand(1).item()))
        else:
            anchor = self.variant
            scale = ANCHOR_EVAL_SCALE[anchor]
            rolls = (0.0, 0.0)
        return anchor, scale, rolls[0], rolls[1]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, digits = self.samples[index]
        image = self.cache.get(path)
        _, height, width = image.shape
        anchor, scale, roll, roll2 = self._view()
        left, top, crop_w, crop_h = crop_box(
            width, height, self.aspect, scale, anchor, self.jitter, roll, roll2
        )
        window = v2.functional.crop(image, top, left, crop_h, crop_w)
        window = v2.functional.resize(
            window, [self.out_height, self.out_width], antialias=True
        )
        return self.transform(window), torch.tensor(digits, dtype=torch.long)


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
