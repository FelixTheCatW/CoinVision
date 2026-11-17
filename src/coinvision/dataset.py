from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

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


def build_transform(image_size: tuple[int, int]) -> v2.Compose:
    """image_size is (width, height) as in the config; torchvision wants (height, width)."""
    width, height = image_size
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((int(height), int(width)), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


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
    ) -> None:
        self.manifest = Path(manifest)
        self.num_digits = num_digits
        self.transform = transform if transform is not None else build_transform(image_size)
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
