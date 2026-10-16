from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    source_dir: str = "data/source"
    manifest_dir: str = "data/manifests"
    train_manifest: str = "data/manifests/train.csv"
    val_manifest: str = "data/manifests/val.csv"
    val_fraction: float = 0.2
    image_size: int = 224
    num_workers: int = 4
    extensions: tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png", ".bmp")

