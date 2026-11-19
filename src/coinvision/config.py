from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/default.toml")


@dataclass
class ProjectConfig:
    name: str = "coinvision"
    seed: int = 42


@dataclass
class DataConfig:
    source_dir: str = "data/source"
    manifest_dir: str = "data/manifests"
    train_manifest: str = "data/manifests/train.csv"
    val_manifest: str = "data/manifests/val.csv"
    val_fraction: float = 0.2
    image_size: tuple[int, int] = (448, 224)
    num_workers: int = 4
    augment: bool = False
    extensions: tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = False
    num_digits: int = 4
    num_classes_per_digit: int = 10
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.0001
    amp: bool = False
    num_workers: int = 4
    out_dir: str = "runs"


@dataclass
class InferConfig:
    checkpoint: str = "runs/best.pt"
    device: str = "auto"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


SECTIONS: dict[str, type] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "infer": InferConfig,
}


def _coerce(default: Any, raw: Any) -> Any:
    """Keep TOML arrays compatible with the tuple defaults used by the dataclasses."""
    if isinstance(default, tuple) and isinstance(raw, list):
        return tuple(raw)
    return raw


def _build(cls: Any, values: dict[str, Any] | None) -> Any:
    defaults = {spec.name: spec.default for spec in fields(cls)}
    kwargs = {
        name: _coerce(defaults[name], raw)
        for name, raw in (values or {}).items()
        if name in defaults
    }
    return cls(**kwargs)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load a TOML config; every section and key is optional and falls back to defaults."""
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)

    return Config(
        project=_build(ProjectConfig, raw.get("project")),
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        infer=_build(InferConfig, raw.get("infer")),
    )
