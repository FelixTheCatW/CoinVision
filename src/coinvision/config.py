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
    cache: bool = True
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
    deterministic: bool = False
    label_smoothing: float = 0.0
    lr_schedule: str = "none"
    warmup_epochs: int = 1
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


def _read_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def _overlay(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge TOML documents: section by section, override wins on the keys it defines."""
    merged = {section: dict(values) for section, values in base.items()}
    for section, values in override.items():
        merged[section] = {**merged.get(section, {}), **values}
    return merged


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    base: str | Path | None = DEFAULT_CONFIG_PATH,
) -> Config:
    """Load a TOML config, layered on top of ``base`` (the project config by default).

    A partial experiment file therefore inherits every setting it does not mention from
    configs/default.toml instead of silently falling back to the dataclass defaults - which is how
    an experiment once lost ``pretrained = true`` and trained from scratch.
    """
    raw = _read_toml(path)
    if base is not None:
        base_path, path_obj = Path(base), Path(path)
        if base_path.is_file() and base_path.resolve() != path_obj.resolve():
            raw = _overlay(_read_toml(base_path), raw)

    return Config(
        project=_build(ProjectConfig, raw.get("project")),
        data=_build(DataConfig, raw.get("data")),
        model=_build(ModelConfig, raw.get("model")),
        train=_build(TrainConfig, raw.get("train")),
        infer=_build(InferConfig, raw.get("infer")),
    )
