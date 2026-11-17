from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .config import ModelConfig


class CoinNet(nn.Module):
    """A CNN backbone with one classification head per digit position."""

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        num_digits: int,
        num_classes_per_digit: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            nn.Linear(in_features, num_classes_per_digit) for _ in range(num_digits)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.dropout(torch.flatten(self.backbone(images), 1))
        return torch.stack([head(features) for head in self.heads], dim=1)


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if not name.startswith("resnet"):
        raise ValueError(f"unsupported backbone {name!r}: only resnet* is supported for now")
    factory = getattr(models, name)
    backbone = factory(weights="DEFAULT" if pretrained else None)
    in_features = backbone.fc.in_features
    return backbone, int(in_features)


def build_model(config: ModelConfig) -> CoinNet:
    backbone, in_features = _build_backbone(config.backbone, config.pretrained)
    return CoinNet(
        backbone,
        in_features,
        num_digits=config.num_digits,
        num_classes_per_digit=config.num_classes_per_digit,
        dropout=config.dropout,
    )
