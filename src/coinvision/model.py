from __future__ import annotations

from typing import cast

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


def _adapt_first_conv(backbone: nn.Module, in_channels: int) -> None:
    """Widen conv1 for stacked views, keeping the pretrained filters (repeated, rescaled)."""
    if in_channels == 3:
        return
    if in_channels % 3 != 0:
        raise ValueError(f"in_channels must be a multiple of 3, got {in_channels}")
    conv = cast(nn.Conv2d, backbone.conv1)
    kernel = int(conv.kernel_size[0])
    stride = int(conv.stride[0])
    padding = int(conv.padding[0])
    widened = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel,
        stride,
        padding,
        bias=conv.bias is not None,
    )
    repeats = in_channels // 3
    with torch.no_grad():
        widened.weight.copy_(conv.weight.repeat(1, repeats, 1, 1) / repeats)
        if conv.bias is not None and widened.bias is not None:
            widened.bias.copy_(conv.bias)
    backbone.conv1 = widened


def build_model(config: ModelConfig, in_channels: int = 3) -> CoinNet:
    backbone, in_features = _build_backbone(config.backbone, config.pretrained)
    _adapt_first_conv(backbone, in_channels)
    return CoinNet(
        backbone,
        in_features,
        num_digits=config.num_digits,
        num_classes_per_digit=config.num_classes_per_digit,
        dropout=config.dropout,
    )
