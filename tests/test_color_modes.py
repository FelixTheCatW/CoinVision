from __future__ import annotations

import pytest
import torch
from PIL import Image

from coinvision.config import DataConfig
from coinvision.dataset import (
    GRAY_MEAN,
    Binarize,
    _otsu_threshold,
    build_cached_transform,
    build_transform,
    normalization_for,
)


def _image(width: int = 64, height: int = 32) -> Image.Image:
    """Left half dark grey, right half light grey: one-sided colour, easy to reason about."""
    image = Image.new("RGB", (width, height), (40, 60, 90))
    for y in range(height):
        for x in range(width // 2, width):
            image.putpixel((x, y), (210, 190, 170))
    return image


def test_rgb_mode_keeps_three_channels_and_imagenet_statistics():
    tensor = build_transform((64, 32))( _image())

    assert tensor.shape == (3, 32, 64)
    mean, std = normalization_for("rgb")
    assert mean[0] > mean[2]  # ImageNet is red-heavy, not the grey equivalent
    assert std[0] > std[2]


def test_gray_mode_repeats_one_channel_three_times():
    tensor = build_transform((64, 32), color_mode="gray")(_image())

    assert tensor.shape == (3, 32, 64)
    assert torch.allclose(tensor[0], tensor[1])
    assert torch.allclose(tensor[1], tensor[2])


def test_gray_mode_uses_the_luma_normalisation():
    mean, _ = normalization_for("gray")

    assert mean == GRAY_MEAN
    assert len(set(mean)) == 1


def test_binary_mode_leaves_only_two_values():
    tensor = build_transform((64, 32), color_mode="binary")(_image())

    assert tensor.shape == (3, 32, 64)
    # Normalised with mean 0.5 and std 0.5, so a 0/1 mask lands on exactly -1 and +1.
    values = torch.unique(tensor)
    assert torch.allclose(values, torch.tensor([-1.0, 1.0]))
    assert tensor[0].eq(tensor[1]).all()


def test_binary_mode_keeps_the_bright_half():
    image = _image()
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = build_cached_transform(color_mode="binary")(pixels.reshape(32, 64, 3).permute(2, 0, 1))

    assert tensor.shape == (3, 32, 64)
    left = tensor[0, :, :30].mean()
    right = tensor[0, :, 34:].mean()
    assert left < 0 < right


def test_otsu_splits_a_bimodal_image():
    gray = torch.zeros(32, 64)
    gray[:, 32:] = 0.8

    threshold = _otsu_threshold(gray)

    assert 0.0 < float(threshold) < 0.8


def test_otsu_on_a_flat_image_does_not_crash():
    threshold = _otsu_threshold(torch.full((8, 8), 0.4))

    assert 0.0 <= float(threshold) <= 1.0


def test_binarize_accepts_a_single_channel():
    module = Binarize()
    out = module(torch.rand(1, 16, 16))

    assert out.shape == (3, 16, 16)


def test_unknown_colour_mode_is_rejected():
    with pytest.raises(ValueError, match="color_mode"):
        build_transform((64, 32), color_mode="sepia")


def test_data_config_defaults_to_rgb():
    assert DataConfig().color_mode == "rgb"
