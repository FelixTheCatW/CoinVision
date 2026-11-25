from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from coinvision.localize import (
    DateCandidate,
    crop_upright,
    detect_coins,
    ring_windows,
    to_gray,
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


def _coin(size: int = 420, centre: tuple[float, float] | None = None, radius: float = 180.0,
          year: str = "1832", text_angle_deg: float = 70.0) -> Image.Image:
    """Bright disc with a rim and four digits laid along an arc, like an engraved date."""
    image = Image.new("RGB", (size, size), (32, 32, 36))
    draw = ImageDraw.Draw(image)
    cx, cy = centre or (size / 2, size / 2)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(198, 172, 96))
    rng = np.random.default_rng(7)
    noise = rng.integers(-18, 19, (size, size, 3), dtype=np.int16)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    inside = np.hypot(grid_y - cy, grid_x - cx) < radius
    disc = np.asarray(image, dtype=np.int16) + noise * inside[..., None]
    image = Image.fromarray(np.uint8(np.clip(disc, 0, 255)))
    draw.ellipse(
        [cx - radius * 0.94, cy - radius * 0.94, cx + radius * 0.94, cy + radius * 0.94],
        outline=(120, 100, 60),
        width=4,
    )

    font = _font(max(int(radius * 0.12), 12))
    step = math.radians(7.0)
    text_radius = radius * 0.84
    start = math.radians(text_angle_deg) - step * (len(year) - 1) / 2
    for index, char in enumerate(year):
        angle = start + step * index
        x = cx + text_radius * math.cos(angle)
        y = cy + text_radius * math.sin(angle)
        draw.text((x, y), char, fill=(70, 55, 30), font=font, anchor="mm")
    return image


def test_detect_coins_finds_a_synthetic_coin():
    circle = detect_coins(to_gray(_coin()), max_coins=1)[0]

    assert circle.radius == pytest.approx(180, rel=0.12)
    assert circle.cx == pytest.approx(210, abs=25)
    assert circle.cy == pytest.approx(210, abs=25)


def test_detect_coins_finds_both_coins_of_a_pair():
    image = Image.new("RGB", (900, 440), (28, 28, 32))
    left = _coin(size=400, centre=(200, 220), radius=170)
    right = _coin(size=400, centre=(200, 220), radius=170)
    image.paste(left, (0, 20))
    image.paste(right, (500, 20))

    circles = detect_coins(to_gray(image), max_coins=2)

    assert len(circles) == 2
    assert min(circle.cx for circle in circles) < 350 < max(circle.cx for circle in circles)
    assert all(circle.radius == pytest.approx(170, rel=0.15) for circle in circles)


def test_rim_radius_prefers_the_outermost_ring():
    """A shield or a portrait closes its own contour; the circle is still the rim, farther out."""
    image = _coin()
    draw = ImageDraw.Draw(image)
    radius = 180 * 0.5
    draw.ellipse(
        [210 - radius, 210 - radius, 210 + radius, 210 + radius], outline=(20, 20, 20), width=6
    )

    circle = detect_coins(to_gray(image), max_coins=1)[0]

    assert circle.radius == pytest.approx(180, rel=0.15)


def test_a_caption_does_not_become_a_circle():
    """A watermark or a caption gives the accumulator a peak but no rim; it must not be a circle."""
    image = Image.new("RGB", (900, 620), (24, 24, 28))
    image.paste(_coin(size=520, centre=(240, 260), radius=210), (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((40, 540), "\u00a9 http://example.com 1837", fill=(235, 235, 235), font=_font(44))

    circles = detect_coins(to_gray(image), max_coins=2)

    assert len(circles) == 1
    assert circles[0].radius == pytest.approx(210, rel=0.15)


def test_uniform_image_yields_no_circle():
    blank = Image.new("RGB", (448, 224), (255, 255, 255))

    assert detect_coins(to_gray(blank)) == []


def test_ring_windows_cover_the_whole_ring():
    circle = detect_coins(to_gray(_coin()), max_coins=1)[0]
    windows = ring_windows(circle)

    angles = sorted((window.angle_deg + 90.0) % 360.0 for window in windows)
    assert len(windows) == len(angles)
    assert len({round(window.angle_deg, 3) for window in windows}) > 30
    # Two radius bands times a full turn of windows.
    assert len(windows) >= 40
    gaps = np.diff(angles)
    assert gaps.max() < 20.0
    for window in windows:
        x0, y0, x1, y1 = window.box
        assert circle.cx - circle.radius - 1 <= x0 <= x1 <= circle.cx + circle.radius + 1
        assert circle.cy - circle.radius - 1 <= y0 <= y1 <= circle.cy + circle.radius + 1


def _stripes(size: tuple[int, int] = (200, 100)) -> Image.Image:
    """Vertical stripes: rotation by 90 degrees turns them into horizontal ones."""
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for column in range(size[0]):
        value = 255 if (column // 10) % 2 == 0 else 20
        array[:, column, :] = value
    return Image.fromarray(array)


def test_crop_upright_keeps_an_upright_candidate_upright():
    image = _stripes()
    candidate = DateCandidate(
        box=(40, 20, 160, 80),
        center=(100.0, 50.0),
        angle_deg=0.0,
        arc_span=120.0,
        radial_span=60.0,
        digits=4,
        score=1.0,
        circle=None,  # type: ignore[arg-type]
    )

    crop = crop_upright(image, candidate, size=(120, 60), margin=0.0)

    assert crop.size == (120, 60)
    # Vertical stripes stay vertical: every row swings between light and dark, columns do not.
    values = np.asarray(crop.convert("L"), dtype=np.float32)
    assert values.std(axis=1).mean() > 1.5 * values.std(axis=0).mean()


def test_crop_upright_rotates_by_ninety_degrees():
    image = _stripes()
    candidate = DateCandidate(
        box=(40, 20, 160, 80),
        center=(100.0, 50.0),
        angle_deg=-90.0,
        arc_span=120.0,
        radial_span=60.0,
        digits=4,
        score=1.0,
        circle=None,  # type: ignore[arg-type]
    )

    crop = crop_upright(image, candidate, size=(60, 120), margin=0.0)

    values = np.asarray(crop.convert("L"), dtype=np.float32)
    # The stripes are now horizontal: columns swing, rows do not.
    assert values.std(axis=0).mean() > 1.5 * values.std(axis=1).mean()


def test_crop_upright_does_not_return_a_blank_frame():
    image = _coin()
    circle = detect_coins(to_gray(image), max_coins=1)[0]
    window = ring_windows(circle)[0]

    crop = crop_upright(image, window, size=(128, 64), margin=0.2)

    assert crop.size == (128, 64)
    assert float(np.asarray(crop.convert("L")).std()) > 1.0
