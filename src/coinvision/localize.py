"""Find the engraved year in a coin photo and cut an upright crop of it.

The network trained on whole frames reads the year only when it happens to be a few dozen pixels
high, and it answers confidently on frames it has never seen at all (a blank white canvas scores
0.66 on 1836). This module is the missing pre-step: it locates the four digits and hands the
classifier a crop in which they are large and horizontal.

The approach is classical and label-free, because the dataset has no date boxes:

1. a coin is a circle, so vote for its centre along the image gradient directions, take the
   strongest peaks (a photo often shows an obverse and a reverse side by side) and read each rim
   radius off the radial gradient profile;
2. unwrap the annulus just inside the rim into a strip indexed by (angle, radius); the engraved text
   becomes dark blobs, letters of the legend and the digits alike;
3. the year is exactly four glyphs of similar size sitting on one radius with regular spacing, so
   group blobs into runs and prefer runs of four;
4. map the winning run back to a rotated rectangle and resample it upright.

Nothing here is trained, and everything is deterministic, so it can be tested and diffed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from PIL.Image import Resampling, Transform
from scipy import ndimage

# The year sits in a band just inside the rim; the beaded border of the rim itself starts around
# 0.95R and would flood the glyph detector with dots.
RING_INNER = 0.55
RING_OUTER = 0.93

# Glyph geometry, as fractions of the coin radius: the year on a raw photo is ~0.05R high.
GLYPH_MIN_HEIGHT = 0.035
GLYPH_MAX_HEIGHT = 0.16
GLYPH_MIN_ASPECT = 0.45  # width / height
GLYPH_MAX_ASPECT = 1.3

YEAR_DIGITS = 4


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    radius: float


@dataclass(frozen=True)
class Blob:
    """One engraved glyph in strip coordinates."""

    angle_index: float
    radius: float
    arc_width: float
    height: float
    strength: float


@dataclass(frozen=True)
class DateCandidate:
    """A run of four glyphs: where it is, how to rotate it, and how much to trust it."""

    box: tuple[int, int, int, int]
    center: tuple[float, float]
    angle_deg: float
    arc_span: float
    radial_span: float
    digits: int
    score: float
    circle: Circle


def _wrap_degrees(value: float) -> float:
    """Fold a rotation into (-180, 180]; crops only care about the angle modulo a turn."""
    return float((value + 180.0) % 360.0 - 180.0)


def to_gray(image: Image.Image) -> np.ndarray:
    """Float32 luminance in [0, 1]."""
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _resize_for_search(gray: np.ndarray, longest: int = 800) -> tuple[np.ndarray, float]:
    height, width = gray.shape
    scale = longest / max(height, width)
    if scale >= 1.0:
        return gray, 1.0
    pixels = np.asarray(np.clip(gray, 0.0, 1.0) * 255.0, dtype=np.uint8)
    resized = Image.fromarray(pixels).resize(
        (max(int(width * scale), 8), max(int(height * scale), 8)), Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0, scale


def _rim_radius(magnitude: np.ndarray, cx: float, cy: float, shortest: int) -> float | None:
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    probe = np.linspace(0.18 * shortest, 0.62 * shortest, 220)
    profile = np.zeros(probe.size, dtype=np.float32)
    for index, radius in enumerate(probe):
        sample_x = cx + radius * np.cos(angles)
        sample_y = cy + radius * np.sin(angles)
        values = ndimage.map_coordinates(magnitude, [sample_y, sample_x], order=1, mode="constant")
        profile[index] = float(np.median(values))
    profile = ndimage.gaussian_filter1d(profile, 2.0)
    best = int(np.argmax(profile))
    if profile[best] <= 1e-6:
        return None
    return float(probe[best])


def detect_coins(gray: np.ndarray, max_coins: int = 2) -> list[Circle]:
    """Circle centres by gradient voting, then a rim radius per centre; strongest first."""
    small, scale = _resize_for_search(gray)
    height, width = small.shape
    shortest = min(height, width)
    smooth = ndimage.gaussian_filter(small, 1.2)
    grad_y, grad_x = np.gradient(smooth)
    magnitude = np.hypot(grad_x, grad_y)

    threshold = np.percentile(magnitude, 96.0)
    edge_y, edge_x = np.nonzero(magnitude >= threshold)
    if edge_y.size < 50:
        return []

    directions = np.arctan2(grad_y[edge_y, edge_x], grad_x[edge_y, edge_x])
    weights = magnitude[edge_y, edge_x]
    accumulator = np.zeros_like(small)
    for radius in np.linspace(0.18 * shortest, 0.58 * shortest, 20):
        for sign in (1.0, -1.0):
            vote_x = np.rint(edge_x + sign * radius * np.cos(directions)).astype(np.int64)
            vote_y = np.rint(edge_y + sign * radius * np.sin(directions)).astype(np.int64)
            inside = (vote_x >= 0) & (vote_x < width) & (vote_y >= 0) & (vote_y < height)
            votes = (vote_y[inside], vote_x[inside])
            np.add.at(accumulator, votes, weights[inside] / max(radius, 1.0))

    accumulator = ndimage.gaussian_filter(accumulator, 3.0)
    circles: list[Circle] = []
    working = accumulator.copy()
    for _ in range(max_coins):
        flat = int(np.argmax(working))
        if float(working.flat[flat]) <= 0.0:
            break
        center_y, center_x = divmod(flat, width)
        radius = _rim_radius(magnitude, float(center_x), float(center_y), shortest)
        if radius is not None:
            circles.append(Circle(cx=center_x / scale, cy=center_y / scale, radius=radius / scale))
        # Non-maximum suppression: one coin per centre.
        radius_px = max(radius or 0.2 * shortest, 0.2 * shortest)
        yy, xx = np.ogrid[:height, :width]
        working[(xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius_px * 1.1) ** 2] = 0.0
    return circles


def detect_coin(gray: np.ndarray) -> Circle | None:
    coins = detect_coins(gray, max_coins=1)
    return coins[0] if coins else None


def unwrap_ring(
    gray: np.ndarray,
    circle: Circle,
    inner: float = RING_INNER,
    outer: float = RING_OUTER,
    n_angles: int = 1800,
    n_radii: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the annulus into [n_angles, n_radii]: rows run along the rim, columns along r."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(circle.radius * inner, circle.radius * outer, n_radii)
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing="ij")
    sample_x = circle.cx + radius_grid * np.cos(angle_grid)
    sample_y = circle.cy + radius_grid * np.sin(angle_grid)
    strip = ndimage.map_coordinates(gray, [sample_y, sample_x], order=1, mode="nearest")
    return np.asarray(strip, dtype=np.float32), angles, radii


def _local_normalise(strip: np.ndarray, size: int = 15) -> np.ndarray:
    """Remove the lighting gradient: glyphs are found against their neighbourhood, not the frame."""
    local_mean = ndimage.uniform_filter(strip, size=size)
    local_var = ndimage.uniform_filter(strip * strip, size=size) - local_mean * local_mean
    return (strip - local_mean) / np.sqrt(np.maximum(local_var, 1e-6))


def find_glyphs(strip: np.ndarray, circle: Circle) -> list[Blob]:
    """Dark glyph blobs: rows of the strip are angles, columns are radii."""
    normalised = _local_normalise(strip)
    mask = normalised < -1.0
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
    mask = ndimage.binary_closing(mask, structure=np.ones((2, 2)))

    labels, count = ndimage.label(mask)
    if count == 0:
        return []

    n_angles, n_radii = strip.shape
    radius_per_step = (circle.radius * RING_OUTER - circle.radius * RING_INNER) / n_radii
    angle_per_step = 2.0 * np.pi / n_angles
    radial_origin = circle.radius * RING_INNER

    glyphs: list[Blob] = []
    for index, slices in enumerate(ndimage.find_objects(labels)):
        if slices is None:
            continue
        angular_slice, radial_slice = slices
        if angular_slice is None or radial_slice is None:
            continue

        radial_mid_index = (radial_slice.start + radial_slice.stop) / 2.0
        radial_center = radial_origin + radial_mid_index * radius_per_step
        height = (radial_slice.stop - radial_slice.start) * radius_per_step
        arc_width = (angular_slice.stop - angular_slice.start) * angle_per_step * radial_center
        strength = float(-normalised[labels == index + 1].mean())

        if not (circle.radius * GLYPH_MIN_HEIGHT <= height <= circle.radius * GLYPH_MAX_HEIGHT):
            continue
        aspect = arc_width / max(height, 1e-6)
        if not (GLYPH_MIN_ASPECT <= aspect <= GLYPH_MAX_ASPECT):
            continue
        if strength < 1.2:
            continue

        glyphs.append(
            Blob(
                angle_index=float(angular_slice.start),
                radius=float(radial_center),
                arc_width=float(arc_width),
                height=float(height),
                strength=strength,
            )
        )
    glyphs.sort(key=lambda blob: blob.angle_index)
    return glyphs


def _run_candidates(
    glyphs: list[Blob], circle: Circle, n_angles: int
) -> list[tuple[list[Blob], float]]:
    """Every run of four consecutive glyphs that could be a year, with its score."""
    if len(glyphs) < YEAR_DIGITS:
        return []

    angle_per_step = 2.0 * np.pi / n_angles
    runs: list[tuple[list[Blob], float]] = []

    for start in range(len(glyphs) - YEAR_DIGITS + 1):
        run = glyphs[start : start + YEAR_DIGITS]
        heights = np.array([blob.height for blob in run])
        radii = np.array([blob.radius for blob in run])
        widths = np.array([blob.arc_width for blob in run])

        if heights.std() > 0.30 * heights.mean():
            continue
        if radii.std() > 0.10 * circle.radius:
            continue

        spacing = np.diff([blob.angle_index for blob in run]) * angle_per_step * radii.mean()
        if spacing.min() < 0.5 * widths.mean() or spacing.max() > 2.0 * widths.mean():
            continue

        consistency = 1.0 - min(float(heights.std() / heights.mean()), 1.0)
        regularity = 1.0 - min(float(spacing.std() / max(spacing.mean(), 1e-6)), 1.0)
        strength = float(np.mean([blob.strength for blob in run]))
        # Years sit inside the legend band rather than on the outermost ring of letters.
        depth = 1.0 - min(abs(radii.mean() / circle.radius - 0.85) / 0.15, 1.0) * 0.3
        runs.append((run, strength * consistency * regularity * depth))

    return runs


def find_date(
    gray: np.ndarray,
    circle: Circle,
    n_angles: int = 1800,
    n_radii: int = 96,
    top_k: int = 1,
) -> list[DateCandidate]:
    """Up to top_k year candidates on this coin, best first."""
    strip, angles, _ = unwrap_ring(gray, circle, n_angles=n_angles, n_radii=n_radii)
    glyphs = find_glyphs(strip, circle)
    runs = _run_candidates(glyphs, circle, n_angles)
    if not runs:
        return []

    angle_per_step = float(angles[1] - angles[0])
    candidates: list[DateCandidate] = []
    for run, score in sorted(runs, key=lambda item: item[1], reverse=True)[:top_k]:
        radii = np.array([blob.radius for blob in run])
        heights = np.array([blob.height for blob in run])
        first = float(angles[int(run[0].angle_index)])
        last_width = run[-1].arc_width / radii.mean() / angle_per_step
        last_step = int(run[-1].angle_index + round(last_width))
        last = float(angles[min(last_step, len(angles) - 1)])
        if last <= first:
            last += 2.0 * np.pi

        inner = float(radii.min() - 0.7 * heights.mean())
        outer = float(radii.max() + 0.7 * heights.mean())
        mid_angle = 0.5 * (first + last)

        corners = [
            (circle.cx + radius * np.cos(angle), circle.cy + radius * np.sin(angle))
            for angle in (first, last)
            for radius in (inner, outer)
        ]
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]

        candidates.append(
            DateCandidate(
                box=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                center=(
                    float(circle.cx + 0.5 * (inner + outer) * np.cos(mid_angle)),
                    float(circle.cy + 0.5 * (inner + outer) * np.sin(mid_angle)),
                ),
                angle_deg=_wrap_degrees(-(np.degrees(mid_angle) + 90.0)),
                arc_span=float((last - first) * 0.5 * (inner + outer)),
                radial_span=float(outer - inner),
                digits=len(run),
                score=score,
                circle=circle,
            )
        )
    return candidates


def localize_date(
    image: Image.Image, max_coins: int = 2, top_k_candidates: int = 1
) -> list[DateCandidate]:
    """Search every detected coin; returns candidates sorted by score, best first."""
    gray = to_gray(image)
    found: list[DateCandidate] = []
    for circle in detect_coins(gray, max_coins=max_coins):
        found.extend(find_date(gray, circle, top_k=top_k_candidates))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return found


def ring_windows(
    circle: Circle, arc_degrees: float = 46.0, step_degrees: float = 11.5
) -> list[DateCandidate]:
    """Cover the ring with overlapping windows, without any glyph segmentation.

    The glyph grouping fires on the legend and the beaded rim as often as on the year, so this brute
    sweep is the fallback that guarantees the year is somewhere in the candidate set.
    """
    windows: list[DateCandidate] = []
    arc = np.radians(arc_degrees)
    step = np.radians(step_degrees)
    count = int(round(2.0 * np.pi / step))
    for inner_f, outer_f in ((0.52, 0.76), (0.74, 0.92)):
        inner = circle.radius * inner_f
        outer = circle.radius * outer_f
        radius_mid = 0.5 * (inner + outer)
        for index in range(count):
            mid = index * step
            half = 0.5 * arc
            corners = [
                (
                    circle.cx + radius * np.cos(mid + sign * half),
                    circle.cy + radius * np.sin(mid + sign * half),
                )
                for sign in (-1.0, 1.0)
                for radius in (inner, outer)
            ]
            xs = [corner[0] for corner in corners]
            ys = [corner[1] for corner in corners]
            windows.append(
                DateCandidate(
                    box=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    center=(
                        float(circle.cx + radius_mid * np.cos(mid)),
                        float(circle.cy + radius_mid * np.sin(mid)),
                    ),
                    angle_deg=_wrap_degrees(-(np.degrees(mid) + 90.0)),
                    arc_span=float(arc * radius_mid),
                    radial_span=float(outer - inner),
                    digits=0,
                    score=0.0,
                    circle=circle,
                )
            )
    return windows


def candidates(image: Image.Image, max_coins: int = 2, top_k: int = 8) -> list[DateCandidate]:
    """Glyph runs first (they carry a score), then the brute-force ring sweep."""
    gray = to_gray(image)
    found: list[DateCandidate] = []
    for circle in detect_coins(gray, max_coins=max_coins):
        found.extend(find_date(gray, circle, top_k=top_k))
        found.extend(ring_windows(circle))
    found.sort(key=lambda candidate: candidate.score, reverse=True)
    return found


def crop_upright(
    image: Image.Image,
    candidate: DateCandidate,
    size: tuple[int, int] = (256, 128),
    margin: float = 0.5,
) -> Image.Image:
    """Rotate the run so the digits are horizontal and resample it to size (width, height)."""
    width, height = size
    arc = float(candidate.arc_span) * (1.0 + margin)
    radial = float(candidate.radial_span) * (1.0 + margin)
    alpha = np.radians(-candidate.angle_deg)
    cos_a, sin_a = float(np.cos(alpha)), float(np.sin(alpha))
    center_x, center_y = candidate.center

    scale_x = arc / width
    scale_y = radial / height
    a = cos_a * scale_x
    b = sin_a * scale_y
    d = -sin_a * scale_x
    e = cos_a * scale_y
    c = center_x - a * (width / 2.0) - b * (height / 2.0)
    f = center_y - d * (width / 2.0) - e * (height / 2.0)
    return image.convert("RGB").transform(
        size, Transform.AFFINE, (a, b, c, d, e, f), resample=Resampling.BICUBIC
    )


def annotate(image: Image.Image, candidate: DateCandidate, crop: Image.Image) -> Image.Image:
    """Debug sheet: the source with the circle and the box marked, plus the upright crop."""
    from PIL import ImageDraw

    source = image.convert("RGB").copy()
    draw = ImageDraw.Draw(source)
    circle = candidate.circle
    draw.ellipse(
        [
            circle.cx - circle.radius,
            circle.cy - circle.radius,
            circle.cx + circle.radius,
            circle.cy + circle.radius,
        ],
        outline=(0, 160, 255),
        width=3,
    )
    draw.rectangle(candidate.box, outline=(255, 0, 0), width=3)

    pad = 10
    sheet = Image.new(
        "RGB",
        (source.width + crop.width + 3 * pad, max(source.height, crop.height + 2 * pad)),
        (20, 20, 24),
    )
    sheet.paste(source, (0, 0))
    sheet.paste(crop, (source.width + 2 * pad, pad))
    return sheet
