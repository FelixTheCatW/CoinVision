"""Locate the year block directly, with a heatmap-and-regression network.

The ring grid in `localize` covers the annulus blindly and its ceiling is what it is: on 8.3% of
annotated photos no window holds the whole year, because the detected circle is the wrong circle or
the year sits outside the sweeps. This module predicts the year's rotated rectangle instead.

The network sees a letterboxed 512x320 canvas and returns seven maps: a heatmap of the block centre
at stride 4, the block width and height as fractions of the canvas, the doubled angle as a
(cos, sin) pair, and a sub-cell offset. The offset is not decoration: at stride 4 the centre is
otherwise up to 4 canvas pixels out, which is around 12 pixels on a 1500 px photo and wider than
the error a crop holding a 50 px year block can afford - without it the measured coverage stalled
at 0.68 against 0.92 with it.

Only the ONNX runtime path lives here; training is `scripts/date_detector.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Resampling

# width, height of the letterboxed canvas and the stride of the heatmap
CANVAS = (512, 320)
STRIDE = 4
HEATMAP = (CANVAS[0] // STRIDE, CANVAS[1] // STRIDE)

# The predicted quad is grown by this before it is cropped: the network's size regression runs about
# 15-20% tight, which shows up as coverage 0.41 at 1.0 against 0.85 at 1.3 on annotated photos.
EXPAND = 1.3

# Below this heatmap peak nothing is there; a photo of pure noise produces one weak peak.
HEAT_MIN = 0.05

# Points per side when measuring how much of the year a window holds.
EVAL_SAMPLES = 32


def quad_from_geometry(
    cx: float, cy: float, width: float, height: float, angle: float, expand: float = 1.0
) -> np.ndarray:
    """Corners of a rotated rectangle as [top-left, top-right, bottom-right, bottom-left]."""
    ux, uy = np.cos(angle), np.sin(angle)
    vx, vy = -uy, ux
    half_w, half_h = 0.5 * width * expand, 0.5 * height * expand
    corners = [
        (cx + sx * half_w * ux + sy * half_h * vx, cy + sx * half_w * uy + sy * half_h * vy)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return np.asarray(corners, dtype=np.float64)


def containment(window: np.ndarray, quad: np.ndarray, samples: int = EVAL_SAMPLES) -> float:
    """Share of the year quad that falls inside the window quad, by sampling the year.

    This is the metric the whole stage is tuned against, not IoU: a window may be much wider than
    the year, but the year has to be inside it, or the crop shows four digits cut in half.
    """
    u = np.linspace(0.0, 1.0, samples)[:, None, None]
    v = np.linspace(0.0, 1.0, samples)[None, :, None]
    top = quad[0] + v * (quad[1] - quad[0])
    bottom = quad[3] + v * (quad[2] - quad[3])
    points = (top + u * (bottom - top)).reshape(-1, 2)

    origin = window[0]
    e1 = window[1] - origin
    e2 = window[3] - origin
    det = e1[0] * e2[1] - e1[1] * e2[0]
    if abs(det) < 1e-9:
        return 0.0
    delta = points - origin
    a = (e2[1] * delta[:, 0] - e2[0] * delta[:, 1]) / det
    b = (-e1[1] * delta[:, 0] + e1[0] * delta[:, 1]) / det
    return float(((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).mean())


def to_canvas(image: Image.Image) -> tuple[np.ndarray, float, float, float]:
    """Letterbox into CANVAS without distorting; returns the array, the scale and the offset."""
    width, height = image.size
    scale = min(CANVAS[0] / width, CANVAS[1] / height)
    new = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.convert("RGB").resize(new, Resampling.LANCZOS)
    canvas = np.zeros((CANVAS[1], CANVAS[0], 3), dtype=np.uint8)
    ox = (CANVAS[0] - new[0]) // 2
    oy = (CANVAS[1] - new[1]) // 2
    canvas[oy : oy + new[1], ox : ox + new[0]] = np.asarray(resized)
    return canvas, scale, float(ox), float(oy)


def decode(output: np.ndarray, top_k: int = 3, expand: float = EXPAND) -> list[np.ndarray]:
    """Turn the seven maps into up to top_k quads, strongest heatmap peak first.

    Peaks within two cells of an already picked one are suppressed: a year block lights up a cluster
    of cells, and taking the neighbours too would fill the candidate list with the same answer.
    """
    heat = 1.0 / (1.0 + np.exp(-output[0]))
    regression = output[1:]
    flat = heat.reshape(-1)
    order = np.argsort(-flat)[: top_k * 20]
    picked: list[tuple[int, int]] = []
    for index in order:
        row, col = divmod(int(index), HEATMAP[0])
        if any(abs(row - r) <= 2 and abs(col - c) <= 2 for r, c in picked):
            continue
        if flat[index] < HEAT_MIN:
            continue
        picked.append((row, col))
        if len(picked) >= top_k:
            break

    quads = []
    for row, col in picked:
        cx = (col + 0.5 + float(regression[4, row, col])) * STRIDE
        cy = (row + 0.5 + float(regression[5, row, col])) * STRIDE
        width = float(regression[0, row, col]) * CANVAS[0]
        height = float(regression[1, row, col]) * CANVAS[1]
        angle = 0.5 * float(np.arctan2(regression[3, row, col], regression[2, row, col]))
        if width <= 1 or height <= 1:
            continue
        quads.append(quad_from_geometry(cx, cy, width, height, angle, expand))
    return quads


class DateDetector:
    """The exported detector: one photo in, year quads in that photo's own coordinates out."""

    def __init__(self, onnx: str | Path, meta: str | Path | None = None) -> None:
        import onnxruntime as ort

        self.meta = json.loads(Path(meta).read_text()) if meta else {}
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.expand = float(self.meta.get("expand", EXPAND))
        self.top_k = int(self.meta.get("top_k", 3))

    def quads(self, image: Image.Image, top_k: int | None = None) -> list[np.ndarray]:
        """Year quads in image pixels, best first; empty when the heatmap stays cold."""
        canvas, scale, ox, oy = to_canvas(image)
        batch = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        output = self.session.run(None, {self.input_name: batch})[0][0]
        found = decode(output, top_k=top_k or self.top_k, expand=self.expand)
        # the canvas is a letterbox, so the predicted coordinates have to come back out of it
        return [(quad - np.array([ox, oy])) / scale for quad in found]
