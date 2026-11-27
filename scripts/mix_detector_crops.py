"""Merge the detector's crops into the arc-window cache, so one ranker scores both.

Two separately trained scorers have no reason to share a score scale, and the arbitration between
them ("whoever is more confident") is fragile exactly because of that. Teaching one ranker to score
both candidate types removes the comparison: the pipeline takes the single highest score, whatever
kind of window it came from.

The detector's crops are restricted to the photos already in the arc cache so the population stays
the same 1637 photos and every earlier number remains comparable.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from date_detector import DateDetector as TrainedDetector  # noqa: E402

from coinvision.date_detector import CANVAS, decode  # noqa: E402

CROP = (160, 80)
TOP_K = 3
EXPAND = 1.3
ARK = ROOT / "data/annotations/windows_cache_full.npz"
OUT = ROOT / "data/annotations/windows_cache_mixed.npz"
DETECTOR = ROOT / "data/annotations/date_detector.pt"
SAMPLES = 32


def quad_affine(quad, size=CROP):
    (x1, y1), (x2, y2), _, (x4, y4) = quad
    width, height = size
    return ((x2 - x1) / width, (x4 - x1) / height, x1, (y2 - y1) / width, (y4 - y1) / height, y1)


def quad_crop(image, quad, size=CROP):
    width, height = size
    (x1, y1), (x2, y2), _, (x4, y4) = quad
    return image.transform(
        size,
        Image.AFFINE,
        ((x2 - x1) / width, (x4 - x1) / height, x1, (y2 - y1) / width, (y4 - y1) / height, y1),
        resample=Image.BILINEAR,
    )


def quad_samples(quad, samples=SAMPLES):
    u = np.linspace(0.0, 1.0, samples)[:, None, None]
    v = np.linspace(0.0, 1.0, samples)[None, :, None]
    top = quad[0] + v * (quad[1] - quad[0])
    bottom = quad[3] + v * (quad[2] - quad[3])
    return (top + u * (bottom - top)).reshape(-1, 2)


def containment(affine, points):
    a, b, c, d, e, f = affine
    det = a * e - b * d
    x, y = points[:, 0] - c, points[:, 1] - f
    u = (e * x - b * y) / det
    v = (-d * x + a * y) / det
    return float(((u >= 0) & (u <= CROP[0]) & (v >= 0) & (v <= CROP[1])).mean())


def main() -> int:
    arc = np.load(ARK, allow_pickle=False)
    order = {str(name): index for index, name in enumerate(arc["image"])}

    cache = np.load(ROOT / "data/annotations/date_detector_cache.npz", allow_pickle=False)
    check = torch.load(DETECTOR, map_location="cpu", weights_only=False)
    detector = TrainedDetector()
    detector.load_state_dict(check["state"])
    detector.eval()

    truth, kinds = {}, {}
    for row in csv.DictReader(open(ROOT / "data/annotations/date_boxes.csv", encoding="utf-8")):
        if not all((row.get(f"x{i}") or "").strip() for i in range(1, 5)):
            continue
        name = row["image"]
        truth.setdefault(name, []).append(
            np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)])
        )
        kinds.setdefault(name, (row.get("kind") or "").strip().lower())

    crops, overlaps, photo_of = [], [], []
    for start in range(0, len(cache["image"]), 8):
        batch = list(range(start, min(start + 8, len(cache["image"]))))
        pixels = torch.from_numpy(cache["images"][batch].astype(np.float32) / 255.0)
        with torch.no_grad():
            output = detector(pixels.permute(0, 3, 1, 2)).numpy()
        for offset, index in enumerate(batch):
            name = str(cache["image"][index])
            photo = order.get(name)
            if photo is None or kinds.get(name, "") != "engraved":
                continue
            path = ROOT / name
            if not path.is_file():
                continue
            quads = decode(output[offset], top_k=TOP_K, expand=EXPAND)
            if not quads:
                continue
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            width, height = image.size
            scale = min(CANVAS[0] / width, CANVAS[1] / height)
            ox = (CANVAS[0] - max(1, int(round(width * scale)))) // 2
            oy = (CANVAS[1] - max(1, int(round(height * scale)))) // 2
            for canvas_quad in quads:
                quad = (canvas_quad - np.array([ox, oy])) / scale
                crops.append(np.asarray(quad_crop(image, quad), dtype=np.uint8))
                affine = quad_affine(quad)
                overlaps.append(max(containment(affine, quad_samples(q)) for q in truth[name]))
                photo_of.append(photo)
        if (start // 8) % 40 == 0:
            print(f"[{min(start + 8, len(cache['image']))}/{len(cache['image'])}]", flush=True)

    added_crops = np.stack(crops)
    added_photo = np.asarray(photo_of, dtype=np.int32)
    added_containment = np.asarray(overlaps, dtype=np.float32)
    merged_crops = np.concatenate([arc["crops"], added_crops])
    merged_containment = np.concatenate([arc["containment"], added_containment])
    merged_photo = np.concatenate([arc["photo"], added_photo])

    np.savez(
        OUT,
        crops=merged_crops,
        containment=merged_containment,
        photo=merged_photo,
        image=arc["image"],
        year=arc["year"],
        kind=arc["kind"],
    )
    per_photo = merged_crops.shape[0] / len(arc["image"])
    positives = int((merged_containment >= 0.9).sum())
    detector_positives = int((added_containment >= 0.9).sum())
    print(
        f"дуговых {arc['crops'].shape[0]}, детекторных {added_crops.shape[0]} "
        f"на {len(np.unique(added_photo))} фото"
    )
    print(
        f"всего {merged_crops.shape[0]} кропов, {per_photo:.1f} на фото, "
        f"позитивов {positives} ({positives / merged_crops.shape[0]:.3f}), "
        f"из них детекторных {detector_positives}"
    )
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
