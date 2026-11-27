"""Find the date block directly, instead of hoping a ring window happens to cover it.

The rim sweep tops out at 0.917 of photos: on 8.3% the detected circle is not the coin, or the coin
is small in the frame, so no arc window holds the whole year and no ranker or reader downstream can
fix that. The 1770 annotated quads are exactly the training set of a detector that does not care
about rims at all.

  cache      resize every annotated photo into a fixed canvas and store the boxes in canvas space
  train      heatmap of the year centre + size and orientation regressed at the peak, listing folds
  evaluate   share of photos where the predicted quad covers >= 90% of the annotated year

The metric is containment, not IoU, for the same reason as everywhere else in this repo: a window
wider than the year is allowed to be a window, and coverage is what the crop needs.

    uv run python scripts/date_detector.py cache
    uv run python scripts/date_detector.py train --folds 5
    uv run python scripts/date_detector.py evaluate --checkpoint data/annotations/date_detector.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from PIL import Image
from torch import nn

from coinvision.data import listing_id

# geometry, the canvas transform and the decoder are shared with the pipeline in src so that the
# deployed path and this training script cannot drift apart
from coinvision.date_detector import (  # noqa: F401
    CANVAS,
    EVAL_SAMPLES,
    EXPAND,
    HEAT_MIN,
    HEATMAP,
    STRIDE,
    containment,
    decode,
    quad_from_geometry,
    to_canvas,
)

GAUSS_RADIUS = 2.0
CACHE = Path("data/annotations/date_detector_cache.npz")
CHECKPOINT = Path("data/annotations/date_detector.pt")


# ----------------------------------------------------------------------------- geometry


def quad_geometry(quad: np.ndarray) -> tuple[float, float, float, float, float]:
    """(cx, cy, w, h, angle) - x1->x2 is the reading baseline, as in every other script here."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
    cx = float(np.mean([x1, x2, x3, x4]))
    cy = float(np.mean([y1, y2, y3, y4]))
    width = float(np.hypot(x2 - x1, y2 - y1))
    height = 0.5 * (float(np.hypot(x4 - x1, y4 - y1)) + float(np.hypot(x3 - x2, y3 - y2)))
    angle = float(np.arctan2(y2 - y1, x2 - x1))
    return cx, cy, width, height, angle


# ----------------------------------------------------------------------------- data


def read_boxes() -> dict[str, dict]:
    photos: dict[str, dict] = {}
    for row in csv.DictReader(open("data/annotations/date_boxes.csv", encoding="utf-8")):
        if not all((row.get(f"x{i}") or "").strip() for i in range(1, 5)):
            continue
        quad = np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 5)])
        entry = photos.setdefault(row["image"], {"quads": [], "year": ""})
        entry["quads"].append(quad)
        entry["year"] = entry["year"] or (row.get("year") or "").strip()
    return photos


def cmd_cache(args: argparse.Namespace) -> int:
    photos = read_boxes()
    images = sorted(photos)
    crops: list[np.ndarray] = []
    boxes: list[np.ndarray] = []  # per photo, padded to 4 boxes, 5 numbers each
    keep: list[str] = []
    years: list[str] = []
    listings: list[str] = []
    skipped = 0
    for number, image_path in enumerate(images, 1):
        path = Path(image_path)
        if not path.is_file():
            skipped += 1
            continue
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        canvas, scale, ox, oy = to_canvas(image)
        per_photo = []
        for quad in photos[image_path]["quads"]:
            moved = quad * scale + np.array([ox, oy])
            cx, cy, width, height, angle = quad_geometry(moved)
            per_photo.append([cx, cy, width, height, angle])
        stacked = np.zeros((4, 5), dtype=np.float32)
        stacked[: min(len(per_photo), 4)] = np.asarray(per_photo[:4], dtype=np.float32)
        crops.append(canvas)
        boxes.append(stacked)
        keep.append(image_path)
        years.append(photos[image_path]["year"])
        listings.append(listing_id(image_path))
        if number % 100 == 0 or number == len(images):
            print(f"[{number}/{len(images)}]", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        images=np.stack(crops),
        boxes=np.stack(boxes),
        image=np.asarray(keep),
        year=np.asarray(years),
        listing=np.asarray(listings),
    )
    print(f"{len(crops)} photos ({skipped} skipped), canvas {CANVAS[0]}x{CANVAS[1]} -> {out}")
    return 0


# ----------------------------------------------------------------------------- targets


def build_targets(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heatmap (96x160) plus regression (4x96x160): w, h, cos2a, sin2a, all in canvas pixels.

    The Gaussian is centred on a cell centre, so the argmax of the heatmap is always a cell that
    carries the regression - an off-by-one there decoded to a zero-size quad. The regression is
    smeared across the peak plateau for the same reason: if the network peaks one cell away it
    still reads the box instead of reading zeros.
    """
    heat = np.zeros((HEATMAP[1], HEATMAP[0]), dtype=np.float32)
    regression = np.zeros((6, HEATMAP[1], HEATMAP[0]), dtype=np.float32)
    mask = np.zeros((HEATMAP[1], HEATMAP[0]), dtype=np.float32)
    for cx, cy, width, height, angle in boxes:
        if width <= 0 or height <= 0:
            continue
        cell_c = int(np.clip(round(cx / STRIDE - 0.5), 0, HEATMAP[0] - 1))
        cell_r = int(np.clip(round(cy / STRIDE - 0.5), 0, HEATMAP[1] - 1))
        centre_c, centre_r = cell_c + 0.5, cell_r + 0.5
        for r in range(max(0, cell_r - 3), min(HEATMAP[1], cell_r + 4)):
            for c in range(max(0, cell_c - 3), min(HEATMAP[0], cell_c + 4)):
                gap = ((c + 0.5 - centre_c) ** 2 + (r + 0.5 - centre_r) ** 2) / (GAUSS_RADIUS**2)
                value = float(np.exp(-0.5 * gap))
                if value > heat[r, c]:
                    heat[r, c] = value
                if value >= 0.35:
                    regression[0, r, c] = width / CANVAS[0]
                    regression[1, r, c] = height / CANVAS[1]
                    regression[2, r, c] = float(np.cos(2 * angle))
                    regression[3, r, c] = float(np.sin(2 * angle))
                    # sub-cell offset: a stride-4 heatmap alone puts the centre up to 4 canvas px
                    # away, which is 12 px on a 1500 px photo - wider than the error a 90%
                    # coverage allows for a 50 px year block
                    regression[4, r, c] = cx / STRIDE - (c + 0.5)
                    regression[5, r, c] = cy / STRIDE - (r + 0.5)
                    mask[r, c] = 1.0
    return heat, regression, mask


# ----------------------------------------------------------------------------- model


class DateDetector(nn.Module):
    """Small U-Net: heatmap of the year centre and the quad regressed at the peak."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()

        def block(src: int, dst: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(src, dst, 3, padding=1),
                nn.BatchNorm2d(dst),
                nn.ReLU(inplace=True),
                nn.Conv2d(dst, dst, 3, padding=1),
                nn.BatchNorm2d(dst),
                nn.ReLU(inplace=True),
            )

        self.down1 = block(3, base)
        self.down2 = block(base, base * 2)
        self.down3 = block(base * 2, base * 4)
        self.down4 = block(base * 4, base * 6)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 6, base * 4, 2, stride=2)
        self.mix3 = block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.mix2 = block(base * 4, base * 2)
        self.head = nn.Conv2d(base * 2, 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))  # stride 8
        u3 = self.mix3(torch.cat([self.up3(d4), d3], dim=1))
        u2 = self.mix2(torch.cat([self.up2(u3), d2], dim=1))  # stride 2
        out = self.head(u2)  # 5 x (H/2) x (W/2)
        return torch.nn.functional.interpolate(
            out, size=(HEATMAP[1], HEATMAP[0]), mode="bilinear", align_corners=False
        )


# ----------------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> int:
    torch.backends.cudnn.benchmark = True
    if args.folds < 2:
        # with one fold every listing lands in fold 0, so the training split is empty and the run
        # silently trains on nothing - this guard exists because that happened twice
        raise SystemExit("--folds must be >= 2, otherwise there is no training split")
    payload = np.load(args.cache, allow_pickle=False)
    images = payload["images"]
    boxes = payload["boxes"]
    listings = payload["listing"]
    print(
        f"{len(images)} photos, canvas {images.shape[2]}x{images.shape[1]}, "
        f"heatmap {HEATMAP[0]}x{HEATMAP[1]}, device {args.device}",
        flush=True,
    )

    targets = [build_targets(per_photo) for per_photo in boxes]
    device = torch.device(args.device)
    groups = sorted(set(listings))
    rng = np.random.default_rng(args.seed)

    fold_of = {}
    assign = rng.permutation(len(groups))
    for index, group_index in enumerate(assign):
        fold_of[groups[group_index]] = index % args.folds

    results = []
    for fold in range(args.folds):
        train_idx = [i for i, tag in enumerate(listings) if fold_of[tag] != fold]
        val_idx = [i for i, tag in enumerate(listings) if fold_of[tag] == fold]
        torch.manual_seed(args.seed + fold)
        model = DateDetector().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
        heat_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        for _epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            rng.shuffle(train_idx)
            running, steps = 0.0, 0
            for start in range(0, len(train_idx) - args.batch + 1, args.batch):
                batch = train_idx[start : start + args.batch]
                pixels = torch.from_numpy(images[batch].astype(np.float32) / 255.0)
                pixels = pixels.permute(0, 3, 1, 2).to(device)
                heat = torch.from_numpy(np.stack([targets[i][0] for i in batch])).to(device)
                reg = torch.from_numpy(np.stack([targets[i][1] for i in batch])).to(device)
                mask = torch.from_numpy(np.stack([targets[i][2] for i in batch])).to(device)
                output = model(pixels)
                heat_loss = heat_loss_fn(output[:, 0], heat)
                weight = 1.0 + 8.0 * heat  # the peak cells carry the signal, empty space is noise
                heat_loss = (heat_loss * weight).mean()
                reg_loss = ((output[:, 1:] - reg).abs() * mask.unsqueeze(1)).sum() / (
                    mask.sum() * 6 + 1e-6
                )
                loss = heat_loss + 2.0 * reg_loss
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running += float(loss)
                steps += 1
            scheduler.step()
            elapsed = time.time() - epoch_start
            if _epoch % 5 == 0 or _epoch == args.epochs - 1:
                print(
                    f"  epoch {_epoch + 1:3d}  loss {running / max(steps, 1):.4f}  "
                    f"{elapsed:5.1f}s  {steps / max(elapsed, 1e-6):4.1f} steps/s",
                    flush=True,
                )

        # evaluate this fold on its held-out photos
        model.eval()
        hits, seen = 0, 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch):
                batch = val_idx[start : start + args.batch]
                pixels = torch.from_numpy(images[batch].astype(np.float32) / 255.0)
                pixels = pixels.permute(0, 3, 1, 2).to(device)
                output = model(pixels).cpu().numpy()
                for offset, index in enumerate(batch):
                    quads = decode(output[offset])
                    truth = [quad_from_geometry(*row) for row in boxes[index] if row[2] > 0]
                    best = max((containment(quad, t) for quad in quads for t in truth), default=0.0)
                    hits += int(best >= 0.9)
                    seen += 1
        results.append(hits / max(seen, 1))
        print(f"fold {fold}: photos {seen}  covered@0.9 {results[-1]:.3f}", flush=True)

    print(f"\npooled covered@0.9 {np.mean(results):.3f} +- {np.std(results):.3f}", flush=True)
    torch.save(
        {
            "state": model.state_dict(),
            "canvas": CANVAS,
            "stride": STRIDE,
            "folds": results,
            "epochs": args.epochs,
        },
        args.save_model,
    )
    print(f"final model (last fold) -> {args.save_model}")
    return 0


# ----------------------------------------------------------------------------- evaluate


def cmd_evaluate(args: argparse.Namespace) -> int:
    payload = np.load(args.cache, allow_pickle=False)
    images, boxes = payload["images"], payload["boxes"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = DateDetector()
    model.load_state_dict(checkpoint["state"])
    model.eval()

    hits = {1: 0, 3: 0}
    total = 0
    with torch.no_grad():
        for start in range(0, len(images), 8):
            batch = range(start, min(start + 8, len(images)))
            pixels = torch.from_numpy(images[list(batch)].astype(np.float32) / 255.0)
            output = model(pixels.permute(0, 3, 1, 2)).numpy()
            for offset, index in enumerate(batch):
                for top_k in hits:
                    quads = decode(output[offset], top_k=top_k)
                    truth = [quad_from_geometry(*row) for row in boxes[index] if row[2] > 0]
                    best = max((containment(quad, t) for quad in quads for t in truth), default=0.0)
                    hits[top_k] += int(best >= 0.9)
                total += 1
    print(f"photos {total}")
    for top_k, value in hits.items():
        print(f"covered@0.9 top-{top_k}: {value / total:.3f}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = DateDetector()
    model.load_state_dict(payload["state"])
    model.eval()

    out = Path(args.onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, CANVAS[1], CANVAS[0])
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["maps"],
        opset_version=17,
        dynamo=False,
    )

    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = model(dummy).numpy()[0]
    exported = session.run(None, {"image": dummy.numpy()})[0][0]
    parity = float(np.abs(reference - exported).max())
    print(f"{out}  parity max |torch - onnx| = {parity:.2e}")

    meta = {
        "canvas": list(CANVAS),
        "stride": STRIDE,
        "heatmap": list(HEATMAP),
        "expand": EXPAND,
        "heat_min": float(HEAT_MIN),
        "top_k": 3,
        "trained_on": {
            "photos": int(payload.get("photos", 0)) if isinstance(payload, dict) else 0,
            "folds": payload.get("folds"),
            "epochs": payload.get("epochs"),
        },
        "coverage": payload.get("coverage"),
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.meta}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="resize the annotated photos into one array")
    cache.add_argument("--out", default=str(CACHE))
    cache.set_defaults(func=cmd_cache)

    train = subparsers.add_parser("train", help="heatmap + quad regression, listing folds")
    train.add_argument("--cache", default=str(CACHE))
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--save-model", default=str(CHECKPOINT))
    train.set_defaults(func=cmd_train)

    export = subparsers.add_parser("export", help="ONNX export of the trained detector")
    export.add_argument("--model", default=str(CHECKPOINT))
    export.add_argument("--onnx", default="onnx/date_detector.onnx")
    export.add_argument("--meta", default="onnx/date_detector.json")
    export.set_defaults(func=cmd_export)

    evaluate = subparsers.add_parser("evaluate", help="coverage of the predicted quad")
    evaluate.add_argument("--cache", default=str(CACHE))
    evaluate.add_argument("--checkpoint", default=str(CHECKPOINT))
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
