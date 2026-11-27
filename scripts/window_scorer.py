"""Rank the rim sweep: pick the window that holds the year.

`coinvision.localize` proposes 30 arc windows per photo across the detected coins, and those
windows do contain the handwritten year box in roughly 70% of the annotated photos, while the glyph
runs it groups sit on hair and hatching. So the missing piece is ranking, not proposal - which is
exactly what a few hundred boxes can teach.

  cache   render every window of every annotated photo, labelled by how much of the year quad falls
          inside it (`containment`); IoU is not used - it divides by the union, so it punishes a
          window for being wider than the year, which is what a reading crop is allowed to be
  train   a small CNN over those crops, listing-grouped folds, recall@1/@5 at containment 0.9/0.5

`--kinds engraved` is the default: the year on an NGC slab is printed on the label, outside the
coin the sweep looks at, so slabs are out of scope until a label proposal exists.

    uv run python scripts/window_scorer.py cache
    uv run python scripts/window_scorer.py train --folds 5

The cache lives next to the annotations and is ~0.6 GiB; `data/` is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from coinvision.data import listing_id, read_manifest
from coinvision.localize import DateCandidate, crop_upright, detect_coins, ring_windows, to_gray

# One thick band and a wide arc: a year block is ~0.3 r thick, so the default two thin bands of
# `ring_windows` never fit it whole. Measured on 80 annotated photos this grid puts the whole year
# inside a single window on 82% of them, against 0% for the default grid.
# The step is half of what it was: neighbouring windows used to overlap by 89%, i.e. nine copies of
# one arc. Over the 271 annotated years the share that fits inside a single window barely moves -
# 91.1% at step 12, 90.8% at step 24, 86.7% at step 32 - while the sweep halves from 30 windows per
# coin to 15, which is most of the runtime of a request.
# A second band, 0.60-1.25 r, is for the photos where the rim radius comes out short of the coin:
# 8 of the 26 photos the single band misses have the year at 1.23-1.50 r of the circle, i.e. just
# past its outer edge, all of them on circles whose rim score is a real 0.39-0.78. Adding the band
# takes the ceiling from 0.904 to 0.933 of photos and the sweep from 30 windows per frame to 58.
SWEEP_ARC = 110.0
SWEEP_STEP = 24.0
SWEEP_BANDS = ((0.30, 1.10), (0.60, 1.25))
CROP_SIZE = (160, 80)
# "The crop shows the whole year" and "the crop holds most of it", as fractions of the year area.
CONTAINMENT_TIGHT = 0.9
CONTAINMENT_LOOSE = 0.5
CONTAINMENT_SAMPLES = (32, 16)
# p(whole year inside the top window) above which the pipeline answers "there is a date". 0.5 was
# the right cut for the 250-photo model, whose negatives reached 0.87 and whose positives sat at
# 0.95; the 1637-photo model ranks better but scores lower, so nothing but its own scale fits it.
# Measured over 1747 annotated photos: 0.10 passes 97.1% of them with 89.7% of the passes right,
# against 88.8% / 94.3% at 0.5 - and the ceiling of the grid is 91.7%, so precision cannot go much
# higher than that. Exported into the metadata, so the consumer reads the operating point from the
# model instead of carrying a constant of its own.
VERDICT_THRESHOLD = 0.10
DEFAULT_CACHE = "data/annotations/windows_cache.npz"


# --------------------------------------------------------------------------- cache


def truth_quads(annotations: str | Path) -> dict[str, list[list[tuple[float, float]]]]:
    """The annotated four corners per image, one quad per date instance."""
    quads: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for row in read_manifest(annotations):
        if not (row.get("x1") or "").strip():
            continue
        quads[row["image"]].append(
            [(float(row[f"x{i}"]), float(row[f"y{i}"])) for i in range(1, 5)]
        )
    return dict(quads)


def quad_samples(quad: Sequence[tuple[float, float]]) -> np.ndarray:
    """A uniform grid of points inside the quad, for area fractions."""
    across, down = CONTAINMENT_SAMPLES
    s = (np.arange(across) + 0.5) / across
    t = (np.arange(down) + 0.5) / down
    grid_s, grid_t = np.meshgrid(s, t, indexing="ij")
    p1, p2, p3, p4 = (np.asarray(point, dtype=np.float64) for point in quad)
    points = (
        (1 - grid_s)[..., None] * (1 - grid_t)[..., None] * p1
        + grid_s[..., None] * (1 - grid_t)[..., None] * p2
        + grid_s[..., None] * grid_t[..., None] * p3
        + (1 - grid_s)[..., None] * grid_t[..., None] * p4
    )
    return points.reshape(-1, 2)


def containment(windows: Sequence[DateCandidate], quad_points: np.ndarray) -> np.ndarray:
    """Fraction of the year quad inside each window, in the window's own upright frame.

    `crop_upright` maps the destination +u axis onto `(cos a, -sin a)` with `a = -angle_deg`, so a
    source point sits inside the arc rectangle when `|u| <= arc_span / 2` and
    `|v| <= radial_span / 2` in the same frame - which is exactly what the crop will show.
    """
    arc = np.asarray([window.arc_span for window in windows], dtype=np.float64)[:, None]
    radial = np.asarray([window.radial_span for window in windows], dtype=np.float64)[:, None]
    alpha = np.radians(-np.asarray([window.angle_deg for window in windows],
                                   dtype=np.float64))[:, None]
    centre_x = np.asarray([window.center[0] for window in windows], dtype=np.float64)[:, None]
    centre_y = np.asarray([window.center[1] for window in windows], dtype=np.float64)[:, None]
    dx = quad_points[None, :, 0] - centre_x
    dy = quad_points[None, :, 1] - centre_y
    u = np.cos(alpha) * dx - np.sin(alpha) * dy
    v = np.sin(alpha) * dx + np.cos(alpha) * dy
    inside = (np.abs(u) <= arc / 2.0) & (np.abs(v) <= radial / 2.0)
    return inside.mean(axis=1)


def annotation_kinds(annotations: str | Path) -> dict[str, str]:
    """`engraved` / `slab` per image, from the first filled row of that image."""
    kinds: dict[str, str] = {}
    for row in read_manifest(annotations):
        if (row.get("x1") or "").strip():
            kinds.setdefault(row["image"], (row.get("kind") or "").strip().lower())
    return kinds


def cmd_cache(args: argparse.Namespace) -> int:
    quads = truth_quads(args.annotations)
    kinds_of_image = annotation_kinds(args.annotations)
    wanted = {kind.strip().lower() for kind in args.kinds.split(",") if kind.strip()}
    images = [image for image in quads
              if Path(image).is_file() and kinds_of_image.get(image, "") in wanted]
    missing = len(quads) - len(images)
    crops: list[np.ndarray] = []
    overlaps: list[float] = []
    photo_of: list[int] = []
    years: list[str] = []
    kinds: list[str] = []
    per_photo: list[int] = []

    for number, image_path in enumerate(images, start=1):
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        circles = detect_coins(to_gray(image), max_coins=args.max_coins)
        windows = [
            window
            for circle in circles
            for window in ring_windows(circle, arc_degrees=SWEEP_ARC,
                                        step_degrees=SWEEP_STEP, bands=SWEEP_BANDS)
        ]
        for window in windows:
            crop = crop_upright(image, window, size=CROP_SIZE, margin=0.0)
            crops.append(np.asarray(crop, dtype=np.uint8))
            photo_of.append(number - 1)
        if windows:
            best = np.zeros(len(windows), dtype=np.float64)
            for quad in quads[image_path]:
                best = np.maximum(best, containment(windows, quad_samples(quad)))
            overlaps.extend(best.tolist())
        per_photo.append(len(windows))
        years.append(Path(image_path).stem.rsplit("-", 1)[-1])
        kinds.append(kinds_of_image.get(image_path, ""))
        if number % 25 == 0 or number == len(images):
            print(f"[{number}/{len(images)}] {len(crops)} crops, "
                  f"{np.mean(per_photo):.0f} per photo", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        crops=np.stack(crops),
        containment=np.asarray(overlaps, dtype=np.float32),
        photo=np.asarray(photo_of, dtype=np.int32),
        image=np.asarray(images),
        year=np.asarray(years),
        kind=np.asarray(kinds),
    )
    print(f"{len(crops)} crops from {len(images)} photos ({missing} skipped: missing on disk or "
          f"another kind) -> {out}")
    print(f"kinds kept: {dict(Counter(kinds))}")
    print(f"positives (containment >= {CONTAINMENT_TIGHT}): "
          f"{int((np.asarray(overlaps) >= CONTAINMENT_TIGHT).sum())} of {len(crops)}")
    return 0


# --------------------------------------------------------------------------- model


class WindowScorer(nn.Module):
    """Small CNN over a window crop; one logit per window."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )
        self.head = nn.Linear(96, 1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = self.features(batch)
        pooled = features.mean(dim=(2, 3))
        return self.head(pooled).squeeze(1)


# --------------------------------------------------------------------------- folds


def grouped_folds(years: np.ndarray, images: np.ndarray, folds: int) -> np.ndarray:
    """One fold per photo, never splitting the shots of one listing; years stay balanced."""
    listings: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        listings[listing_id(image)].append(index)
    order = sorted(listings, key=lambda key: (str(years[listings[key][0]]), key))
    fold_of = np.zeros(len(images), dtype=np.int64)
    for position, key in enumerate(order):
        for index in listings[key]:
            fold_of[index] = position % folds
    return fold_of


# --------------------------------------------------------------------------- training


def sample_batch(
    photo_ids: np.ndarray,
    positive: np.ndarray,
    negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """All positives of every photo plus `negatives` random negatives - 60:1 is not learnable."""
    chosen: list[int] = []
    by_photo: dict[int, list[int]] = defaultdict(list)
    for index, photo in enumerate(photo_ids):
        by_photo[int(photo)].append(index)
    for _photo, indices in by_photo.items():
        indices = np.asarray(indices)
        is_positive = positive[indices]
        chosen.extend(indices[is_positive].tolist())
        pool = indices[~is_positive]
        if len(pool):
            take = min(negatives, len(pool))
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.asarray(chosen, dtype=np.int64)


def evaluate(
    model: nn.Module,
    crops: np.ndarray,
    photo_ids: np.ndarray,
    containment: np.ndarray,
    device: torch.device,
    batch: int = 512,
) -> dict[str, float]:
    """Recall of the best-ranked window per photo; `photo_ids` are already 0..n-1."""
    model.eval()
    scores = np.zeros(len(crops), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(crops), batch):
            chunk = torch.from_numpy(crops[start:start + batch]).to(device).float()
            chunk = chunk.permute(0, 3, 1, 2) / 255.0 - 0.5
            scores[start:start + batch] = model(chunk).cpu().numpy()

    order = np.argsort(-scores)
    top1, top5, random_hit = [], [], []
    for photo in np.unique(photo_ids):
        rows = order[photo_ids[order] == photo]
        if not len(rows):
            continue
        top1.append(float(containment[rows[0]]))
        top5.append(float(containment[rows[:5]].max()))
        random_hit.append(float((containment[rows] >= CONTAINMENT_TIGHT).mean()))
    top1 = np.asarray(top1)
    top5 = np.asarray(top5)
    possible = top1 >= CONTAINMENT_TIGHT
    return {
        "photos": float(len(top1)),
        "photos_possible": float(possible.sum()),
        "random@0.9": float(np.mean(random_hit)) if random_hit else 0.0,
        "recall@1@0.9": float((top1 >= CONTAINMENT_TIGHT).mean()) if len(top1) else 0.0,
        "recall@1@0.5": float((top1 >= CONTAINMENT_LOOSE).mean()) if len(top1) else 0.0,
        "recall@5@0.9": float((top5 >= CONTAINMENT_TIGHT).mean()) if len(top5) else 0.0,
        "recall@5@0.5": float((top5 >= CONTAINMENT_LOOSE).mean()) if len(top5) else 0.0,
        "recall@1@0.9_of_possible": float(possible[possible].mean()) if possible.any() else 0.0,
        "median_top1_containment": float(np.median(top1)) if len(top1) else 0.0,
    }


def fit(
    train_crops: np.ndarray,
    train_positive: np.ndarray,
    train_photo_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Train one scorer on the given windows, sampling positives with a few negatives per photo."""
    model = WindowScorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    rng = np.random.default_rng(seed)
    for _epoch in range(args.epochs):
        model.train()
        picks = sample_batch(train_photo_ids, train_positive, args.negatives, rng)
        order = rng.permutation(picks)
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            chunk = torch.from_numpy(train_crops[batch]).to(device).float()
            chunk = chunk.permute(0, 3, 1, 2) / 255.0 - 0.5
            target = torch.from_numpy(train_positive[batch].astype(np.float32)).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(chunk), target)
            loss.backward()
            optimizer.step()
    return model


def cmd_train(args: argparse.Namespace) -> int:
    payload = np.load(args.cache, allow_pickle=False)
    crops, containment_all, photo_of = payload["crops"], payload["containment"], payload["photo"]
    images, years = payload["image"], payload["year"]
    kinds = payload["kind"] if "kind" in payload else np.asarray([""] * len(images))
    wanted = {kind.strip().lower() for kind in args.kinds.split(",") if kind.strip()}
    positive = containment_all >= args.positive

    fold_of_photo = grouped_folds(years, images, max(args.folds, 1))
    groups = {int(photo): np.flatnonzero(photo_of == photo) for photo in np.unique(photo_of)}
    kept = [photo for photo in groups if str(kinds[photo]).lower() in wanted]
    dropped = len(groups) - len(kept)
    groups = {photo: groups[photo] for photo in kept}
    possible = {photo: bool(positive[rows].any()) for photo, rows in groups.items()}

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feasible = sum(possible.values())
    kept_windows = int(sum(len(rows) for rows in groups.values()))
    kept_positives = int(sum(int(positive[rows].sum()) for rows in groups.values()))
    print(f"{kept_windows} windows, {len(groups)} photos ({dropped} dropped by --kinds), "
          f"{kept_positives} positives (containment >= {args.positive}), "
          f"{len(groups) - feasible} photos with no positive window "
          f"(ceiling {100.0 * feasible / len(groups):.1f}%), device {device}", flush=True)

    results = []
    # `--folds 0` trains only the deployable model, without spending time on evaluation.
    for fold in range(args.folds if args.folds >= 2 else 0):
        train_photos = [photo for photo in groups if fold_of_photo[photo] != fold]
        val_photos = [photo for photo in groups if fold_of_photo[photo] == fold]
        if not train_photos or not val_photos:
            print(f"fold {fold}: skipped, train {len(train_photos)} val {len(val_photos)} photos")
            continue

        train_idx = np.concatenate([groups[photo] for photo in train_photos])
        val_idx = np.concatenate([groups[photo] for photo in val_photos])
        train_crops, val_crops = crops[train_idx], crops[val_idx]
        train_positive, val_containment = positive[train_idx], containment_all[val_idx]
        # Renumbered per side so evaluate() can index photos 0..n-1.
        train_photo_ids = np.concatenate(
            [np.full(len(groups[photo]), index) for index, photo in enumerate(train_photos)])
        val_photo_ids = np.concatenate(
            [np.full(len(groups[photo]), index) for index, photo in enumerate(val_photos)])

        model = fit(train_crops, train_positive, train_photo_ids, args, device,
                    args.seed + fold)

        metrics = evaluate(model, val_crops, val_photo_ids, val_containment, device)
        metrics["fold"] = fold
        metrics["train_photos"] = float(len(train_photos))
        results.append(metrics)
        print(f"fold {fold}: photos {int(metrics['photos'])} "
              f"(possible {int(metrics['photos_possible'])})  "
              f"r@1@0.9 {metrics['recall@1@0.9']:.3f}  r@1@0.5 {metrics['recall@1@0.5']:.3f}  "
              f"r@5@0.9 {metrics['recall@5@0.9']:.3f}  r@1@0.9|pos "
              f"{metrics['recall@1@0.9_of_possible']:.3f}  random {metrics['random@0.9']:.3f}  "
              f"median containment {metrics['median_top1_containment']:.3f}", flush=True)

    if args.save_model:
        all_photos = list(groups)
        all_idx = np.concatenate([groups[photo] for photo in all_photos])
        all_photo_ids = np.concatenate(
            [np.full(len(groups[photo]), index) for index, photo in enumerate(all_photos)])
        deployable = fit(crops[all_idx], positive[all_idx], all_photo_ids, args, device,
                         args.seed + 1000)
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": deployable.state_dict(),
                "crop_size": CROP_SIZE,
                "sweep": {"arc_degrees": SWEEP_ARC, "step_degrees": SWEEP_STEP,
                          "bands": [list(band) for band in SWEEP_BANDS]},
                "positive": args.positive,
                "epochs": args.epochs,
            },
            args.save_model,
        )
        print(f"final model on all {len(all_photos)} photos -> {args.save_model}", flush=True)

    if not results:
        if args.folds >= 2:
            print("no fold had both training and validation photos - need at least two folds",
                  file=sys.stderr)
            return 1
        return 0
    keys = ["random@0.9", "recall@1@0.9", "recall@1@0.5", "recall@5@0.9", "recall@5@0.5",
            "recall@1@0.9_of_possible", "median_top1_containment"]
    pooled = {key: float(np.mean([row[key] for row in results])) for key in keys}
    pooled_std = {key: float(np.std([row[key] for row in results])) for key in keys}
    print("\npooled: " + "  ".join(
        f"{key} {pooled[key]:.3f}+-{pooled_std[key]:.3f}" for key in keys))
    if args.out:
        Path(args.out).write_text(json.dumps({"folds": results, "pooled": pooled,
                                              "pooled_std": pooled_std}, indent=2),
                                  encoding="utf-8")
        print(f"-> {args.out}")
    return 0


# --------------------------------------------------------------------------- entry point


# --------------------------------------------------------------------------- serve


class DeployScorer(nn.Module):
    """The scorer with its normalisation folded in, so ONNX takes RGB in [0, 1]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.model(windows - 0.5)


def cmd_export(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = WindowScorer()
    model.load_state_dict(payload["state"])
    deploy = DeployScorer(model).eval()

    dummy = torch.rand(2, 3, payload["crop_size"][1], payload["crop_size"][0])
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deploy,
        dummy,
        str(onnx_path),
        input_names=["windows"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        reference = deploy(dummy).numpy()
    served = session.run(None, {"windows": dummy.numpy()})[0]
    diff = float(np.abs(reference - served).max())

    meta = {
        "format": "coinvision-window-scorer",
        "version": 1,
        "input": {
            "name": "windows",
            "layout": "NCHW",
            "dtype": "float32",
            "width": int(payload["crop_size"][0]),
            "height": int(payload["crop_size"][1]),
            "range": [0.0, 1.0],
            "channel_order": "RGB",
            "resize": "affine upright crop of the window, bicubic, no margin",
        },
        "output": {
            "name": "logits",
            "shape": ["batch"],
            "activation": "sigmoid, p(whole year inside this window)",
        },
        "sweep": payload["sweep"],
        "threshold": VERDICT_THRESHOLD,
        "trained_on": {"kinds": "engraved", "containment": payload["positive"],
                       "epochs": payload["epochs"]},
        "parity_max_abs_diff": diff,
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{onnx_path}  parity max |torch - onnx| = {diff:.2e}  -> {args.meta}")
    return 0 if diff < 1e-4 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="render every annotated window into one array")
    cache.add_argument("--annotations", default="data/annotations/date_boxes.csv")
    cache.add_argument("--manifest", default="data/manifests/all.csv")
    cache.add_argument("--out", default=DEFAULT_CACHE)
    cache.add_argument("--max-coins", type=int, default=2)
    cache.add_argument("--kinds", default="engraved",
                       help="comma-separated kinds to keep; slabs are out of scope for now")
    cache.set_defaults(func=cmd_cache)

    train = subparsers.add_parser("train", help="train the window ranker on grouped folds")
    train.add_argument("--cache", default=DEFAULT_CACHE)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--positive", type=float, default=CONTAINMENT_TIGHT,
                       help="containment at which a window counts as positive")
    # Three negatives per positive window, not six: with the half-size sweep the 1:6 batch left
    # sigmoid on true windows around 0.43, so the demo's 0.5 threshold rejected half of them.
    # At 1:3 the same folds give recall@1@0.9 0.912 and positives sit at 0.98.
    train.add_argument("--negatives", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--kinds", default="engraved",
                       help="comma-separated kinds to keep; slabs are out of scope for now")
    train.add_argument("--out", default="data/annotations/window_scorer.json")
    train.add_argument("--save-model", default="data/annotations/window_scorer.pt",
                       help="where to write the model trained on all photos, \"\" to skip")
    train.set_defaults(func=cmd_train)

    export = subparsers.add_parser("export", help="ONNX export of the trained scorer")
    export.add_argument("--model", default="data/annotations/window_scorer.pt")
    export.add_argument("--onnx", default="onnx/window_scorer.onnx")
    export.add_argument("--meta", default="onnx/window_scorer.json")
    export.set_defaults(func=cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
