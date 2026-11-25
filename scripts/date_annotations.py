"""Prepare and check the date-box annotations the localiser cannot produce on its own.

`coinvision.localize` finds whole coins and proposes windows that contain the year, but it cannot
pick the right one (EXPERIMENTS.md: 2/16 correct by classifier confidence, 1/12 on the ring sweep),
so the missing ingredient is ground truth: four corners around the digits in every photo. This
script makes that annotation cheap and makes a wrong box visible:

  template  an empty annotation CSV, one row per manifest image
  sheets    one PNG per photo: the localiser's glyph candidates drawn and numbered on the original,
            plus a stack of upright crops with the same numbers, and a CSV of those quads to copy
  check     every filled row validated and re-rendered as an upright crop

Annotation is four points per date instance (see `annotation_header`), all in the original frame and
all integer pixels: `x1,y1` is the top-left corner of the year as the annotator reads it, then the
corners follow clockwise - `x2,y2` top-right, `x3,y3` bottom-right, `x4,y4` bottom-left. The quad
is taken literally - it is never squared off from a diagonal, so a box that follows the curve of
the digits keeps its skew - and `x1,y1 -> x2,y2` is the text baseline the review crop is resampled
along. Round text has no geometric "up" - on the lower half of a coin the letters' tops face the
centre - so the convention is the reader's, not the rim's. `img-explorer/` copies exactly this
click order to the clipboard as `x1,y1,x2,y2,x3,y3,x4,y4`.

    uv run python scripts/date_annotations.py template --out data/annotations/date_boxes.csv
    uv run python scripts/date_annotations.py sheets --count 300 --seed 42
    uv run python scripts/date_annotations.py check --annotations data/annotations/date_boxes.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coinvision.data import read_manifest
from coinvision.localize import Circle, DateCandidate, candidates, crop_upright

# Annotation is four points, clockwise from the top-left corner of the year as it is read.
QUAD_COLUMNS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]

# Review sheets: photo panel cap, crop panel geometry.
PHOTO_MAX = (1080, 700)
CROP_SIZE = (256, 96)
CROP_LABEL_WIDTH = 34
CAPTION_HEIGHT = 26
BACKGROUND = (20, 20, 24)
CIRCLE_COLOUR = (0, 160, 255)
CANDIDATE_COLOUR = (255, 70, 70)
TRUTH_COLOUR = (60, 220, 90)

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - older Pillow
        return ImageFont.load_default()


def annotation_header() -> list[str]:
    return ["image", *QUAD_COLUMNS, "kind", "year"]


def candidate_quad(candidate: DateCandidate) -> list[tuple[float, float]]:
    """Four corners of the candidate as a text block, clockwise from the top-left.

    `crop_upright` maps its destination +u axis onto the source direction
    `(cos alpha, -sin alpha)` with `alpha = -angle_deg`, and +v onto `(sin alpha, cos alpha)`.
    Text on a coin is read left to right, so the pair is flipped 180 degrees whenever +u would
    point leftwards; without that flip a run on the lower half of the coin comes out inverted.
    The quad is then `c -+ half_arc * u -+ half_rad * v`, which makes `quad_to_candidate` its
    inverse up to that flip.
    """
    alpha = math.radians(-candidate.angle_deg)
    ux, uy = math.cos(alpha), -math.sin(alpha)
    vx, vy = math.sin(alpha), math.cos(alpha)
    if ux < 0.0:
        ux, uy, vx, vy = -ux, -uy, -vx, -vy
    half_arc = candidate.arc_span / 2.0
    half_rad = candidate.radial_span / 2.0
    centre_x, centre_y = candidate.center
    return [
        (centre_x - half_arc * ux - half_rad * vx, centre_y - half_arc * uy - half_rad * vy),
        (centre_x + half_arc * ux - half_rad * vx, centre_y + half_arc * uy - half_rad * vy),
        (centre_x + half_arc * ux + half_rad * vx, centre_y + half_arc * uy + half_rad * vy),
        (centre_x - half_arc * ux + half_rad * vx, centre_y - half_arc * uy + half_rad * vy),
    ]


def quad_to_candidate(points: Sequence[tuple[float, float]]) -> DateCandidate:
    """The inverse of `candidate_quad`: an annotated quad that `crop_upright` can resample."""
    (p1, p2, p3, _p4) = points
    along_x, along_y = p2[0] - p1[0], p2[1] - p1[1]
    across_x, across_y = p3[0] - p2[0], p3[1] - p2[1]
    arc_span = math.hypot(along_x, along_y)
    radial_span = math.hypot(across_x, across_y)
    # +u = (cos alpha, -sin alpha)  =>  alpha = atan2(-u_y, u_x)  and  angle_deg = -alpha.
    alpha = math.atan2(-along_y, along_x)
    centre_x = sum(point[0] for point in points) / 4.0
    centre_y = sum(point[1] for point in points) / 4.0
    return DateCandidate(
        box=(
            int(min(point[0] for point in points)),
            int(min(point[1] for point in points)),
            int(max(point[0] for point in points)),
            int(max(point[1] for point in points)),
        ),
        center=(centre_x, centre_y),
        angle_deg=-math.degrees(alpha),
        arc_span=arc_span,
        radial_span=radial_span,
        digits=0,
        score=0.0,
        # Only `crop_upright` reads the candidate, and it uses nothing but angle, centre and span.
        circle=Circle(cx=centre_x, cy=centre_y, radius=max(arc_span, radial_span) / 2.0),
    )


# --------------------------------------------------------------------------- template


def cmd_template(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(annotation_header())
        for row in rows:
            writer.writerow([row["image"], *([""] * len(QUAD_COLUMNS)), "", row["year"]])
    print(f"{len(rows)} rows -> {out}")
    print("x1,y1 = top-left of the year, then clockwise: x2,y2 top-right, x3,y3 bottom-right,")
    print("x4,y4 bottom-left, empty = no legible date")
    print("kind is engraved or slab")
    return 0


# --------------------------------------------------------------------------- sheets


def _allocate(counts: dict[str, int], wanted: int) -> dict[str, int]:
    """Split `wanted` between the years proportionally, never asking for more than exists."""
    keys = sorted(counts)
    total = sum(counts.values())
    if wanted >= total:
        return dict(counts)
    raw = {key: wanted * counts[key] / total for key in keys}
    quota = {key: min(int(raw[key]), counts[key]) for key in keys}
    remaining = wanted - sum(quota.values())
    while remaining > 0:
        order = sorted(keys, key=lambda key: raw[key] - int(raw[key]), reverse=True)
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if quota[key] < counts[key]:
                quota[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quota


def select_batch(rows: list[dict[str, str]], count: int | None, seed: int) -> list[dict[str, str]]:
    """A stratified, deterministic slice of the manifest: every year keeps its share."""
    if count is None or count >= len(rows):
        return list(rows)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["year"], []).append(row)
    quota = _allocate({year: len(group) for year, group in grouped.items()}, count)
    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    for year in sorted(grouped):
        group = grouped[year]
        indices = sorted(rng.sample(range(len(group)), quota[year]))
        chosen.extend(group[index] for index in indices)
    chosen.sort(key=lambda row: row["image"])
    return chosen


def _split_candidates(found: Sequence[DateCandidate], top_k: int) -> tuple[list[DateCandidate],
                                                                         list[DateCandidate]]:
    glyph_runs = [item for item in found if item.digits > 0][:top_k]
    windows = [item for item in found if item.digits == 0]
    return glyph_runs, windows


def _unique_circles(found: Sequence[DateCandidate]) -> list[Circle]:
    """Every coin the sweep saw, in detection order - the year is not always on the coin with
    glyph runs, so the sheet has to mark all of them."""
    seen: dict[tuple[int, int, int], Circle] = {}
    for candidate in found:
        circle = candidate.circle
        key = (round(circle.cx), round(circle.cy), round(circle.radius))
        seen.setdefault(key, circle)
    return list(seen.values())


def _draw_badge(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, panel: tuple[int, int]):
    font = _font(18)
    left = max(2.0, min(x, panel[0] - 30.0))
    top = max(2.0, min(y, panel[1] - 24.0))
    draw.rectangle([left, top, left + 26, top + 22], fill=(250, 230, 60))
    draw.text((left + 6, top + 2), text, fill=(20, 20, 24), font=font)


def render_sheet(
    image: Image.Image,
    stem: str,
    year: str,
    glyph_runs: Sequence[DateCandidate],
    total_candidates: int,
    circles: Sequence[Circle],
) -> Image.Image:
    """Original with numbered candidate quads, next to the same candidates as upright crops."""
    scale = min(
        1.0,
        PHOTO_MAX[0] / image.width,
        PHOTO_MAX[1] / image.height,
    )
    photo_w = max(int(image.width * scale), 8)
    photo_h = max(int(image.height * scale), 8)
    photo = image.resize((photo_w, photo_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(photo)

    for circle in circles:
        draw.ellipse(
            [
                (circle.cx - circle.radius) * scale,
                (circle.cy - circle.radius) * scale,
                (circle.cx + circle.radius) * scale,
                (circle.cy + circle.radius) * scale,
            ],
            outline=CIRCLE_COLOUR,
            width=2,
        )

    for index, candidate in enumerate(glyph_runs, start=1):
        points = [(x * scale, y * scale) for x, y in candidate_quad(candidate)]
        draw.line([*points, points[0]], fill=CANDIDATE_COLOUR, width=2)
        # Candidates cluster (hair, shield hatching), so stagger the badges or the numbers pile up.
        stagger = ((index - 1) % 4) * 16.0
        _draw_badge(
            draw,
            points[0][0] - 13.0,
            points[0][1] - 26.0 - stagger,
            str(index),
            (photo_w, photo_h),
        )

    crop_col = CROP_LABEL_WIDTH + CROP_SIZE[0]
    gap = 10
    canvas_w = photo_w + (gap + crop_col if glyph_runs else 0)
    canvas_h = max(photo_h + CAPTION_HEIGHT, len(glyph_runs) * CROP_SIZE[1] + 4)
    sheet = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    sheet.paste(photo, (0, 0))

    caption_font = _font(15)
    caption = f"{stem}   year={year}   glyph runs={len(glyph_runs)} of {total_candidates}"
    if not glyph_runs:
        caption += "   (no glyph run: draw the corners by hand, see candidates.csv windows)"
    ImageDraw.Draw(sheet).text((4, photo_h + 4), caption, fill=(210, 210, 220), font=caption_font)

    label_draw = ImageDraw.Draw(sheet)
    label_font = _font(20)
    for index, candidate in enumerate(glyph_runs, start=1):
        crop = crop_upright(image, candidate, size=CROP_SIZE)
        top = (index - 1) * CROP_SIZE[1] + 2
        sheet.paste(crop, (photo_w + gap + CROP_LABEL_WIDTH, top))
        label_draw.text((photo_w + gap + 8, top + CROP_SIZE[1] // 2 - 12), str(index),
                        fill=(250, 230, 60), font=label_font)
    return sheet


def cmd_sheets(args: argparse.Namespace) -> int:
    rows = read_manifest(args.manifest)
    batch = select_batch(rows, args.count, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "batch.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "year"])
        writer.writeheader()
        writer.writerows(batch)

    columns = [
        "image", "candidate", "source", "score", "digits", "angle_deg",
        *QUAD_COLUMNS,
    ]
    quad_rows: list[dict[str, object]] = []
    missing = 0
    for number, row in enumerate(batch, start=1):
        path = Path(row["image"])
        if not path.is_file():
            print(f"skip {path}: not a file", file=sys.stderr, flush=True)
            missing += 1
            continue
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        found = candidates(image, max_coins=args.max_coins, top_k=args.top_k)
        glyph_runs, windows = _split_candidates(found, args.top_k)

        for index, candidate in enumerate(glyph_runs, start=1):
            quad_rows.append(
                _quad_row(row["image"], index, "glyph", candidate)
            )
        for index, candidate in enumerate(windows, start=len(glyph_runs) + 1):
            quad_rows.append(_quad_row(row["image"], index, "window", candidate))

        circles = _unique_circles(found)
        render_sheet(image, path.stem, row["year"], glyph_runs, len(found), circles).save(
            out / f"{path.stem}_sheet.png"
        )
        print(f"[{number}/{len(batch)}] {path.stem[:40]:42} glyph={len(glyph_runs):2} "
              f"all={len(found):3}", flush=True)

    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(quad_rows)

    print(f"\n{len(batch) - missing}/{len(batch)} sheets, {len(quad_rows)} candidate quads "
          f"-> {out}")
    if missing:
        print(f"{missing} images were missing from disk", file=sys.stderr)
    return 0


def _quad_row(image: str, index: int, source: str, candidate: DateCandidate) -> dict[str, object]:
    quad = candidate_quad(candidate)
    row: dict[str, object] = {
        "image": image,
        "candidate": index,
        "source": source,
        "score": round(candidate.score, 4),
        "digits": candidate.digits,
        "angle_deg": round(candidate.angle_deg, 2),
    }
    # The quad is the annotation itself: p1 top-left, then clockwise.
    for point_index, (x, y) in enumerate(quad, start=1):
        row[f"x{point_index}"] = int(round(x))
        row[f"y{point_index}"] = int(round(y))
    return row


# --------------------------------------------------------------------------- check


def _read_quad(row: dict[str, str]) -> list[tuple[float, float]] | None:
    """The four annotated corners, or None when the row says 'no legible date'."""
    values = [(row.get(name) or "").strip() for name in QUAD_COLUMNS]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("partially filled quad")
    try:
        numbers = [float(value) for value in values]
    except ValueError:
        raise ValueError("a coordinate is not a number") from None
    return [(numbers[index], numbers[index + 1]) for index in range(0, len(numbers), 2)]


def _cross(origin, a, b) -> float:
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])


def validate_annotation(
    points: Sequence[tuple[float, float]],
    size: tuple[int, int],
) -> tuple[list[str], list[str]]:
    """Hard errors and soft warnings for one annotated quad, in image coordinates."""
    errors: list[str] = []
    warnings: list[str] = []
    width, height = size

    for point in points:
        if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
            errors.append(
                f"corner outside the frame ({point[0]:.0f},{point[1]:.0f}) vs {width}x{height}"
            )
    if errors:
        return errors, warnings

    # Clockwise in image coordinates gives four positive turns; a bow-tie mixes the signs.
    turns = [_cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)]
    if not (min(turns) > 0.0 or max(turns) < 0.0):
        errors.append("the corners cross each other or repeat a point - check the click order")
        return errors, warnings
    if max(turns) < 0.0:
        warnings.append("corners are counter-clockwise: p1 top-left then p2 top-right is expected, "
                        "so the crop is resampled sideways")

    sides = [
        math.hypot(points[(i + 1) % 4][0] - points[i][0], points[(i + 1) % 4][1] - points[i][1])
        for i in range(4)
    ]
    across = 0.5 * (sides[1] + sides[3])
    if across < 8.0:
        errors.append(f"glyph height {across:.1f} px is too small to read")

    area = 0.5 * abs(
        sum(
            points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]
            for i in range(4)
        )
    )
    if area > 0.25 * width * height:
        warnings.append(
            f"the quad covers {100.0 * area / (width * height):.0f}% of the frame - wrong place?"
        )
    return errors, warnings


def cmd_check(args: argparse.Namespace) -> int:
    annotations = read_manifest(args.annotations)
    manifest = {row["image"]: row for row in read_manifest(args.manifest)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    filled = 0
    empty = 0
    known = set()
    for row_number, row in enumerate(annotations, start=1):
        image = row.get("image", "")
        year = row.get("year") or manifest.get(image, {}).get("year", "")
        stem = Path(image).stem
        tag = f"{stem}_r{row_number:03d}"
        try:
            points = _read_quad(row)
        except ValueError as exc:
            print(f"{tag[:60]:62} ERROR {exc}")
            known.add(stem)
            continue
        if points is None:
            empty += 1
            known.add(stem)
            continue
        path = Path(image)
        if not path.is_file():
            print(f"{tag[:60]:62} ERROR image not found")
            known.add(stem)
            continue
        with Image.open(path) as handle:
            image_obj = handle.convert("RGB")
        errors, warnings = validate_annotation(points, image_obj.size)
        kind = (row.get("kind") or "").strip().lower()
        if kind and kind not in {"engraved", "slab"}:
            warnings.append(f"kind {kind!r} is neither engraved nor slab")
        known.add(stem)
        for message in errors:
            print(f"{tag[:60]:62} ERROR {message}")
        for message in warnings:
            print(f"{tag[:60]:62} warn  {message}")
        filled += 1
        # One review per row, not per image: a pair photo carries two dates.
        _write_review(image_obj, stem, year, points, out / f"{tag}_review.png")

    print(f"\n{filled} filled, {empty} empty (no date) of {len(annotations)} rows -> {out}")
    if args.sheets:
        batch = Path(args.sheets) / "batch.csv"
        if batch.is_file():
            pending = [row for row in read_manifest(batch) if Path(row["image"]).stem not in known]
            print(f"batch {batch}: {len(pending)} of {len(read_manifest(batch))} rows still empty")
    return 0


def _write_review(image: Image.Image, stem: str, year: str, points, path: Path) -> None:
    """The annotated quad on the original, plus both readings of the crop it produces.

    The second crop is the same quad rotated 180 degrees: round text has no geometric "up", so if
    the corners were entered in the wrong direction the digits are readable there and obviously
    upside down in the first panel.
    """
    scale = min(1.0, PHOTO_MAX[0] / image.width, PHOTO_MAX[1] / image.height)
    photo = image.resize((max(int(image.width * scale), 8), max(int(image.height * scale), 8)),
                         Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(photo)
    scaled = [(x * scale, y * scale) for x, y in points]
    draw.line([*scaled, scaled[0]], fill=TRUTH_COLOUR, width=3)

    crop = crop_upright(image, quad_to_candidate(points), size=CROP_SIZE, margin=0.25)
    panels = [(crop, "as written"), (crop.rotate(180), "rotated 180")]
    canvas = Image.new(
        "RGB",
        (photo.width + 10 + len(panels) * (CROP_SIZE[0] + 10),
         max(photo.height + CAPTION_HEIGHT, CROP_SIZE[1] + CAPTION_HEIGHT)),
        BACKGROUND,
    )
    canvas.paste(photo, (0, 0))
    label_draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(panels):
        left = photo.width + 10 + index * (CROP_SIZE[0] + 10)
        canvas.paste(panel, (left, 2))
        label_draw.text((left + 2, CROP_SIZE[1] + 6), label, fill=(210, 210, 220), font=_font(14))
    label_draw.text(
        (4, photo.height + 4), f"{stem}  year={year}", fill=(210, 210, 220), font=_font(15)
    )
    canvas.save(path)


# --------------------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template", help="write the empty annotation CSV")
    template.add_argument("--manifest", default="data/manifests/all.csv")
    template.add_argument("--out", default="data/annotations/date_boxes.csv")
    template.set_defaults(func=cmd_template)

    sheets = subparsers.add_parser("sheets", help="numbered candidate sheets for a batch")
    sheets.add_argument("--manifest", default="data/manifests/all.csv")
    sheets.add_argument("--out", default="data/annotations/sheets")
    sheets.add_argument("--count", type=int, default=None, help="batch size, None = every row")
    sheets.add_argument("--seed", type=int, default=42)
    sheets.add_argument("--max-coins", type=int, default=2)
    sheets.add_argument("--top-k", type=int, default=8)
    sheets.set_defaults(func=cmd_sheets)

    check = subparsers.add_parser("check", help="validate annotations and render review sheets")
    check.add_argument("--annotations", default="data/annotations/date_boxes.csv")
    check.add_argument("--manifest", default="data/manifests/all.csv")
    check.add_argument("--out", default="data/annotations/review")
    check.add_argument("--sheets", default="data/annotations/sheets",
                       help="batch directory whose batch.csv tracks what is still unannotated")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
