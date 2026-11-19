from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from .analyze import check_image_file
from .utils import parse_year


def build_manifest(
    source_dir: str | Path,
    out_csv: str | Path,
    extensions: tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png", ".bmp"),
    validate: bool = True,
    dedup: bool = True,
) -> dict[str, int]:
    source_dir = Path(source_dir)
    rows: list[tuple[str, str]] = []
    skipped_broken = 0
    skipped_no_year = 0
    duplicates_removed = 0
    seen_content: set[str] = set()

    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        year = parse_year(path)
        if year is None:
            skipped_no_year += 1
            continue

        content: str | None = None
        if validate or dedup:
            probe = check_image_file(path, deep=True)
            if probe["status"] != "ok":
                skipped_broken += 1
                continue
            content = probe["pixel_sha256"]
        if dedup and content is not None:
            if content in seen_content:
                duplicates_removed += 1
                continue
            seen_content.add(content)

        rows.append((str(path.as_posix()), year))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "year"])
        writer.writerows(rows)
    return {
        "written": len(rows),
        "skipped_broken": skipped_broken,
        "skipped_no_year": skipped_no_year,
        "duplicates_removed": duplicates_removed,
    }


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def listing_id(image: str | Path) -> str:
    """Group key shared by the shots of one listing: ``<listing>-<shot>-<year>``.

    Several images usually come from the same product listing (e.g. ``64a5ace...-1-1832`` and
    ``64a5ace...-2-1832``). They are not pixel-identical, so de-duplication keeps them, but they
    must never be split across train and val: that leaks near-duplicates into the evaluation.
    """
    stem = Path(image).stem
    parts = stem.rsplit("-", 2)
    if len(parts) == 3 and parts[0] and parts[1].isdigit() and parts[2].isdigit():
        if len(parts[2]) == 4:
            return parts[0]
    return stem


def _units_by_year(
    rows: list[dict[str, str]], group_by_listing: bool
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Map year -> {unit key: rows}; a unit (listing or single image) is always moved as a block."""
    units: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = listing_id(row["image"]) if group_by_listing else row["image"]
        units[row["year"]][key].append(row)
    return dict(units)


def _write_manifest(path: str | Path, rows: list[dict[str, str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "year"])
        for row in rows:
            writer.writerow([row["image"], row["year"]])
    return path


def split_manifest(
    all_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_by_listing: bool = False,
) -> dict[str, int]:
    """Stratify by year, optionally keeping all shots of one listing inside a single split."""
    rows = read_manifest(all_csv)
    units = _units_by_year(rows, group_by_listing)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for year in sorted(units):
        blocks = list(units[year].items())
        rng.shuffle(blocks)
        # Heavy blocks first so the greedy fill hits the target share without overshooting, and
        # the sort is stable so ties keep the seeded order (the split stays reproducible).
        blocks.sort(key=lambda item: len(item[1]), reverse=True)
        total = sum(len(block) for _, block in blocks)
        n_val = max(1, round(total * val_fraction)) if total > 1 else 0

        year_val: list[dict[str, str]] = []
        year_train: list[dict[str, str]] = []
        for _, block in blocks:
            if len(year_val) < n_val:
                year_val.extend(block)
            else:
                year_train.extend(block)
        train_rows.extend(year_train)
        val_rows.extend(year_val)

    _write_manifest(train_csv, train_rows)
    _write_manifest(val_csv, val_rows)
    return {"train": len(train_rows), "val": len(val_rows)}


def make_folds(
    all_csv: str | Path,
    out_dir: str | Path,
    folds: int = 5,
    seed: int = 42,
    group_by_listing: bool = True,
) -> dict[str, dict[str, int]]:
    """Write <folds> stratified folds as fold<i>_train.csv / fold<i>_val.csv.

    Every fold is a year-stratified split; with ``group_by_listing`` all shots of a listing land
    in the same fold, so validation never sees a near-duplicate of a training image.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")

    rows = read_manifest(all_csv)
    units = _units_by_year(rows, group_by_listing)

    rng = random.Random(seed)
    fold_of: dict[tuple[str, str], int] = {}
    for year in sorted(units):
        blocks = list(units[year].items())
        rng.shuffle(blocks)
        blocks.sort(key=lambda item: len(item[1]), reverse=True)
        sizes = [0] * folds
        for key, block in blocks:
            target = min(range(folds), key=lambda index: (sizes[index], index))
            fold_of[(year, key)] = target
            sizes[target] += len(block)

    out_dir = Path(out_dir)
    counts: dict[str, dict[str, int]] = {}
    for fold in range(folds):
        train_rows: list[dict[str, str]] = []
        val_rows: list[dict[str, str]] = []
        for row in rows:
            key = listing_id(row["image"]) if group_by_listing else row["image"]
            destination = val_rows if fold_of[(row["year"], key)] == fold else train_rows
            destination.append(row)
        _write_manifest(out_dir / f"fold{fold}_train.csv", train_rows)
        _write_manifest(out_dir / f"fold{fold}_val.csv", val_rows)
        counts[f"fold{fold}"] = {"train": len(train_rows), "val": len(val_rows)}
    return counts
