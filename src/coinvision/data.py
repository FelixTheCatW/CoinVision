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
            if not probe["ok"]:
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


def split_manifest(
    all_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    rows = read_manifest(all_csv)
    by_year: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_year[row["year"]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for year in sorted(by_year):
        items = by_year[year]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    for path, split_rows in ((train_csv, train_rows), (val_csv, val_rows)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["image", "year"])
            for row in split_rows:
                writer.writerow([row["image"], row["year"]])

    return {"train": len(train_rows), "val": len(val_rows)}
