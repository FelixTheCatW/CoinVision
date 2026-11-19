import csv
from pathlib import Path

import pytest

from coinvision.data import listing_id, make_folds, read_manifest, split_manifest


def _manifest(tmp_path: Path, years=("1831", "1832", "1833"), listings=6, shots=2) -> Path:
    """Synthetic manifest: <listing>-<shot>-<year>, several shots per listing."""
    rows = []
    for year in years:
        for listing in range(listings):
            for shot in range(1, shots + 1):
                name = f"listing{listing}{year}-{shot}-{year}.webp"
                rows.append((f"data/source/{name}", year))

    path = tmp_path / "all.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "year"])
        writer.writerows(rows)
    return path


def test_listing_id_strips_shot_and_year():
    prefix = "data/source/64a5ace8330832a0c2dfa5ba"
    assert listing_id(f"{prefix}-1-1832.webp") == "64a5ace8330832a0c2dfa5ba"
    assert listing_id(f"{prefix}-2-1833.webp") == "64a5ace8330832a0c2dfa5ba"
    assert listing_id("data/source/plain.webp") == "plain"
    assert listing_id("data/source/abc-x-1832.webp") == "abc-x-1832"


def test_grouped_split_keeps_listings_together(tmp_path: Path):
    manifest = _manifest(tmp_path)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"

    counts = split_manifest(
        manifest, train_csv, val_csv, val_fraction=0.2, seed=42, group_by_listing=True
    )

    train_ids = {listing_id(row["image"]) for row in read_manifest(train_csv)}
    val_ids = {listing_id(row["image"]) for row in read_manifest(val_csv)}
    assert train_ids.isdisjoint(val_ids)
    assert counts["train"] + counts["val"] == 36
    # 6 listings of 2 shots per year -> one listing (2 shots) per year lands in val
    assert counts["val"] == 6


def test_ungrouped_split_can_still_leak(tmp_path: Path):
    manifest = _manifest(tmp_path)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"

    split_manifest(manifest, train_csv, val_csv, val_fraction=0.5, seed=7, group_by_listing=False)

    train_ids = {listing_id(row["image"]) for row in read_manifest(train_csv)}
    val_ids = {listing_id(row["image"]) for row in read_manifest(val_csv)}
    # With half of the shots of each listing on each side the groups overlap by construction.
    assert train_ids & val_ids


def test_split_is_deterministic_and_stratified(tmp_path: Path):
    manifest = _manifest(tmp_path)
    first = tmp_path / "t1.csv"
    second = tmp_path / "t2.csv"
    val_first = tmp_path / "v1.csv"
    val_second = tmp_path / "v2.csv"

    split_manifest(manifest, first, val_first, seed=5, group_by_listing=True)
    split_manifest(manifest, second, val_second, seed=5, group_by_listing=True)

    assert read_manifest(val_first) == read_manifest(val_second)
    val_years = {row["year"] for row in read_manifest(val_first)}
    assert val_years == {"1831", "1832", "1833"}


def test_make_folds_covers_every_row_once_and_keeps_groups(tmp_path: Path):
    manifest = _manifest(tmp_path)
    out_dir = tmp_path / "folds"

    counts = make_folds(manifest, out_dir, folds=3, seed=42, group_by_listing=True)

    assert set(counts) == {"fold0", "fold1", "fold2"}
    val_rows_per_fold = [
        read_manifest(out_dir / f"fold{index}_val.csv") for index in range(3)
    ]
    seen: list[str] = []
    for rows in val_rows_per_fold:
        seen.extend(row["image"] for row in rows)
    assert sorted(seen) == sorted(row["image"] for row in read_manifest(manifest))
    for rows in val_rows_per_fold:
        listings = {listing_id(row["image"]) for row in rows}
        # every listing contributes exactly its shot count, so no group is split inside a fold
        assert len(rows) == 2 * len(listings)

    for index, rows in enumerate(val_rows_per_fold):
        train_rows = read_manifest(out_dir / f"fold{index}_train.csv")
        train_ids = {listing_id(row["image"]) for row in train_rows}
        val_ids = {listing_id(row["image"]) for row in rows}
        assert train_ids.isdisjoint(val_ids)
        assert {row["year"] for row in rows} == {"1831", "1832", "1833"}
        assert len(train_rows) + len(rows) == 36


def test_make_folds_rejects_single_fold(tmp_path: Path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError):
        make_folds(manifest, tmp_path / "folds", folds=1)
