from pathlib import Path

import pandas as pd
from PIL import Image

from coinvision.analyze import (
    ANALYZE_COLUMNS,
    analyze_to_csv,
    check_image_file,
    resolution_stats_from_csv,
    summary_from_csv,
)


def _make_image(path: Path, size: tuple[int, int], mode: str, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


def test_analyze_writes_csv_next_to_directory(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))
    _make_image(images / "b-2-1832.jpg", (10, 8), "RGB", (0, 255, 0))
    _make_image(images / "c-3-1833.png", (20, 16), "L", 128)

    out = analyze_to_csv(images, progress=False)

    assert out == tmp_path / "images_analyze.csv"
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == list(ANALYZE_COLUMNS)
    assert [Path(p).name for p in df["image"]] == [
        "a-1-1832.png",
        "b-2-1832.jpg",
        "c-3-1833.png",
    ]
    assert set(df["status"]) == {"ok"}
    assert df.loc[0, "year"] == 1832
    assert df.loc[0, "width"] == 10
    assert df.loc[2, "color"] == "grayscale"
    assert len(str(df.loc[0, "file_sha256"])) == 64
    assert len(str(df.loc[0, "pixel_sha256"])) == 64


def test_summary_from_csv_counts(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))
    _make_image(images / "b-2-1832.jpg", (10, 8), "RGB", (0, 255, 0))
    _make_image(images / "c-3-1833.png", (20, 16), "L", 128)
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    (images / "empty-4-1834.png").write_bytes(b"")

    stats = summary_from_csv(analyze_to_csv(images, progress=False))

    assert stats["Files"] == 4
    assert stats["Integrity"] == "3/4 ok, 1 problems"
    assert stats["Problem reasons"] == {"empty": 1}
    assert stats["Color"] == {"color": 2, "grayscale": 1}
    assert stats["Year in name"] == "4 with / 0 without"
    assert stats["Year counts"] == {"1832": 2, "1833": 1, "1834": 1}
    assert stats["Resolution"] == "2 unique, min [10, 8], max [20, 16]"


def test_broken_file_and_format_mismatch(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir(parents=True)
    (images / "broken-1832.webp").write_text("nope", encoding="utf-8")
    Image.new("RGB", (10, 10), (5, 5, 5)).save(images / "mismatch-1833.jpg", format="PNG")

    out = analyze_to_csv(images, progress=False)
    df = pd.read_csv(out)
    rows = {
        Path(name).name: record
        for name, record in zip(df["image"], df.to_dict("records"), strict=True)
    }

    broken = rows["broken-1832.webp"]
    assert broken["status"] == "fail"
    assert broken["reason"] == "not_an_image"

    stats = summary_from_csv(out)
    assert stats["Format mismatch"] == 1
    assert stats["Problem reasons"] == {"not_an_image": 1}


def test_check_image_file_status_and_hash(tmp_path: Path):
    ok = tmp_path / "ok.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(ok)
    result = check_image_file(ok)
    assert result["status"] == "ok"
    assert result["width"] == 4 and result["height"] == 4
    assert len(result["pixel_sha256"]) == 64

    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    empty_result = check_image_file(empty)
    assert empty_result["status"] == "fail"
    assert empty_result["reason"] == "empty"


def test_resolution_stats_percentiles(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (100, 50), "RGB", (1, 1, 1))
    _make_image(images / "b-1-1832.png", (200, 100), "RGB", (2, 2, 2))
    _make_image(images / "c-1-1832.png", (400, 200), "RGB", (3, 3, 3))

    table = resolution_stats_from_csv(analyze_to_csv(images, progress=False))

    assert table.loc["width", "count"] == 3
    assert table.loc["width", "min"] == 100
    assert table.loc["width", "p50"] == 200
    assert table.loc["width", "max"] == 400
    assert table.loc["aspect_ratio", "min"] == 2.0
    assert table.loc["aspect_ratio", "max"] == 2.0


def test_analyze_shallow_mode_skips_pixel_hash(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))

    df = pd.read_csv(analyze_to_csv(images, deep_check=False, progress=False))

    assert df.loc[0, "status"] == "ok"
    assert pd.isna(df.loc[0, "pixel_sha256"])


def test_analyze_reports_unreadable_file(tmp_path: Path, monkeypatch):
    import coinvision.analyze as analyze_module

    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))

    def deny(path):
        raise OSError("access denied")

    monkeypatch.setattr(analyze_module, "sha256_file", deny)

    df = pd.read_csv(analyze_to_csv(images, progress=False))

    assert df.loc[0, "status"] == "fail"
    assert df.loc[0, "reason"] == "unreadable"
    assert pd.isna(df.loc[0, "file_sha256"])
