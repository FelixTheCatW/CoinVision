import csv
from pathlib import Path

from PIL import Image

from coinvision.analyze import (
    ANALYZE_COLUMNS,
    analyze_to_csv,
    check_image_file,
    read_analyze_csv,
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
    rows = read_analyze_csv(out)
    assert [r["image"].split("/")[-1] for r in rows] == [
        "a-1-1832.png",
        "b-2-1832.jpg",
        "c-3-1833.png",
    ]
    assert list(rows[0].keys()) == list(ANALYZE_COLUMNS)
    assert rows[0]["status"] == "ok"
    assert rows[0]["year"] == "1832"
    assert rows[0]["width"] == "10"
    assert rows[2]["color"] == "grayscale"
    assert len(rows[0]["file_sha256"]) == 64
    assert len(rows[0]["pixel_sha256"]) == 64


def test_summary_from_csv_counts(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))
    _make_image(images / "b-2-1832.jpg", (10, 8), "RGB", (0, 255, 0))
    _make_image(images / "c-3-1833.png", (20, 16), "L", 128)
    (images / "notes.txt").write_text("not an image", encoding="utf-8")
    (images / "empty-4-1834.png").write_bytes(b"")

    out = analyze_to_csv(images, progress=False)
    stats = summary_from_csv(out)

    assert stats["files"] == 4
    assert stats["integrity"]["ok"] == 3
    assert stats["integrity"]["problems"] == 1
    assert stats["integrity"]["by_reason"]["empty"] == 1
    assert stats["color"] == {"color": 2, "grayscale": 1}
    assert stats["year"]["with_year"] == 4
    assert stats["year"]["distribution"] == {"1832": 2, "1833": 1, "1834": 1}
    assert stats["resolution"]["min"] == [10, 8]
    assert stats["resolution"]["max"] == [20, 16]


def test_broken_file_row_and_format_mismatch(tmp_path: Path):
    images = tmp_path / "images"
    (images).mkdir(parents=True)
    (images / "broken-1832.webp").write_text("nope", encoding="utf-8")
    Image.new("RGB", (10, 10), (5, 5, 5)).save(images / "mismatch-1833.jpg", format="PNG")

    out = analyze_to_csv(images, progress=False)
    rows = {Path(r["image"]).name: r for r in read_analyze_csv(out)}

    assert rows["broken-1832.webp"]["status"] == "not_an_image"
    assert rows["broken-1832.webp"]["width"] == ""
    stats = summary_from_csv(out)
    assert stats["format_mismatch"]["count"] == 1


def test_check_image_file_returns_pixel_hash(tmp_path: Path):
    ok = tmp_path / "ok.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(ok)
    result = check_image_file(ok)
    assert result["ok"] is True
    assert result["width"] == 4 and result["height"] == 4
    assert len(result["pixel_sha256"]) == 64


def test_csv_is_plain_tabular(tmp_path: Path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png", (10, 8), "RGB", (255, 0, 0))
    out = analyze_to_csv(images, progress=False)
    with out.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == list(ANALYZE_COLUMNS)
