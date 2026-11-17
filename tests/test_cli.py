import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from coinvision.cli import main


def _make_image(path: Path, color=(10, 20, 30), size=(20, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_cli_config_prints_sections(capsys):
    assert main(["config"]) == 0

    out = capsys.readouterr().out
    assert "[model]" in out
    assert "resnet18" in out
    assert "image_size = (448, 224)" in out


def test_cli_config_falls_back_to_defaults(tmp_path, capsys):
    code = main(["--config", str(tmp_path / "missing.toml"), "config"])

    assert code == 0
    assert "built-in defaults" in capsys.readouterr().err


def test_cli_analyze_is_deep_by_default(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png")

    assert main(["analyze", str(images), "--no-progress"]) == 0

    df = pd.read_csv(tmp_path / "images_analyze.csv")
    assert len(df) == 1
    assert len(str(df.loc[0, "pixel_sha256"])) == 64


def test_cli_analyze_shallow_flag(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "a-1-1832.png")

    assert main(["analyze", str(images), "--shallow", "--no-progress"]) == 0

    df = pd.read_csv(tmp_path / "images_analyze.csv")
    assert pd.isna(df.loc[0, "pixel_sha256"])


def test_cli_hashes_writes_json(tmp_path):
    source = tmp_path / "source"
    _make_image(source / "a.png", color=(1, 2, 3))
    _make_image(source / "b.png", color=(1, 2, 3))
    report_path = tmp_path / "report.json"

    assert main(["hashes", str(source), "--json", str(report_path), "--no-progress"]) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["files_hashed"] == 2
    assert report["content_duplicates"]["groups"] == 1
    assert report["content_duplicates"]["extra_files"] == 1


def test_cli_manifest_and_split(tmp_path):
    source = tmp_path / "source"
    names = ("a-1-1832.png", "b-1-1832.png", "c-1-1833.png", "d-1-1833.png")
    for index, name in enumerate(names):
        _make_image(source / name, color=(index * 20 + 5, index * 7 + 3, index + 1))

    manifest = tmp_path / "all.csv"
    assert main(["manifest", str(source), str(manifest)]) == 0

    rows = pd.read_csv(manifest)
    assert len(rows) == 4
    assert set(rows["year"]) == {1832, 1833}

    train_csv, val_csv = tmp_path / "train.csv", tmp_path / "val.csv"
    code = main(
        [
            "split",
            str(manifest),
            "--train",
            str(train_csv),
            "--val",
            str(val_csv),
            "--val-fraction",
            "0.5",
        ]
    )

    assert code == 0
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    assert len(train) == 2
    assert len(val) == 2
    assert set(train["year"]) == {1832, 1833}
    assert set(train["image"]).isdisjoint(set(val["image"]))


def test_cli_reports_missing_directory(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope"), "--no-progress"]) == 1
    assert "not a directory" in capsys.readouterr().err


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
