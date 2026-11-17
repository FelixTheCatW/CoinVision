from pathlib import Path

import pytest
import torch
from PIL import Image

from coinvision.dataset import CoinDataset, build_transform, digits_to_year, year_to_digits


def test_year_to_digits():
    assert year_to_digits("1832") == [1, 8, 3, 2]
    assert year_to_digits(1835) == [1, 8, 3, 5]


@pytest.mark.parametrize("bad", ["183", "18322", "18a2", ""])
def test_year_to_digits_rejects_bad_labels(bad):
    with pytest.raises(ValueError):
        year_to_digits(bad)


def test_digits_to_year_roundtrip():
    assert digits_to_year(year_to_digits("1832")) == "1832"


def test_dataset_item_shape(tmp_path: Path):
    image_path = tmp_path / "a.png"
    Image.new("RGB", (200, 100), (1, 2, 3)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"image,year\n{image_path.as_posix()},1832\n", encoding="utf-8")

    dataset = CoinDataset(manifest, image_size=(64, 32))
    tensor, target = dataset[0]

    assert len(dataset) == 1
    assert tensor.shape == (3, 32, 64)
    assert target.tolist() == [1, 8, 3, 2]
    assert target.dtype == torch.long


def test_transform_output_shape_and_dtype():
    transform = build_transform((48, 24))

    tensor = transform(Image.new("RGB", (100, 100), (255, 255, 255)))

    assert tensor.shape == (3, 24, 48)
    assert tensor.dtype == torch.float32
