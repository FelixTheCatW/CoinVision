from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from coinvision.cache import ImageCache
from coinvision.dataset import (
    ANCHOR_EVAL_SCALE,
    CROP_ANCHORS,
    CroppedCoinDataset,
    crop_box,
)


def _image(tmp_path: Path, name: str = "a.png", size: tuple[int, int] = (400, 200)) -> Path:
    path = tmp_path / name
    # a gradient so crops taken from different places differ in content
    width, height = size
    xs = torch.arange(width).view(1, width).expand(height, width)
    ys = torch.arange(height).view(height, 1).expand(height, width)
    array = torch.stack([xs % 256, ys % 256, (xs + ys) % 256], dim=-1).to(torch.uint8)
    Image.fromarray(array.numpy().astype("uint8")).save(path)
    return path


def _manifest(tmp_path: Path, *paths: Path) -> Path:
    rows = [(path.as_posix(), "1832") for path in paths]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["image", "year"]).to_csv(manifest, index=False)
    return manifest


def test_native_cache_keeps_the_original_size(tmp_path: Path):
    path = _image(tmp_path, size=(400, 200))
    cache = ImageCache(None)

    tensor = cache.get(path)

    assert tensor.shape == (3, 200, 400)
    assert tensor.dtype == torch.uint8


@pytest.mark.parametrize("anchor", CROP_ANCHORS)
def test_crop_box_stays_inside_the_image(anchor):
    for width, height in ((1608, 800), (1125, 800), (500, 240)):
        scale = ANCHOR_EVAL_SCALE[anchor]
        left, top, crop_w, crop_h = crop_box(width, height, 2.0, scale, anchor, 0.1, 1.0, 1.0)

        assert 0 <= left and left + crop_w <= width
        assert 0 <= top and top + crop_h <= height
        assert crop_w == pytest.approx(2 * crop_h, abs=2)


def test_crop_box_anchors_sit_in_their_corners():
    width, height = 1600, 800
    _, date_top, _, date_h = crop_box(width, height, 2.0, 0.35, "date", 0.0, 0.0, 0.0)
    date_left, _, date_w, _ = crop_box(width, height, 2.0, 0.35, "date", 0.0, 0.0, 0.0)
    assert date_left + date_w == width
    assert date_top + date_h == height

    label_left, label_top, _, _ = crop_box(width, height, 2.0, 0.35, "label", 0.0, 0.0, 0.0)
    assert (label_left, label_top) == (0, 0)


@pytest.mark.parametrize("variant", ["uniform", "date", "label"])
def test_eval_views_are_deterministic(tmp_path: Path, variant: str):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    dataset = CroppedCoinDataset(manifest, ImageCache(None), (64, 32), variant=variant)

    first, target = dataset[0]
    second, _ = dataset[0]

    assert first.shape == (3, 32, 64)
    assert torch.equal(first, second)
    assert target.tolist() == [1, 8, 3, 2]


def test_random_views_differ_between_draws(tmp_path: Path):
    path = _image(tmp_path, size=(400, 200))
    manifest = _manifest(tmp_path, path)
    dataset = CroppedCoinDataset(manifest, ImageCache(None), (64, 32), variant="random")

    views = {dataset[0][0].numpy().tobytes() for _ in range(6)}

    assert len(views) > 1


def test_crop_dataset_rejects_unknown_variant(tmp_path: Path):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    with pytest.raises(ValueError):
        CroppedCoinDataset(manifest, ImageCache(None), (64, 32), variant="middle")
