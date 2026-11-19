from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from coinvision.cache import ImageCache
from coinvision.config import ModelConfig
from coinvision.dataset import STACK_VIEWS, StackedViewsDataset, view_window
from coinvision.model import build_model


def _image(tmp_path: Path, name: str = "a.png", size: tuple[int, int] = (400, 200)) -> Path:
    path = tmp_path / name
    width, height = size
    xs = torch.arange(width).view(1, width).expand(height, width)
    ys = torch.arange(height).view(height, 1).expand(height, width)
    array = torch.stack([xs % 256, ys % 256, (xs + ys) % 256], dim=-1).to(torch.uint8)
    Image.fromarray(array.numpy().astype("uint8")).save(path)
    return path


def _manifest(tmp_path: Path, path: Path) -> Path:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([(path.as_posix(), "1832")], columns=["image", "year"]).to_csv(
        manifest, index=False
    )
    return manifest


@pytest.mark.parametrize("anchor", ["centre", "top_left", "bottom_right"])
def test_view_window_stays_inside_the_image(anchor):
    left, top, width, height = view_window(1608, 800, 2.0, 0.35, anchor)

    assert 0 <= left and left + width <= 1608
    assert 0 <= top and top + height <= 800
    assert width == pytest.approx(2 * height, abs=2)


def test_view_window_anchors_are_in_their_corners():
    assert view_window(1600, 800, 2.0, 1.0, "centre")[:2] == (0, 0)
    left, top, width, height = view_window(1600, 800, 2.0, 0.35, "bottom_right")
    assert (left + width, top + height) == (1600, 800)
    assert view_window(1600, 800, 2.0, 0.35, "top_left")[:2] == (0, 0)


def test_stacked_dataset_has_nine_channels(tmp_path: Path):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    dataset = StackedViewsDataset(manifest, ImageCache(None), (64, 32))

    tensor, target = dataset[0]

    assert tensor.shape == (3 * len(STACK_VIEWS), 32, 64)
    assert target.tolist() == [1, 8, 3, 2]


def test_stacked_dataset_rejects_unknown_view(tmp_path: Path):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    with pytest.raises(ValueError):
        StackedViewsDataset(manifest, ImageCache(None), (64, 32), views=("full", "back"))


def test_model_accepts_the_stacked_channels_and_keeps_pretrained_filters():
    model = build_model(ModelConfig(), in_channels=9)

    assert model.backbone[0].weight.shape[1] == 9
    output = model(torch.zeros(2, 9, 64, 64))

    assert output.shape == (2, 4, 10)


def test_model_rejects_bad_channel_count():
    with pytest.raises(ValueError):
        build_model(ModelConfig(), in_channels=4)


def test_checkpoint_records_the_channel_count(tmp_path: Path):
    model = build_model(ModelConfig(), in_channels=9)
    payload = {
        "model_config": asdict(ModelConfig()),
        "image_size": [64, 32],
        "model": model.state_dict(),
        "in_channels": 9,
    }
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)

    from coinvision.infer import CoinPredictor

    predictor = CoinPredictor.from_checkpoint(path, "cpu")
    assert predictor.model.backbone[0].weight.shape[1] == 9
