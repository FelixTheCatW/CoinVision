from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from coinvision.cache import ImageCache
from coinvision.dataset import (
    CachedCoinDataset,
    CoinDataset,
    build_cached_transform,
)


def _image(tmp_path: Path, name: str = "a.png", size: tuple[int, int] = (200, 100)) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, (7, 14, 21)).save(path)
    return path


def _manifest(tmp_path: Path, *paths: Path) -> Path:
    rows = [(str(path.as_posix()), "1832") for path in paths]
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["image", "year"]).to_csv(manifest, index=False)
    return manifest


def test_cache_matches_the_resize_pipeline(tmp_path: Path):
    path = _image(tmp_path)
    cache = ImageCache((64, 32))

    cached = cache.get(path)
    with Image.open(path) as image:
        reference = v2.functional.resize(
            v2.functional.to_image(image.convert("RGB")), [32, 64], antialias=True
        )

    assert cached.dtype == torch.uint8
    assert cached.shape == (3, 32, 64)
    assert torch.equal(cached, reference)


def test_cache_preload_is_idempotent(tmp_path: Path):
    first = _image(tmp_path, "a.png")
    second = _image(tmp_path, "b.png", (300, 150))
    cache = ImageCache((64, 32))

    assert cache.preload([first, second]) == 2
    assert cache.preload([first, second]) == 0
    assert len(cache) == 2
    assert cache.memory_bytes() == 2 * 3 * 32 * 64


def test_cached_dataset_matches_plain_dataset(tmp_path: Path):
    first = _image(tmp_path, "a.png")
    second = _image(tmp_path, "b.png", (300, 150))
    manifest = _manifest(tmp_path, first, second)

    plain = CoinDataset(manifest, image_size=(64, 32))
    cached = CachedCoinDataset(manifest, ImageCache((64, 32)))

    for index in range(len(plain)):
        plain_tensor, plain_target = plain[index]
        cached_tensor, cached_target = cached[index]
        assert torch.equal(plain_target, cached_target)
        assert torch.allclose(plain_tensor, cached_tensor, atol=1e-6)


def test_cached_dataset_shares_one_cache_between_splits(tmp_path: Path):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    cache = ImageCache((64, 32))

    first = CachedCoinDataset(manifest, cache)
    second = CachedCoinDataset(manifest, cache)

    assert len(cache) == 1
    assert first.cache is second.cache


def test_cached_dataset_supports_augmentation(tmp_path: Path):
    path = _image(tmp_path)
    manifest = _manifest(tmp_path, path)
    dataset = CachedCoinDataset(manifest, ImageCache((64, 32)), augment=True)

    tensor, target = dataset[0]

    assert tensor.shape == (3, 32, 64)
    assert target.tolist() == [1, 8, 3, 2]


def test_cached_transform_keeps_shape_and_normalises():
    transform = build_cached_transform()
    pixels = torch.randint(0, 255, (3, 32, 64), dtype=torch.uint8)

    tensor = transform(pixels)

    assert tensor.shape == (3, 32, 64)
    assert tensor.dtype == torch.float32
    assert tensor.min() < 0 < tensor.max()
