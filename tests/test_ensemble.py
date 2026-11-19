from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from coinvision.config import ModelConfig
from coinvision.ensemble import build_eval_views, evaluate_ensemble
from coinvision.model import build_model


def _dataset(tmp_path: Path, count: int = 4) -> Path:
    rows = []
    for index in range(count):
        path = tmp_path / f"img-{index}-183{index % 3}.png"
        Image.new("RGB", (128, 64), (index * 30, 80, 160)).save(path)
        rows.append((path.as_posix(), f"183{index % 3}"))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["image", "year"]).to_csv(manifest, index=False)
    return manifest


def _checkpoint(tmp_path: Path, name: str, seed: int) -> Path:
    torch.manual_seed(seed)
    model = build_model(ModelConfig())
    payload = {
        "model_config": asdict(ModelConfig()),
        "image_size": [64, 32],
        "model": model.state_dict(),
    }
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_build_eval_views_counts_and_validation():
    assert len(build_eval_views((64, 32), 1)) == 1
    assert len(build_eval_views((64, 32), 2)) == 2
    assert len(build_eval_views((64, 32), 3)) == 3
    with pytest.raises(ValueError):
        build_eval_views((64, 32), 0)


def test_eval_views_produce_the_model_input_shape():
    from torchvision.transforms import v2

    image = Image.new("RGB", (200, 100), (10, 20, 30))
    for view in build_eval_views((64, 32), 3):
        assert isinstance(view, v2.Compose)
        tensor = view(image)
        assert tensor.shape == (3, 32, 64)
        assert tensor.dtype == torch.float32


def test_evaluate_ensemble_reports_per_view(tmp_path: Path):
    manifest = _dataset(tmp_path)
    first = _checkpoint(tmp_path, "a.pt", seed=1)
    second = _checkpoint(tmp_path, "b.pt", seed=2)

    report = evaluate_ensemble([first, second], manifest, (64, 32), views=2, device_spec="cpu")

    assert report["num_samples"] == 4
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["views"] == 2
    assert len(report["checkpoints"]) == 2
    assert report["labels"] == [0, 1, 2]


def test_evaluate_ensemble_requires_a_checkpoint(tmp_path: Path):
    manifest = _dataset(tmp_path)
    with pytest.raises(ValueError):
        evaluate_ensemble([], manifest, (64, 32), device_spec="cpu")
