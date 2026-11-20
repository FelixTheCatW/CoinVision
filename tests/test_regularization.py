from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from coinvision.config import Config, TrainConfig
from coinvision.train import _build_scheduler, fit, multihead_loss


def _tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index in range(6):
        year = "1832" if index < 4 else "1833"
        path = source / f"img-{index}-{year}.png"
        Image.new("RGB", (64, 32), (index * 20, 90, 200 - index * 20)).save(path)
        rows.append((path.as_posix(), year))
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows[:4], columns=["image", "year"]).to_csv(train_csv, index=False)
    pd.DataFrame(rows[4:], columns=["image", "year"]).to_csv(val_csv, index=False)
    return train_csv, val_csv


def _config(tmp_path: Path, train_csv: Path, val_csv: Path) -> Config:
    config = Config()
    config.data.train_manifest = str(train_csv)
    config.data.val_manifest = str(val_csv)
    config.data.image_size = (64, 32)
    config.model.pretrained = False
    config.train.epochs = 2
    config.train.batch_size = 2
    config.train.num_workers = 0
    config.train.out_dir = str(tmp_path / "runs")
    return config


def test_label_smoothing_softens_confident_predictions():
    # labels are 0 and 2, but the model is confidently sure about class 5: smoothing caps how much
    # probability mass it wants to concentrate on the wrong class, so the loss drops
    logits = torch.zeros(2, 4, 10)
    targets = torch.tensor([[1, 8, 3, 0], [1, 8, 3, 2]])
    logits[:, -1, 5] = 6.0

    plain = float(multihead_loss(logits, targets))
    smoothed = float(multihead_loss(logits, targets, label_smoothing=0.1))

    assert smoothed < plain


def test_scheduler_is_none_by_default():
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    assert _build_scheduler(optimizer, TrainConfig()) is None


def test_cosine_scheduler_decays_over_epochs():
    parameter = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = _build_scheduler(
        optimizer, TrainConfig(lr_schedule="cosine", warmup_epochs=1, epochs=6)
    )

    assert scheduler is not None
    learning_rates = []
    for _ in range(6):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert learning_rates[0] < learning_rates[1]  # warmup
    assert learning_rates[-1] < learning_rates[1]  # decay
    assert all(rate > 0 for rate in learning_rates)


def test_scheduler_rejects_unknown_schedule():
    optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    with pytest.raises(ValueError):
        _build_scheduler(optimizer, TrainConfig(lr_schedule="step"))


def test_fit_runs_with_smoothing_and_cosine(tmp_path: Path):
    train_csv, val_csv = _tiny_dataset(tmp_path)
    config = _config(tmp_path, train_csv, val_csv)
    config.train.label_smoothing = 0.1
    config.train.lr_schedule = "cosine"
    config.train.warmup_epochs = 1

    result = fit(config, device_spec="cpu")

    assert len(result["history"]) == 2
    assert result["metrics_path"].endswith("metrics.json")
