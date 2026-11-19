import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from coinvision.config import Config
from coinvision.infer import CoinPredictor
from coinvision.train import (
    compute_class_weights,
    fit,
    format_last_digit_report,
    last_digit_report,
    multihead_loss,
)

YEARS = ("1832", "1832", "1832", "1832", "1833", "1833", "1833", "1833")


def _tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, year in enumerate(YEARS):
        path = source / f"img-{index}-{year}.png"
        Image.new("RGB", (64, 32), (index * 25, 128, 200 - index * 20)).save(path)
        rows.append((path.as_posix(), year))

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    columns = ["image", "year"]
    pd.DataFrame(rows[:6], columns=columns).to_csv(train_csv, index=False)
    pd.DataFrame(rows[6:], columns=columns).to_csv(val_csv, index=False)
    return train_csv, val_csv


def _config(tmp_path: Path, train_csv: Path, val_csv: Path) -> Config:
    config = Config()
    config.data.train_manifest = str(train_csv)
    config.data.val_manifest = str(val_csv)
    config.data.image_size = (64, 32)
    config.model.pretrained = False
    config.train.epochs = 1
    config.train.batch_size = 3
    config.train.num_workers = 0
    config.train.out_dir = str(tmp_path / "runs")
    return config


def test_fit_writes_checkpoints_and_predicts(tmp_path: Path):
    train_csv, val_csv = _tiny_dataset(tmp_path)
    config = _config(tmp_path, train_csv, val_csv)

    result = fit(config, device_spec="cpu")

    assert result["device"] == "cpu"
    assert len(result["history"]) == 1
    assert 0.0 <= result["history"][0]["val"]["year_accuracy"] <= 1.0
    assert (tmp_path / "runs" / "best.pt").is_file()
    assert (tmp_path / "runs" / "last.pt").is_file()

    metrics_path = tmp_path / "runs" / "metrics.json"
    assert metrics_path.is_file()
    assert result["metrics_path"] == str(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["best_epoch"] == 1
    assert metrics["epochs_run"] == 1
    assert metrics["val_last_digit"]["num_samples"] == 2
    assert metrics["config"]["data"]["augment"] is False

    for name in ("history", "confusion"):
        plot_path = Path(result["plots"][name])
        assert plot_path.is_file()
        assert plot_path.parent == tmp_path / "runs"
        assert plot_path.stat().st_size > 0

    predictor = CoinPredictor.from_checkpoint(tmp_path / "runs" / "best.pt", device_spec="cpu")
    prediction = predictor.predict(tmp_path / "source" / "img-0-1832.png")

    assert len(prediction["year"]) == 4
    assert prediction["year"].isdigit()
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert len(prediction["digit_confidence"]) == 4


def test_fit_with_augmentation_reports_last_digit(tmp_path: Path):
    train_csv, val_csv = _tiny_dataset(tmp_path)
    config = _config(tmp_path, train_csv, val_csv)
    config.data.augment = True

    result = fit(config, device_spec="cpu")

    report = result["val_last_digit"]
    assert report["num_samples"] == 2
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["labels"] == [3]
    assert report["majority_baseline"] == 1.0


def test_last_digit_report_confusion_and_f1():
    report = last_digit_report([2, 2, 2, 1], [2, 2, 1, 1], num_classes=10)

    assert report["num_samples"] == 4
    assert report["accuracy"] == 0.75
    assert report["majority_baseline"] == 0.75
    assert report["labels"] == [1, 2]
    assert report["confusion"][2][1] == 1
    assert report["per_class"]["2"]["recall"] == pytest.approx(2 / 3)
    assert report["per_class"]["1"]["precision"] == 0.5
    assert 0.0 <= report["macro_f1"] <= 1.0


def test_last_digit_report_rejects_bad_input():
    with pytest.raises(ValueError):
        last_digit_report([], [])
    with pytest.raises(ValueError):
        last_digit_report([1, 2], [1])


def test_format_last_digit_report_prints_header_and_accuracy():
    text = format_last_digit_report(last_digit_report([2, 1], [2, 1]))

    assert "acc=1.000" in text
    assert "true\\pred" in text
    assert text.count("\n") >= 2


def test_seeded_shuffle_generator_repeats_the_same_batches():
    def order() -> list[int]:
        loader = DataLoader(
            TensorDataset(torch.arange(12)),
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )
        return [int(value) for (batch,) in loader for value in batch]

    assert order() == order()


def test_multihead_loss_is_mean_over_heads():
    logits = torch.zeros(2, 4, 10)
    targets = torch.zeros(2, 4, dtype=torch.long)

    baseline = float(multihead_loss(logits, targets))
    confident = logits.clone()
    confident[:, 0, 0] = 5.0

    assert float(multihead_loss(confident, targets)) < baseline


def test_compute_class_weights_balances_rare_classes():
    samples = [(None, [1, 8, 3, 2])] * 6 + [(None, [1, 8, 3, 0])] * 2

    weights = compute_class_weights(samples, num_digits=4, num_classes=10)

    last = weights[-1]
    # 8 samples over 2 present classes: 8 / (2 * 6) and 8 / (2 * 2)
    assert float(last[2]) == pytest.approx(8 / 12)
    assert float(last[0]) == pytest.approx(8 / 4)
    assert float(last[5]) == 0.0
    # the constant prefix position keeps weight 1.0 for its single class
    assert float(weights[0][1]) == pytest.approx(1.0)
    assert float(weights[0][8]) == 0.0


def test_multihead_loss_upweights_rare_class():
    logits = torch.zeros(2, 4, 10)
    targets = torch.tensor([[1, 8, 3, 0], [1, 8, 3, 2]])
    # the rare class (0) is confidently wrong, the common one (2) is confidently right, so
    # upweighting class 0 must raise the mean loss
    logits[0, -1, 0] = -3.0
    logits[1, -1, 2] = 3.0

    plain = float(multihead_loss(logits, targets))
    weights = [torch.ones(10) for _ in range(4)]
    weights[-1][0] = 5.0

    assert float(multihead_loss(logits, targets, class_weights=weights)) > plain


def test_fit_rejects_unknown_class_weighting(tmp_path: Path):
    train_csv, val_csv = _tiny_dataset(tmp_path)
    config = _config(tmp_path, train_csv, val_csv)
    config.train.class_weighting = "sqrt-balanced"

    with pytest.raises(ValueError):
        fit(config, device_spec="cpu")


def test_fit_runs_with_balanced_class_weights(tmp_path: Path):
    train_csv, val_csv = _tiny_dataset(tmp_path)
    config = _config(tmp_path, train_csv, val_csv)
    config.train.class_weighting = "balanced"

    result = fit(config, device_spec="cpu")

    assert result["metrics_path"].endswith("metrics.json")
    assert result["val_last_digit"]["num_samples"] == 2
