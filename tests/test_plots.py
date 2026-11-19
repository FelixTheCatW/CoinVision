import json
from pathlib import Path

import pytest

from coinvision.plots import (
    CONFUSION_FILENAME,
    HISTORY_FILENAME,
    plot_confusion,
    plot_history,
    plot_metrics,
    render_run,
)
from coinvision.train import last_digit_report


def _history(epochs: int = 3, num_digits: int = 4) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        val_digits = [1.0, 1.0, 1.0, 0.2 + 0.1 * epoch]
        history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": 1.5 / epoch,
                    "digit_accuracy": [1.0, 1.0, 1.0, 0.3 + 0.1 * epoch],
                    "year_accuracy": 0.3 + 0.1 * epoch,
                },
                "val": {
                    "loss": 2.0 / epoch,
                    "digit_accuracy": val_digits,
                    "year_accuracy": val_digits[-1],
                },
            }
        )
    return history


def _report() -> dict:
    targets = [2, 2, 1, 7]
    predictions = [2, 1, 1, 2]
    return last_digit_report(targets, predictions)


def test_plot_history_writes_png(tmp_path: Path):
    path = plot_history(_history(), tmp_path / "history.png", baseline=0.333, best_epoch=2)

    assert path.is_file()
    assert path.stat().st_size > 5_000
    assert path.suffix == ".png"


def test_plot_history_without_baseline_or_best_epoch(tmp_path: Path):
    path = plot_history(_history(epochs=1), tmp_path / "history.png")

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_history_rejects_empty_history(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_history([], tmp_path / "history.png")


def test_plot_confusion_writes_png(tmp_path: Path):
    path = plot_confusion(_report(), tmp_path / "confusion.png")

    assert path.is_file()
    assert path.stat().st_size > 5_000


def test_plot_confusion_rejects_empty_report(tmp_path: Path):
    with pytest.raises(ValueError):
        plot_confusion({"labels": []}, tmp_path / "confusion.png")


def test_plot_metrics_writes_both_plots(tmp_path: Path):
    metrics = {
        "best_epoch": 2,
        "history": _history(),
        "val_last_digit": _report(),
    }

    paths = plot_metrics(metrics, tmp_path / "run")

    assert set(paths) == {"history", "confusion"}
    assert (tmp_path / "run" / HISTORY_FILENAME).is_file()
    assert (tmp_path / "run" / CONFUSION_FILENAME).is_file()


def test_render_run_reads_metrics_json(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = {"best_epoch": 1, "history": _history(epochs=1), "val_last_digit": _report()}
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    paths = render_run(run_dir)

    assert paths["history"].is_file()
    assert paths["confusion"].is_file()


def test_render_run_requires_metrics_json(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        render_run(tmp_path)
