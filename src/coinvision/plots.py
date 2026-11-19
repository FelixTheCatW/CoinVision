"""Plots for a training run: per-epoch curves and the last-digit confusion matrix."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HISTORY_FILENAME = "history.png"
CONFUSION_FILENAME = "confusion.png"
METRICS_FILENAME = "metrics.json"


def _last_digit_accuracy(metrics: dict) -> float:
    return float(metrics["digit_accuracy"][-1])


def plot_history(
    history: Sequence[dict],
    out_path: str | Path,
    baseline: float | None = None,
    best_epoch: int | None = None,
) -> Path:
    """Loss, last-digit accuracy and per-position accuracy over the epochs."""
    if not history:
        raise ValueError("history is empty")

    epochs = [int(entry["epoch"]) for entry in history]
    train_loss = [float(entry["train"]["loss"]) for entry in history]
    val_loss = [float(entry["val"]["loss"]) for entry in history]
    train_digit = [_last_digit_accuracy(entry["train"]) for entry in history]
    val_digit = [_last_digit_accuracy(entry["val"]) for entry in history]
    num_digits = len(history[0]["val"]["digit_accuracy"])

    figure: Figure
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()

    axes[1].plot(epochs, train_digit, marker="o", label="train")
    axes[1].plot(epochs, val_digit, marker="o", label="val")
    if baseline is not None:
        axes[1].axhline(
            baseline, linestyle="--", color="gray", label=f"majority baseline {baseline:.3f}"
        )
    axes[1].set_title("last-digit accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right")

    for index in range(num_digits):
        axes[2].plot(
            epochs,
            [float(entry["val"]["digit_accuracy"][index]) for entry in history],
            marker="o",
            label=f"d{index}",
        )
    axes[2].set_title("val accuracy per year position")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("accuracy")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend(loc="lower right")

    if best_epoch is not None:
        for axis in axes:
            axis.axvline(best_epoch, linestyle=":", color="tab:red", alpha=0.7)

    figure.suptitle(
        f"training curves (dotted line = best epoch {best_epoch})"
        if best_epoch is not None
        else "training curves"
    )
    figure.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_confusion(report: dict, out_path: str | Path) -> Path:
    """Last-digit confusion matrix: annotated with counts, coloured by row-normalised recall."""
    labels = list(report["labels"])
    if not labels:
        raise ValueError("report has no labels")

    matrix = [[int(report["confusion"][row][column]) for column in labels] for row in labels]
    normalized = []
    for row in matrix:
        support = sum(row)
        normalized.append([value / support if support else 0.0 for value in row])

    figure: Figure
    figure, axis = plt.subplots(figsize=(1.15 * len(labels) + 3.2, 1.0 * len(labels) + 2.6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            shade = normalized[row_index][column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if shade > 0.55 else "black",
                fontsize=9,
            )

    axis.set_xticks(range(len(labels)), [str(label) for label in labels])
    axis.set_yticks(range(len(labels)), [str(label) for label in labels])
    axis.set_xlabel("predicted last digit")
    axis.set_ylabel("true last digit")
    axis.set_title(
        f"val last digit | acc={report['accuracy']:.3f} macro F1={report['macro_f1']:.3f} "
        f"baseline={report['majority_baseline']:.3f} n={report['num_samples']}",
        pad=12,
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="row-normalised")
    figure.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def plot_metrics(metrics: dict, out_dir: str | Path) -> dict[str, Path]:
    """Write history.png and confusion.png for a metrics payload (same schema as metrics.json)."""
    out_dir = Path(out_dir)
    report = metrics["val_last_digit"]
    return {
        "history": plot_history(
            metrics["history"],
            out_dir / HISTORY_FILENAME,
            baseline=float(report["majority_baseline"]),
            best_epoch=int(metrics["best_epoch"]),
        ),
        "confusion": plot_confusion(report, out_dir / CONFUSION_FILENAME),
    }


def render_run(run_dir: str | Path) -> dict[str, Path]:
    """Re-render the plots of a finished run from its metrics.json."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return plot_metrics(metrics, run_dir)
