from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cache import ImageCache
from .config import Config
from .dataset import (
    STACK_VIEWS,
    CachedCoinDataset,
    CoinDataset,
    StackedViewsDataset,
    build_transform,
)
from .model import build_model
from .utils import resolve_device, set_deterministic, set_seed


@dataclass
class EpochMetrics:
    loss: float
    digit_accuracy: list[float]
    year_accuracy: float

    def format(self, prefix: str) -> str:
        digits = " ".join(
            f"d{index}={value:.3f}" for index, value in enumerate(self.digit_accuracy)
        )
        return f"{prefix} loss={self.loss:.4f} year={self.year_accuracy:.3f} {digits}"


def multihead_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the digit heads. logits: [B, D, C], targets: [B, D]."""
    losses = [
        nn.functional.cross_entropy(logits[:, index], targets[:, index])
        for index in range(logits.shape[1])
    ]
    return torch.stack(losses).mean()


def _collect(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    hits: torch.Tensor | None = None
    year_hits = 0

    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.detach().argmax(dim=-1)
        batch_hits = (predictions == targets).sum(dim=0)
        hits = batch_hits if hits is None else hits + batch_hits
        year_hits += int((predictions == targets).all(dim=1).sum())
        total_loss += float(loss.detach()) * images.size(0)
        total += images.size(0)

    if hits is None or total == 0:
        raise ValueError("no batches processed: the dataset or manifest is empty")

    return EpochMetrics(
        loss=total_loss / total,
        digit_accuracy=[float(value) / total for value in hits],
        year_accuracy=year_hits / total,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion,
    max_batches: int | None = None,
) -> EpochMetrics:
    return _collect(model, loader, device, criterion, optimizer, max_batches)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    max_batches: int | None = None,
) -> EpochMetrics:
    return _collect(model, loader, device, criterion, None, max_batches)


def last_digit_report(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int = 10
) -> dict:
    """Micro accuracy, macro F1 and a confusion matrix for the last digit of the year."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length")
    if not targets:
        raise ValueError("no samples to report")

    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, predicted in zip(targets, predictions, strict=True):
        matrix[int(true_label)][int(predicted)] += 1

    total = len(targets)
    support = [sum(row) for row in matrix]
    predicted_counts = [
        sum(matrix[row][column] for row in range(num_classes)) for column in range(num_classes)
    ]

    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in range(num_classes):
        if support[label] == 0:
            continue
        true_positive = matrix[label][label]
        precision = true_positive / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = true_positive / support[label]
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator else 0.0
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
            "predicted": predicted_counts[label],
        }
        f1_scores.append(f1)

    hits = sum(matrix[label][label] for label in range(num_classes))
    return {
        "num_samples": total,
        "accuracy": hits / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "majority_baseline": max(support) / total,
        "labels": [int(label) for label in per_class],
        "per_class": per_class,
        "confusion": matrix,
    }


def evaluate_last_digit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    max_batches: int | None = None,
) -> dict:
    """Run the model over a loader and report the last-digit classification quality."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for batch_index, (images, batch_targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=-1)[:, -1].tolist())
            targets.extend(batch_targets[:, -1].tolist())
    return last_digit_report(targets, predictions, num_classes)


def format_last_digit_report(report: dict) -> str:
    labels = report["labels"]
    lines = [
        f"val last digit: acc={report['accuracy']:.3f} macro_f1={report['macro_f1']:.3f} "
        f"(majority baseline {report['majority_baseline']:.3f}, n={report['num_samples']})",
        "  true\\pred " + "".join(f"{label:>6}" for label in labels),
    ]
    for label in labels:
        row = report["confusion"][label]
        lines.append(f"{label:>11} " + "".join(f"{row[column]:>6}" for column in labels))
    return "\n".join(lines)


def _seed_worker(worker_id: int) -> None:
    """Seed a loader worker from its torch seed so augmentation does not depend on spawn order."""
    import numpy as np

    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def fit(config: Config, device_spec: str | None = None) -> dict:
    """Train the digit-head model and write best.pt/last.pt into config.train.out_dir."""
    set_seed(config.project.seed)
    set_deterministic(config.train.deterministic)
    device = resolve_device(device_spec or config.infer.device)
    num_digits = config.model.num_digits

    train_dataset: Dataset
    val_dataset: Dataset
    in_channels = 3
    if config.data.stack_views:
        cache = ImageCache(None)
        started = time.perf_counter()
        train_dataset = StackedViewsDataset(
            config.data.train_manifest, cache, config.data.image_size, num_digits
        )
        val_dataset = StackedViewsDataset(
            config.data.val_manifest, cache, config.data.image_size, num_digits
        )
        in_channels = 3 * len(STACK_VIEWS)
        print(
            f"\nimage cache (native): {len(cache)} images in {time.perf_counter() - started:.1f}s "
            f"({cache.memory_bytes() / 1024**3:.1f} GiB RAM), {len(STACK_VIEWS)} stacked views",
            flush=True,
        )
    elif config.data.cache:
        cache = ImageCache(config.data.image_size)
        started = time.perf_counter()
        train_dataset = CachedCoinDataset(
            config.data.train_manifest, cache, num_digits, augment=config.data.augment
        )
        val_dataset = CachedCoinDataset(config.data.val_manifest, cache, num_digits)
        print(
            f"\nimage cache: {len(cache)} images in {time.perf_counter() - started:.1f}s "
            f"({cache.memory_bytes() / 1024**2:.0f} MiB RAM)",
            flush=True,
        )
    else:
        train_transform = build_transform(config.data.image_size, augment=config.data.augment)
        val_transform = build_transform(config.data.image_size)
        train_dataset = CoinDataset(
            config.data.train_manifest, config.data.image_size, num_digits, train_transform
        )
        val_dataset = CoinDataset(
            config.data.val_manifest, config.data.image_size, num_digits, val_transform
        )

    # With the pixel cache the per-sample work is a lookup plus a normalise (~1 ms), and spawning
    # workers would hand every one of them its own copy of the 500 MiB cache, so the loader runs
    # in-process instead.
    loader_kwargs = {
        "batch_size": config.train.batch_size,
        "num_workers": 0 if config.data.cache else config.train.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if loader_kwargs["num_workers"] > 0:
        # On Windows every worker costs several seconds of spawning, so keep them alive
        # across epochs instead of paying that per epoch (measured: 10 -> 56 img/s).
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["worker_init_fn"] = _seed_worker
    # A dedicated generator keeps the shuffle order independent of every other RNG consumer, so a
    # fixed seed really means the same batches (and, with deterministic=True, the same numbers).
    shuffle_generator = torch.Generator().manual_seed(config.project.seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=shuffle_generator, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(config.model, in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_config": asdict(config.model),
        "image_size": list(config.data.image_size),
        "project": asdict(config.project),
        "in_channels": in_channels,
    }

    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.train.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, device, multihead_loss)
        val_metrics = evaluate(model, val_loader, device, multihead_loss)
        epoch_seconds = time.perf_counter() - epoch_started
        print(
            f"epoch {epoch}/{config.train.epochs} ({epoch_seconds:.1f}s) | "
            f"{train_metrics.format('train')} | {val_metrics.format('val')}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch,
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
            }
        )
        checkpoint = {
            **meta,
            "epoch": epoch,
            "model": model.state_dict(),
            "val": asdict(val_metrics),
        }
        if val_metrics.year_accuracy > best_accuracy:
            best_accuracy = val_metrics.year_accuracy
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            torch.save(checkpoint, out_dir / "best.pt")
        torch.save(checkpoint, out_dir / "last.pt")

    # The reported quality is measured on the best weights, not on the last epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
    report = evaluate_last_digit(
        model, val_loader, device, num_classes=config.model.num_classes_per_digit
    )
    print(f"\nbest epoch {best_epoch} (val year accuracy {best_accuracy:.3f})")
    print(format_last_digit_report(report))

    metrics = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_year_accuracy": best_accuracy,
        "epochs_run": len(history),
        "history": history,
        "val_last_digit": report,
        "config": {
            "project": asdict(config.project),
            "data": asdict(config.data),
            "model": asdict(config.model),
            "train": asdict(config.train),
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics -> {metrics_path}")

    from . import plots

    plot_paths = plots.plot_metrics(metrics, out_dir)
    for name, path in plot_paths.items():
        print(f"{name} plot -> {path}")

    return {
        "device": str(device),
        "out_dir": str(out_dir),
        "best_year_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "val_last_digit": report,
        "metrics_path": str(metrics_path),
        "plots": {name: str(path) for name, path in plot_paths.items()},
        "history": history,
    }
