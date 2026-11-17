from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CoinDataset, build_transform
from .model import build_model
from .utils import resolve_device, set_seed


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


def fit(config: Config, device_spec: str | None = None) -> dict:
    """Train the digit-head model and write best.pt/last.pt into config.train.out_dir."""
    set_seed(config.project.seed)
    device = resolve_device(device_spec or config.infer.device)
    transform = build_transform(config.data.image_size)
    num_digits = config.model.num_digits

    train_dataset = CoinDataset(
        config.data.train_manifest, config.data.image_size, num_digits, transform
    )
    val_dataset = CoinDataset(
        config.data.val_manifest, config.data.image_size, num_digits, transform
    )

    loader_kwargs = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_config": asdict(config.model),
        "image_size": list(config.data.image_size),
        "project": asdict(config.project),
    }

    history: list[dict] = []
    best_accuracy = -1.0
    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, multihead_loss)
        val_metrics = evaluate(model, val_loader, device, multihead_loss)
        print(
            f"epoch {epoch}/{config.train.epochs} | "
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
            torch.save(checkpoint, out_dir / "best.pt")
        torch.save(checkpoint, out_dir / "last.pt")

    return {
        "device": str(device),
        "out_dir": str(out_dir),
        "best_year_accuracy": best_accuracy,
        "history": history,
    }
