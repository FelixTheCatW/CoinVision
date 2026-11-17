from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .config import ModelConfig
from .dataset import build_transform, digits_to_year
from .model import build_model
from .utils import resolve_device


class CoinPredictor:
    """Loads a checkpoint and predicts the year of a coin image."""

    def __init__(self, model: torch.nn.Module, transform, device: torch.device) -> None:
        self.model = model
        self.transform = transform
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, device_spec: str = "auto"
    ) -> CoinPredictor:
        device = resolve_device(device_spec)
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        model = build_model(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        transform = build_transform(tuple(payload.get("image_size", (448, 224))))
        return cls(model, transform, device)

    def predict(self, image: str | Path | Image.Image) -> dict:
        if isinstance(image, Image.Image):
            tensor = self.transform(image.convert("RGB"))
        else:
            with Image.open(image) as handle:
                tensor = self.transform(handle.convert("RGB"))

        model_device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0).to(model_device))[0]

        probabilities = logits.softmax(dim=-1)
        digits = probabilities.argmax(dim=-1).tolist()
        digit_confidence = probabilities.max(dim=-1).values.tolist()

        return {
            "year": digits_to_year(digits),
            "digits": digits,
            "confidence": float(torch.prod(torch.tensor(digit_confidence))),
            "digit_confidence": digit_confidence,
        }
