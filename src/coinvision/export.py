"""ONNX export: turns trained checkpoints into a portable graph plus a metadata sidecar."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

from .config import ModelConfig
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model

ONNX_OPSET = 17
METADATA_FILENAME = "model.json"


def _load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model = build_model(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model.to(device)


def export_onnx(
    checkpoints: Sequence[str | Path],
    out_dir: str | Path,
    image_size: tuple[int, int] = (448, 224),
    opset: int = ONNX_OPSET,
    verify: bool = True,
) -> dict:
    """Write one .onnx per checkpoint plus model.json describing the preprocessing.

    The graph takes a float32 NCHW batch and returns raw logits of shape [N, digits, classes],
    so the caller applies softmax (and can average several models, which is what the API does).
    """
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    width, height = int(image_size[0]), int(image_size[1])

    exported: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoints):
        model = _load_model(checkpoint, device)
        name = f"coinvision_{index}.onnx"
        target = out_dir / name
        dummy = torch.zeros(1, 3, height, width, device=device)
        torch.onnx.export(
            model,
            (dummy,),
            str(target),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )

        entry: dict[str, object] = {
            "file": name,
            "source": str(Path(checkpoint).as_posix()),
            "size_bytes": target.stat().st_size,
        }
        if verify:
            entry["max_abs_diff"] = round(_verify(model, target, dummy), 6)
        exported.append(entry)

    metadata = {
        "format": "coinvision-onnx",
        "version": 1,
        "input": {
            "name": "images",
            "layout": "NCHW",
            "dtype": "float32",
            "width": width,
            "height": height,
            "resize": "bilinear antialias, stretch to width x height",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "channel_order": "RGB",
        },
        "output": {"name": "logits", "shape": ["batch", "digits", "classes"]},
        "digits": 4,
        "classes": 10,
        "models": exported,
    }
    (out_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _verify(model: torch.nn.Module, path: Path, dummy: torch.Tensor) -> float:
    """Compare the exported graph against torch on random input; returns the max abs difference."""
    import numpy as np
    import onnxruntime as ort

    generator = torch.Generator().manual_seed(0)
    probe = torch.randn(2, 3, dummy.shape[2], dummy.shape[3], generator=generator)
    with torch.no_grad():
        expected = model(probe).numpy()

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: probe.numpy()})[0]
    return float(np.abs(expected - actual).max())
