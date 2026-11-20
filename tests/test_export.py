from pathlib import Path

import pytest
import torch

from coinvision.export import METADATA_FILENAME, export_onnx

pytest.importorskip("onnxruntime")

from coinvision.config import ModelConfig  # noqa: E402
from coinvision.model import build_model  # noqa: E402


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = build_model(ModelConfig(num_digits=4, num_classes_per_digit=10))
    payload = {
        "model_config": {
            "backbone": "resnet18",
            "pretrained": False,
            "num_digits": 4,
            "num_classes_per_digit": 10,
            "dropout": 0.2,
        },
        "image_size": [64, 32],
        "model": model.state_dict(),
    }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


def test_export_writes_onnx_and_metadata(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    out_dir = tmp_path / "onnx"

    metadata = export_onnx([checkpoint], out_dir, image_size=(64, 32))

    assert (out_dir / METADATA_FILENAME).is_file()
    assert (out_dir / "coinvision_0.onnx").is_file()
    assert metadata["input"]["width"] == 64
    assert metadata["input"]["height"] == 32
    assert metadata["digits"] == 4
    assert metadata["classes"] == 10
    assert metadata["models"][0]["max_abs_diff"] < 1e-3


def test_export_verifies_graph_against_torch(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    out_dir = tmp_path / "onnx"

    metadata = export_onnx([checkpoint], out_dir, image_size=(64, 32), verify=True)
    diff = metadata["models"][0]["max_abs_diff"]

    assert isinstance(diff, float)
    assert diff < 1e-3


def test_export_requires_a_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError):
        export_onnx([], tmp_path / "onnx")
