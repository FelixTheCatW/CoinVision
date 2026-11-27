"""Export the crop reader to ONNX, in the same shape the existing classifier ships in.

The repo had no export command (onnx/coinvision_*.onnx were produced outside the tree), and this
branch needs one because the reader is now trained on detector crops rather than whole frames.

The graph takes RGB in [0, 1] with no normalisation folded in, exactly like onnx/model.json: the
caller resizes to the stated size with bilinear antialiasing, normalises with the mean and std in the
metadata, and reads logits shaped (batch, digits, classes).

    uv run python scripts/export_reader.py \
        --checkpoint runs/reading/fold0/best.pt runs/reading/fold1/best.pt \
        --onnx-prefix onnx/reading_reader --meta onnx/reading_reader.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coinvision.config import ModelConfig  # noqa: E402
from coinvision.model import build_model  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", required=True, help="one or more best.pt files")
    parser.add_argument("--onnx-prefix", default="onnx/reading_reader",
                        help="files are written as <prefix>_<k>.onnx")
    parser.add_argument("--meta", default="onnx/reading_reader.json")
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> int:
    import onnxruntime as ort

    args = build_parser().parse_args()
    prefix = Path(args.onnx_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    digits = classes = 0
    width = height = 0
    for index, checkpoint in enumerate(args.checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**payload["model_config"])
        model = build_model(model_config)
        model.load_state_dict(payload["model"])
        model.eval()

        height, width = int(payload["image_size"][1]), int(payload["image_size"][0])
        digits = int(model_config.num_digits)
        classes = int(model_config.num_classes_per_digit)

        dummy = torch.rand(2, 3, height, width)
        onnx_path = Path(f"{prefix}_{index}.onnx")
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            reference = model(dummy).numpy()
        served = session.run(None, {"images": dummy.numpy()})[0]
        diff = float(np.abs(reference - served).max())

        entries.append({
            "file": onnx_path.name,
            "source": str(checkpoint).replace("\\", "/"),
            "size_bytes": onnx_path.stat().st_size,
            "max_abs_diff": diff,
        })
        print(f"{onnx_path}  parity max |torch - onnx| = {diff:.2e}", flush=True)

    meta = {
        "format": "coinvision-onnx",
        "version": 1,
        "input": {
            "name": "images",
            "layout": "NCHW",
            "dtype": "float32",
            "width": width,
            "height": height,
            "resize": "bilinear antialias, stretch to width x height",
            "mean": MEAN,
            "std": STD,
            "channel_order": "RGB",
        },
        "output": {
            "name": "logits",
            "shape": ["batch", "digits", "classes"],
        },
        "digits": digits,
        "classes": classes,
        "models": entries,
        "trained_on": {
            "crops": "detector quads that hold the year, plus two jitters and the annotated box",
            "manifests": "data/manifests_reading/folds5",
        },
    }
    Path(args.meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    worst = max(entry["max_abs_diff"] for entry in entries)
    print(f"{len(entries)} models -> {args.meta}  worst parity {worst:.2e}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
