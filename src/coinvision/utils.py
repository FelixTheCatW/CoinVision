from __future__ import annotations

import random
import re
from pathlib import Path

YEAR_RE = re.compile(r"(\d{4})")


def parse_year(path: str | Path) -> str | None:
    stem = Path(path).stem
    matches = YEAR_RE.findall(stem)
    return matches[-1] if matches else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(spec: str = "auto"):
    import torch

    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def human_bytes(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"
