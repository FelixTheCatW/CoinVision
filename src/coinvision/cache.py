"""In-memory cache of decoded, resized pixels.

Decoding a 1300x700 WebP costs ~21 ms and resizing another ~6 ms per image, and the training
loop needs the same pixels every epoch. Keeping the resized uint8 pixels (294 KiB per image at
448x224) in RAM removes that work from the loader: measured 30 img/s -> 74 img/s, i.e. the GPU
stops waiting for batches.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

UINT8_BYTES_PER_IMAGE = 3 * 448 * 224


class ImageCache:
    """Decoded uint8 (CHW) pixels, shared by every dataset.

    ``image_size=None`` keeps the native resolution, which is what the crop pipeline needs: the
    date is only a few dozen pixels high in the original and any earlier resize destroys it.
    """

    def __init__(self, image_size: tuple[int, int] | None) -> None:
        if image_size is None:
            self.width = self.height = 0
        else:
            self.width = int(image_size[0])
            self.height = int(image_size[1])
        self._tensors: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._tensors)

    def preload(self, paths: Iterable[str | Path], threads: int = 8) -> int:
        """Decode the missing paths in parallel; returns how many images were added."""
        missing = [
            path
            for path in dict.fromkeys(str(path) for path in paths)
            if path not in self._tensors
        ]
        if not missing:
            return 0

        loaded: dict[str, torch.Tensor] = {}
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            for path, tensor in zip(missing, pool.map(self._read, missing), strict=True):
                loaded[path] = tensor
        with self._lock:
            self._tensors.update(loaded)
        return len(loaded)

    def get(self, path: str | Path) -> torch.Tensor:
        key = str(path)
        tensor = self._tensors.get(key)
        if tensor is None:
            # Not preloaded (e.g. a manifest changed under us): decode on the spot, still correct.
            tensor = self._read(key)
            with self._lock:
                self._tensors[key] = tensor
        return tensor

    def memory_bytes(self) -> int:
        return sum(tensor.numel() for tensor in self._tensors.values())

    def __getstate__(self) -> dict:
        # DataLoader workers are spawned (Windows) and receive a pickled dataset; the lock cannot
        # be pickled, so it is recreated on the other side.
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _read(self, path: str) -> torch.Tensor:
        with Image.open(path) as handle:
            image = v2.functional.to_image(handle.convert("RGB"))
        if self.width == 0:
            return image.contiguous()
        resized = v2.functional.resize(image, [self.height, self.width], antialias=True)
        return resized.contiguous()
