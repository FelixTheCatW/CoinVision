import os

import torch

from coinvision.utils import set_deterministic


def test_set_deterministic_toggles_torch_flags():
    original = torch.are_deterministic_algorithms_enabled()
    try:
        set_deterministic(True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

        set_deterministic(False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is False
    finally:
        set_deterministic(original)
