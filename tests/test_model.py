import pytest
import torch

from coinvision.config import ModelConfig
from coinvision.model import build_model


def test_build_model_output_shape():
    model = build_model(ModelConfig())
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 32))

    assert logits.shape == (2, 4, 10)


def test_head_count_matches_num_digits():
    model = build_model(ModelConfig(num_digits=3))

    assert len(model.heads) == 3


def test_unsupported_backbone_is_rejected():
    with pytest.raises(ValueError, match="unsupported backbone"):
        build_model(ModelConfig(backbone="vgg16"))
