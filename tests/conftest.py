import pytest
import torch
from pi_ai.models import build_model


@pytest.fixture
def model():
    return build_model("tiny_cnn", 3, 1)


@pytest.fixture
def batch():
    return torch.randn(4, 1, 28, 28)
