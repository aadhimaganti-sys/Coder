import torch
from pi_ai.deployment.exporter import export_torchscript


def test_torchscript_export(tmp_path, model):
    path = tmp_path / "model.ts"
    export_torchscript(model, torch.randn(1, 1, 28, 28), path)
    assert path.exists()
