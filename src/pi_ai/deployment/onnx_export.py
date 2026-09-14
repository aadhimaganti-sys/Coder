from pathlib import Path
import torch


def export_onnx(model, example_input: torch.Tensor, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example_input, str(target), input_names=["input"], output_names=["logits"], opset_version=17)
