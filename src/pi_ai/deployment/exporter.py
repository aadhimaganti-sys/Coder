from pathlib import Path
import torch


def export_torchscript(model, example_input: torch.Tensor, path: str | Path) -> None:
    model.eval()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, example_input).save(str(target))
