from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0, metrics: dict | None = None, config: dict | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics or {}, "config": config or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model, optimizer=None, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
