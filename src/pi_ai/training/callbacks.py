from pathlib import Path
from .checkpoint import save_checkpoint


class CheckpointCallback:
    def __init__(self, path: str | Path) -> None:
        self.path = path

    def save(self, model, optimizer, epoch, metrics):
        save_checkpoint(self.path, model, optimizer, epoch, metrics)
