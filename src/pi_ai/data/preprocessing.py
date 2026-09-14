from pathlib import Path


def ensure_dataset_layout(root: str | Path) -> Path:
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path
