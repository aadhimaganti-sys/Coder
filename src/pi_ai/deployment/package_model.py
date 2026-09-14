from pathlib import Path
import shutil


def package_model(model_path: str | Path, destination: str | Path) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(model_path, target))
