from pathlib import Path
from urllib.request import urlretrieve


def download_dataset(url: str, destination: str | Path) -> Path:
    """Download a caller-selected archive or file without adding a dependency."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    return target
