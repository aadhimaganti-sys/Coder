from pathlib import Path


def list_files(directory: str | Path, suffixes: set[str] | None = None) -> list[Path]:
    paths = [path for path in Path(directory).rglob("*") if path.is_file()]
    return sorted(path for path in paths if not suffixes or path.suffix.lower() in suffixes)
