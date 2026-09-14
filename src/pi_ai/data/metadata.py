from pathlib import Path


def write_labels(classes: list[str], path: str | Path) -> None:
    Path(path).write_text("\n".join(classes) + "\n", encoding="utf-8")


def read_labels(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
