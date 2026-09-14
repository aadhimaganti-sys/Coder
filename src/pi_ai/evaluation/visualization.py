from pathlib import Path


def write_confusion_csv(matrix, path: str | Path) -> None:
    Path(path).write_text("\n".join(",".join(str(int(item)) for item in row) for row in matrix), encoding="utf-8")
