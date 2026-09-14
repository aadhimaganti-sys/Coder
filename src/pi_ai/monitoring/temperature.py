from pathlib import Path


def cpu_temperature() -> float | None:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not path.exists():
        return None
    return int(path.read_text().strip()) / 1000
