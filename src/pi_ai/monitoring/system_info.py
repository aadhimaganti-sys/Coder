import os
import platform


def system_info() -> dict[str, str | int]:
    """Return lightweight runtime information useful during Pi benchmarks."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count() or 1,
    }
