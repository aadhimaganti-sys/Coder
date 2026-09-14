import os
import platform


def system_info() -> dict[str, str | int]:
    return {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count() or 1}
