from typing import Any

import torch


def move_to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if hasattr(value, "to") else value
