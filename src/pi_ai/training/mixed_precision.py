import torch


def autocast_context(enabled: bool = False):
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda")
    return torch.autocast(device_type="cpu", enabled=False)
