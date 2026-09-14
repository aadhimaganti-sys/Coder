from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


def build_scheduler(optimizer: Optimizer, step_size: int = 5, gamma: float = 0.5) -> StepLR:
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
