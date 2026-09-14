from torch.utils.data import WeightedRandomSampler


def balanced_sampler(weights, samples: int):
    return WeightedRandomSampler(weights, samples, replacement=True)
