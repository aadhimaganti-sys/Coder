from torch.utils.data._utils.collate import default_collate


def collate_batch(batch):
    return default_collate(batch)
