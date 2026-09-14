from pi_ai.data.transforms import image_transform


def preprocess(size: int, channels: int):
    return image_transform(size, channels)
