from PIL import Image, ImageEnhance


def mild_augmentation(image: Image.Image) -> Image.Image:
    """Cheap deterministic augmentation suitable for small CPU machines."""
    return ImageEnhance.Brightness(image).enhance(0.95)
