from PIL import Image
from pi_ai.data.transforms import image_transform


def test_transform_shape():
    assert image_transform(12, 1)(Image.new("L", (4, 4))).shape == (1, 12, 12)
