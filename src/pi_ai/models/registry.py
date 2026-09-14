MODEL_REGISTRY = {"tiny_cnn", "tiny_mlp"}


def is_supported(name: str) -> bool:
    return name in MODEL_REGISTRY
