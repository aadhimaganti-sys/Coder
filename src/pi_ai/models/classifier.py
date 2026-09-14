from torch import nn


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace a conventional classifier head during finetuning."""
    if not hasattr(model, "classifier"):
        raise ValueError("model has no classifier attribute")
    old = model.classifier
    if not isinstance(old, nn.Linear):
        raise ValueError("classifier must be a Linear layer")
    model.classifier = nn.Linear(old.in_features, num_classes)
    return model
