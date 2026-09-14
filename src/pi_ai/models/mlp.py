from torch import Tensor, nn


class TinyMLP(nn.Module):
    """Fallback model for already-vectorized low-dimensional features."""

    def __init__(self, input_features: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs.flatten(1))
