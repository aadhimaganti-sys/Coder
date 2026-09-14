from torch import Tensor, nn
from .depthwise_separable import DepthwiseSeparableBlock


class TinyPiCNN(nn.Module):
    """Small CNN with bounded maps and no expensive flattened hidden layer."""

    def __init__(self, num_classes: int, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, 2),
            DepthwiseSeparableBlock(24, 32, 2),
            DepthwiseSeparableBlock(32, 48),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("expected input shape [batch, channels, height, width]")
        return self.classifier(self.pool(self.features(inputs)).flatten(1))
