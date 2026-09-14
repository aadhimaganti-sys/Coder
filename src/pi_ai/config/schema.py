from dataclasses import dataclass, field
from pathlib import Path

from pi_ai.exceptions import ConfigurationError


@dataclass
class AppConfig:
    model: str = "tiny_cnn"
    num_classes: int = 2
    input_channels: int = 1
    image_size: int = 28
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    num_workers: int = 0
    threads: int = 2
    device: str = "cpu"
    seed: int = 7
    data_dir: Path = Path("data/raw")
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    class_names: list[str] = field(default_factory=list)

    def validate(self) -> "AppConfig":
        if self.num_classes < 2 or self.input_channels not in (1, 3):
            raise ConfigurationError("num_classes must be >= 2 and input_channels must be 1 or 3")
        if self.image_size < 8 or self.batch_size < 1 or self.epochs < 1:
            raise ConfigurationError("image_size, batch_size, and epochs must be positive")
        if self.learning_rate <= 0 or self.threads < 1 or self.num_workers < 0:
            raise ConfigurationError("learning_rate and threads must be positive; workers cannot be negative")
        return self
