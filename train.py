import argparse
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from pi_ai.config import load_config
from pi_ai.config.hardware import configure_cpu
from pi_ai.data.dataset import FolderImageDataset
from pi_ai.data.loaders import build_loader
from pi_ai.data.transforms import image_transform
from pi_ai.models import build_model
from pi_ai.training.trainer import Trainer
from pi_ai.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact Pi classifier")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, {"epochs": args.epochs} if args.epochs else None)
    configure_cpu(config.threads)
    seed_everything(config.seed)
    data_path = Path(config.data_dir)
    if data_path.is_dir() and any(data_path.iterdir()):
        dataset = FolderImageDataset(data_path, image_transform(config.image_size, config.input_channels))
        if len(dataset.classes) != config.num_classes:
            raise ValueError(f"config expects {config.num_classes} classes, dataset contains {len(dataset.classes)}")
        loader = build_loader(dataset, config.batch_size, shuffle=True, workers=config.num_workers)
    else:
        inputs = torch.randn(128, config.input_channels, config.image_size, config.image_size)
        targets = torch.randint(config.num_classes, (128,))
        loader = DataLoader(TensorDataset(inputs, targets), config.batch_size, shuffle=True, num_workers=0)
    model = build_model(config.model, config.num_classes, config.input_channels)
    history = Trainer(model, config.device, config.learning_rate).fit(loader, epochs=config.epochs, checkpoint=config.checkpoint_dir / "last.pt")
    print(history[-1])


if __name__ == "__main__":
    main()
