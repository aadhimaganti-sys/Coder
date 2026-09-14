import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from pi_ai.config import load_config
from pi_ai.models import build_model
from pi_ai.training.checkpoint import load_checkpoint
from pi_ai.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config("config.yaml")
    model = build_model(config.model, config.num_classes, config.input_channels)
    load_checkpoint(args.checkpoint, model)
    data = torch.randn(32, config.input_channels, config.image_size, config.image_size)
    labels = torch.randint(config.num_classes, (32,))
    print(evaluate(model, DataLoader(TensorDataset(data, labels), config.batch_size)))


if __name__ == "__main__":
    main()
