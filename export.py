import argparse
import torch
from pi_ai.config import load_config
from pi_ai.models import build_model
from pi_ai.training.checkpoint import load_checkpoint
from pi_ai.deployment.exporter import export_torchscript


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TorchScript model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/exports/model.ts")
    args = parser.parse_args()
    config = load_config("config.yaml")
    model = build_model(config.model, config.num_classes, config.input_channels)
    load_checkpoint(args.checkpoint, model)
    example = torch.randn(1, config.input_channels, config.image_size, config.image_size)
    export_torchscript(model, example, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
