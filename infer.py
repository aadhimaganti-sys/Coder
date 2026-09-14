import argparse
import torch
from pi_ai.config import load_config
from pi_ai.models import build_model
from pi_ai.training.checkpoint import load_checkpoint
from pi_ai.inference.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    config = load_config("config.yaml")
    model = build_model(config.model, config.num_classes, config.input_channels)
    load_checkpoint(args.model, model)
    inputs = torch.randn(1, config.input_channels, config.image_size, config.image_size)
    print(Predictor(model).predict(inputs, args.top_k))


if __name__ == "__main__":
    main()
