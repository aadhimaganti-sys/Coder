import argparse
import torch
from pi_ai.config import load_config
from pi_ai.models import build_model
from pi_ai.inference.profiling import latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU model latency")
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    config = load_config("config.yaml")
    model = build_model(config.model, config.num_classes, config.input_channels).eval()
    inputs = torch.randn(1, config.input_channels, config.image_size, config.image_size)
    print(latency(model, inputs, args.repetitions))


if __name__ == "__main__":
    main()
