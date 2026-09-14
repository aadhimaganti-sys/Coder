import argparse
import torch
from pi_ai.config import load_config
from pi_ai.models import build_model
from pi_ai.training.checkpoint import load_checkpoint, save_checkpoint
from pi_ai.training.finetuner import freeze_features, trainable_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune a saved classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/checkpoints/finetuned.pt")
    parser.add_argument("--unfreeze", action="store_true")
    args = parser.parse_args()
    config = load_config("config.yaml")
    model = build_model(config.model, config.num_classes, config.input_channels)
    payload = load_checkpoint(args.checkpoint, model)
    freeze_features(model, not args.unfreeze)
    save_checkpoint(args.output, model, epoch=payload.get("epoch", 0), metrics={"trainable": trainable_parameters(model)})
    print(f"saved {args.output}; trainable parameters={trainable_parameters(model)}")


if __name__ == "__main__":
    main()
