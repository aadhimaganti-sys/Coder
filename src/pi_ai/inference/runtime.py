import torch


def inference(model, inputs: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model = model.to(device).eval()
    with torch.inference_mode():
        return model(inputs.to(device))
