import torch


def healthcheck() -> dict[str, str]:
    return {"torch": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu"}
