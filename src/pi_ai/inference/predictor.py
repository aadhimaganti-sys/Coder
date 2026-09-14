from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Prediction:
    index: int
    confidence: float
    label: str


class Predictor:
    def __init__(self, model: nn.Module, labels: list[str] | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.labels = labels or []

    def predict(self, inputs: torch.Tensor, top_k: int = 1) -> list[list[Prediction]]:
        with torch.inference_mode():
            probabilities = self.model(inputs.to(self.device)).softmax(1)
            values, indices = probabilities.topk(min(top_k, probabilities.shape[1]), dim=1)
        results = []
        for row_values, row_indices in zip(values.cpu(), indices.cpu()):
            results.append([Prediction(int(index), float(value), self.labels[int(index)] if int(index) < len(self.labels) else str(int(index))) for value, index in zip(row_values, row_indices)])
        return results
