from time import perf_counter
import torch


def latency(model, inputs: torch.Tensor, repetitions: int = 10) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        start = perf_counter()
        for _ in range(repetitions):
            model(inputs)
    average = (perf_counter() - start) / repetitions
    return {"average_ms": average * 1000, "throughput": len(inputs) / average}
