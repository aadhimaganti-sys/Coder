from time import perf_counter


def measure(callable_object, repetitions: int = 10) -> dict[str, float]:
    start = perf_counter()
    for _ in range(repetitions):
        callable_object()
    average = (perf_counter() - start) / repetitions
    return {"average_ms": average * 1000}
