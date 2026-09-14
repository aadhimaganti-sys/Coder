from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timer() :
    start = perf_counter()
    result = {"elapsed": 0.0}
    try:
        yield result
    finally:
        result["elapsed"] = perf_counter() - start
