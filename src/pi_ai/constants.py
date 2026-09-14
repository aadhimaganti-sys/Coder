from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_SIZE = 28
DEFAULT_THREADS = 2
DEFAULT_BATCH_SIZE = 32
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
