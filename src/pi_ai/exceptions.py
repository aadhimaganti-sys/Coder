class PiAIError(Exception):
    """Base error for expected project failures."""


class ConfigurationError(PiAIError, ValueError):
    """Raised when configuration values are invalid."""


class DatasetError(PiAIError, ValueError):
    """Raised when dataset files are invalid."""
