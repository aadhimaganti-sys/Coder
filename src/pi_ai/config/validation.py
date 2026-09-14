from .schema import AppConfig


def validate_config(config: AppConfig) -> AppConfig:
    return config.validate()
