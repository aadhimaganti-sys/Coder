from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .schema import AppConfig
from .defaults import DEFAULT_CONFIG


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    values = dict(DEFAULT_CONFIG)
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        values.update(loaded)
    values.update(overrides or {})
    allowed = {item.name for item in fields(AppConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    config = AppConfig(**values)
    return config.validate()
