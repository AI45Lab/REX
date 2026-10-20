"""Config loading shared by the REX and RMU trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def effective_global_batch_size_for(
    *,
    micro_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    return micro_batch_size * world_size * gradient_accumulation_steps
