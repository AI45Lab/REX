"""REX training configuration: split budgets, learning-rate banks and variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.config import effective_global_batch_size_for
from training.setup import resolve_output_dir


SPLIT_TOKEN_BUDGET_KEYS = ("std", "harmful")


REX_VARIANTS = ("standard", "gradient_mask_only", "without_std_activation_mask", "without_routing_loss")


def parse_rex_variant(config: dict[str, Any]) -> str:
    """Explicit diagnostic controls; the main recipe retains both mechanisms."""
    variant = config.get("rex_variant", "standard")
    if not isinstance(variant, str) or variant not in REX_VARIANTS:
        raise ValueError(f"rex_variant must be one of {REX_VARIANTS}, got {variant!r}.")
    return variant



@dataclass(frozen=True)
class TrainingSetup:
    run_name: str
    checkpoint_dir: Path
    micro_batch_size: int
    effective_global_batch_size: int
    loss_chunk_size: int
    gradient_accumulation_steps: int
    split_token_budgets: dict[str, int]


def _loss_chunk_size(config: dict[str, Any]) -> int:
    loss_chunk_size = int(config.get("loss_chunk_size", 512))
    if loss_chunk_size <= 0:
        raise ValueError(f"loss_chunk_size must be positive, got {loss_chunk_size}.")
    return loss_chunk_size


def _parse_lr_multiplier(
    config: dict[str, Any], key: str, *, allow_zero: bool = False
) -> float:
    requirement = "finite nonnegative number" if allow_zero else "finite positive number"
    value = config.get(key, 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a {requirement}.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or (parsed == 0.0 and not allow_zero):
        raise ValueError(f"{key} must be a {requirement}.")
    return parsed


def parse_std_router_lr_multiplier(config: dict[str, Any]) -> float:
    return _parse_lr_multiplier(config, "std_router_lr_multiplier")


def parse_std_expert_lr_multiplier(config: dict[str, Any]) -> float:
    return _parse_lr_multiplier(config, "std_expert_lr_multiplier")


def parse_harmful_router_lr_multiplier(config: dict[str, Any]) -> float:
    return _parse_lr_multiplier(config, "harmful_router_lr_multiplier")


def parse_harmful_router_std_step_lr_multiplier(config: dict[str, Any]) -> float:
    return _parse_lr_multiplier(config, "harmful_router_std_step_lr_multiplier")


def parse_harmful_expert_lr_multiplier(config: dict[str, Any]) -> float:
    # Zero disables parameter updates, while retaining backward, clipping and
    # optimizer state semantics for a controlled learning-rate comparison.
    return _parse_lr_multiplier(config, "harmful_expert_lr_multiplier", allow_zero=True)


def parse_gemma_harmful_expert_scale_lr(config: dict[str, Any]) -> float | None:
    """Absent keeps the scales in the router banks; zero freezes both scale banks."""
    key = "gemma_harmful_expert_scale_lr"
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} must be a finite nonnegative number")
    return float(value)


def parse_split_token_budgets(
    config: dict[str, Any],
    *,
    max_tokens: int,
) -> dict[str, int]:
    if "split_token_budgets" not in config:
        raise ValueError("split_token_budgets is required.")
    raw_budgets = config["split_token_budgets"]
    if not isinstance(raw_budgets, dict):
        raise ValueError("split_token_budgets must be a mapping.")

    expected_keys = set(SPLIT_TOKEN_BUDGET_KEYS)
    actual_keys = set(raw_budgets)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unknown = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            "split_token_budgets must define exactly std and harmful"
            + (f": {', '.join(details)}" if details else ".")
        )

    budgets: dict[str, int] = {}
    for split_name in SPLIT_TOKEN_BUDGET_KEYS:
        value = raw_budgets[split_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"split_token_budgets.{split_name} must be a non-negative integer."
            )
        budgets[split_name] = value

    if sum(budgets.values()) != max_tokens:
        raise ValueError(
            "sum(split_token_budgets) must equal max_tokens: "
            f"{sum(budgets.values())} != {max_tokens}."
        )
    return budgets


def prepare_training_config(
    config: dict[str, Any],
    config_path: Path,
    *,
    world_size: int,
) -> TrainingSetup:
    max_tokens = int(config["max_tokens"])
    micro_batch_size = int(config["micro_batch_size"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
    loss_chunk_size = _loss_chunk_size(config)
    parse_std_router_lr_multiplier(config)
    parse_std_expert_lr_multiplier(config)
    parse_harmful_router_lr_multiplier(config)
    parse_harmful_router_std_step_lr_multiplier(config)
    parse_harmful_expert_lr_multiplier(config)
    parse_gemma_harmful_expert_scale_lr(config)
    parse_rex_variant(config)

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}."
        )
    effective_global_batch_size = effective_global_batch_size_for(
        micro_batch_size=micro_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    split_token_budgets = parse_split_token_budgets(config, max_tokens=max_tokens)
    run_name, checkpoint_dir = resolve_output_dir(config)
    return TrainingSetup(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        micro_batch_size=micro_batch_size,
        effective_global_batch_size=effective_global_batch_size,
        loss_chunk_size=loss_chunk_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        split_token_budgets=split_token_budgets,
    )


def estimate_training_steps(config: dict[str, Any], *, effective_global_batch_size: int) -> int:
    configured_steps = config.get("scheduler_total_steps")
    if configured_steps is not None:
        if (
            isinstance(configured_steps, bool)
            or not isinstance(configured_steps, int)
            or configured_steps <= 0
        ):
            raise ValueError("scheduler_total_steps must be a positive integer when set.")
        return configured_steps
    max_tokens = int(config["max_tokens"])
    seq_length = int(config["max_seq_length"])
    return max(1, math.ceil(max_tokens / (effective_global_batch_size * seq_length)))
