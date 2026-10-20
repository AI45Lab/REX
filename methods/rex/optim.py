"""REX optimizer: per-bank router/expert learning rates."""

from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator

from methods.rex.config import (
    parse_gemma_harmful_expert_scale_lr,
    parse_harmful_expert_lr_multiplier,
    parse_harmful_router_lr_multiplier,
    parse_harmful_router_std_step_lr_multiplier,
    parse_std_expert_lr_multiplier,
    parse_std_router_lr_multiplier,
)
from methods.rex.masking import GradientMasker
from moe.model_adapters import parameter_role
from training.runtime import build_adamw, default_parameter_groups, uses_weight_decay


def _optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    base_lr: float,
    std_router_lr_multiplier: float,
    std_expert_lr_multiplier: float,
    harmful_router_lr_multiplier: float,
    harmful_router_std_step_lr_multiplier: float,
    harmful_expert_lr_multiplier: float,
) -> list[dict[str, Any]]:
    scale_lr = getattr(model, "_rex_harmful_expert_scale_lr", None)
    if (
        scale_lr is None
        and std_router_lr_multiplier == 1.0
        and std_expert_lr_multiplier == 1.0
        and harmful_router_lr_multiplier == 1.0
        and harmful_router_std_step_lr_multiplier == 1.0
        and harmful_expert_lr_multiplier == 1.0
    ):
        return default_parameter_groups(model, weight_decay)

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    std_router_decay_params: list[torch.nn.Parameter] = []
    std_router_no_decay_params: list[torch.nn.Parameter] = []
    std_expert_decay_params: list[torch.nn.Parameter] = []
    std_expert_no_decay_params: list[torch.nn.Parameter] = []
    harmful_router_decay_params: list[torch.nn.Parameter] = []
    harmful_router_no_decay_params: list[torch.nn.Parameter] = []
    harmful_expert_decay_params: list[torch.nn.Parameter] = []
    harmful_expert_no_decay_params: list[torch.nn.Parameter] = []
    harmful_scale_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = parameter_role(model, name)
        if scale_lr is not None and role.projection == "per_expert_scale":
            if role.role != "harmful_router" or scale_lr <= 0:
                raise ValueError("Frozen expert scale unexpectedly trainable")
            target = harmful_scale_params
        elif role.role == "std_router":
            target = (
                std_router_decay_params
                if uses_weight_decay(name, parameter)
                else std_router_no_decay_params
            )
        elif parameter_role(model, name).role == "std_expert":
            target = (
                std_expert_decay_params
                if uses_weight_decay(name, parameter)
                else std_expert_no_decay_params
            )
        elif parameter_role(model, name).role == "harmful_router":
            target = (
                harmful_router_decay_params
                if uses_weight_decay(name, parameter)
                else harmful_router_no_decay_params
            )
        elif parameter_role(model, name).role == "harmful_expert":
            target = (
                harmful_expert_decay_params
                if uses_weight_decay(name, parameter)
                else harmful_expert_no_decay_params
            )
        else:
            target = decay_params if uses_weight_decay(name, parameter) else no_decay_params
        target.append(parameter)
    return [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr,
        },
        {
            "params": std_router_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr * std_router_lr_multiplier,
        },
        {
            "params": std_router_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr * std_router_lr_multiplier,
        },
        {
            "params": std_expert_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr * std_expert_lr_multiplier,
        },
        {
            "params": std_expert_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr * std_expert_lr_multiplier,
        },
        {
            "params": harmful_router_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr * harmful_router_lr_multiplier,
            "rex_bank": "harmful_router",
        },
        {
            "params": harmful_router_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr * harmful_router_lr_multiplier,
            "rex_bank": "harmful_router",
        },
        {
            "params": harmful_expert_decay_params,
            "weight_decay": weight_decay,
            "lr": base_lr * harmful_expert_lr_multiplier,
        },
        {
            "params": harmful_expert_no_decay_params,
            "weight_decay": 0.0,
            "lr": base_lr * harmful_expert_lr_multiplier,
        },
    ] + ([{"params": harmful_scale_params, "lr": scale_lr, "weight_decay": 0.0,
           "rex_bank": "harmful_expert_scale"}] if harmful_scale_params else [])


def build_rex_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(config["lr"])
    if parse_gemma_harmful_expert_scale_lr(config) != getattr(model, "_rex_harmful_expert_scale_lr", None):
        raise ValueError("Configure scale trainability before building the optimizer")
    return build_adamw(
        _optimizer_parameter_groups(
            model,
            weight_decay=float(config.get("weight_decay", 0.0)),
            base_lr=base_lr,
            std_router_lr_multiplier=parse_std_router_lr_multiplier(config),
            std_expert_lr_multiplier=parse_std_expert_lr_multiplier(config),
            harmful_router_lr_multiplier=parse_harmful_router_lr_multiplier(config),
            harmful_router_std_step_lr_multiplier=parse_harmful_router_std_step_lr_multiplier(config),
            harmful_expert_lr_multiplier=parse_harmful_expert_lr_multiplier(config),
        ),
        config,
    )


def validate_gradient_masker_matches(
    accelerator: Accelerator,
    gradient_masker: GradientMasker,
    *,
    require_expert_params: bool,
) -> dict[str, int]:
    summary = gradient_masker.parameter_match_summary()
    accelerator.print(
        "[gradient-mask] "
        f"trainable_params={summary['trainable_params']} "
        f"expert_params={summary['expert_params']} "
        f"std_bank_expert_params={summary['std_bank_expert_params']} "
        f"harmful_bank_expert_params={summary['harmful_bank_expert_params']} "
        f"router_params={summary['router_params']}"
    )
    if require_expert_params and summary["expert_params"] == 0:
        raise RuntimeError(
            "GradientMasker did not find any trainable MoE expert parameters after model wrapping. "
            "This would break D_std/D_harmful expert gradient isolation. "
            "If using FSDP, ensure fsdp_use_orig_params is true so expert parameter names are preserved."
        )
    return summary
