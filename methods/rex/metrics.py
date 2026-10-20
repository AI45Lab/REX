from __future__ import annotations

from typing import Any

import torch

LOSS_METRIC_FIELDS = {
    "lm_loss": "lm_loss",
    "routing_loss": "aux_routing_loss",
    "weighted_routing_loss": "weighted_routing_loss",
    "rank_loss": "rank_loss",
    "weighted_rank_loss": "weighted_rank_loss",
    "rank_gap": "rank_gap",
    "total_loss": "loss",
}

SPLIT_LABELS = {
    "std": "D_std",
    "harmful": "D_harmful",
}

STD_RANK_METRICS = {
    "loss_rank_D_std": "rank_loss",
    "loss_rank_weighted_D_std": "weighted_rank_loss",
    "routing_rank_gap_D_std": "rank_gap",
}


def rex_split_label(split_name: str) -> str:
    return SPLIT_LABELS.get(split_name, split_name)


def reduce_loss_accumulator(
    accelerator: Any,
    accumulator: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for metric_name, values in accumulator.items():
        if not values:
            reduced[metric_name] = 0.0
            continue
        value = (
            torch.stack(values)
            .mean()
            .detach()
            .to(device=accelerator.device, dtype=torch.float32)
        )
        reduced[metric_name] = (
            accelerator.reduce(value, reduction="sum") / accelerator.num_processes
        ).item()
    return reduced


def sum_layer_routing_stats(
    accumulator: list[dict[str, torch.Tensor]] | None,
    values: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    if accumulator is None:
        return [
            {key: value.detach().float().clone() for key, value in layer.items()}
            for layer in values
        ]
    if len(accumulator) != len(values):
        raise ValueError(
            f"Mismatched routing layer counts: {len(accumulator)} != {len(values)}"
        )
    for target, layer in zip(accumulator, values):
        for key, value in layer.items():
            target[key] += value.detach().to(
                device=target[key].device, dtype=torch.float32
            )
    return accumulator


def reduce_layer_routing_stats(
    accelerator: Any,
    values: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            key: accelerator.reduce(
                value.detach().to(device=accelerator.device, dtype=torch.float32),
                reduction="sum",
            )
            for key, value in layer.items()
        }
        for layer in values
    ]


def finalize_layer_routing_stats(
    layer_stats: list[dict[str, torch.Tensor]],
) -> dict[str, Any]:
    per_layer_mass: list[float] = []
    per_layer_frac: list[float] = []
    total_dispatches = 0.0
    total_harmful_dispatches = 0.0
    for layer in layer_stats:
        tokens = float(layer["tokens"].item())
        dispatches = float(layer["dispatches"].item())
        harmful_dispatches = float(layer["harmful_dispatches"].item())
        harmful_mass_sum = float(layer["harmful_mass_sum"].item())
        per_layer_mass.append(harmful_mass_sum / max(tokens, 1.0))
        per_layer_frac.append(harmful_dispatches / max(dispatches, 1.0))
        total_dispatches += dispatches
        total_harmful_dispatches += harmful_dispatches

    return {
        "mass": sum(per_layer_mass) / len(per_layer_mass) if per_layer_mass else 0.0,
        "frac": total_harmful_dispatches / max(total_dispatches, 1.0),
        "dispatches": int(total_dispatches),
        "harmful_dispatches": int(total_harmful_dispatches),
        "per_layer_mass": per_layer_mass,
        "per_layer_frac": per_layer_frac,
    }


def _add_loss_metrics(
    metrics: dict[str, float],
    loss_values: dict[str, float],
    mapping: dict[str, str],
) -> None:
    metrics.update(
        {
            metric_name: loss_values[value_name]
            for metric_name, value_name in mapping.items()
        }
    )


def _add_routing_summary_metrics(
    metrics: dict[str, float],
    routing_summary: dict[str, Any],
    split_label: str,
) -> None:
    metrics[f"train_routing_harmful_mass_aggregate_{split_label}"] = float(
        routing_summary["mass"]
    )
    metrics[f"train_routing_harmful_frac_aggregate_{split_label}"] = float(
        routing_summary["frac"]
    )
    for layer_idx, value in enumerate(routing_summary["per_layer_mass"]):
        metrics[f"train_routing_harmful_mass_layer{layer_idx}_{split_label}"] = float(
            value
        )
    for layer_idx, value in enumerate(routing_summary["per_layer_frac"]):
        metrics[f"train_routing_harmful_frac_layer{layer_idx}_{split_label}"] = float(
            value
        )


def build_train_metrics(
    *,
    step_kind: str,
    current_step: int,
    current_iter: int,
    tokens_seen: int,
    split_token_budgets: dict[str, int],
    split_tokens_seen: dict[str, int],
    split_updates: dict[str, int],
    learning_rate: float,
    loss_values: dict[str, float],
    routing_summary: dict[str, Any],
    step_time: float,
    tokens_per_sec: float,
    train_elapsed_seconds: float,
    gpu_hours: float,
) -> dict[str, float]:
    split_label = rex_split_label(step_kind)
    train_metrics = {
        "loss": loss_values["total_loss"],
        "iter": float(current_iter),
        "step": float(current_step),
        "tokens": float(tokens_seen),
        "total_tokens": float(tokens_seen),
        "learning_rate": learning_rate,
        f"loss_total_{split_label}": loss_values["total_loss"],
        f"loss_lm_{split_label}": loss_values["lm_loss"],
        f"loss_routing_{split_label}": loss_values["routing_loss"],
        f"loss_routing_weighted_{split_label}": loss_values["weighted_routing_loss"],
        "step_time": step_time,
        "tokens_per_sec": tokens_per_sec,
        "tokens_seen_D_std": float(split_tokens_seen.get("std", 0)),
        "tokens_seen_D_harmful": float(split_tokens_seen.get("harmful", 0)),
        "optimizer_updates_D_std": float(split_updates.get("std", 0)),
        "optimizer_updates_D_harmful": float(split_updates.get("harmful", 0)),
        "current_model_tokens_seen": float(tokens_seen),
        "train_elapsed_seconds": train_elapsed_seconds,
        "gpu_hours": gpu_hours,
    }
    train_metrics.update(
        {
            "split_token_overshoot_D_std": float(
                max(
                    split_tokens_seen.get("std", 0) - split_token_budgets.get("std", 0),
                    0,
                )
            ),
            "split_token_overshoot_D_harmful": float(
                max(
                    split_tokens_seen.get("harmful", 0)
                    - split_token_budgets.get("harmful", 0),
                    0,
                )
            ),
        }
    )
    if split_label == "D_std":
        _add_loss_metrics(train_metrics, loss_values, STD_RANK_METRICS)
    _add_routing_summary_metrics(train_metrics, routing_summary, split_label)
    return train_metrics
