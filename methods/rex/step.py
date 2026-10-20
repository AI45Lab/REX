from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from methods.rex.masking import RoutingStatsCollector


def run_step_forward(*, model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> Any:
    model_batch = {key: value for key, value in batch.items() if key != "labels"}
    return model(**model_batch, use_cache=False, output_hidden_states=False)


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    *,
    chunk_size: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    if labels is None:
        raise ValueError("labels are required to compute chunked causal LM loss.")
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            f"Expected logits [batch, seq, vocab] and labels [batch, seq], got {logits.shape=} {labels.shape=}."
        )

    chunk_size = max(1, int(chunk_size))
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(device=logits.device)
    valid_tokens = shift_labels.ne(ignore_index).sum()

    vocab_size = shift_logits.shape[-1]
    seq_len = shift_logits.shape[1]
    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_labels = shift_labels[:, start:end].reshape(-1)
        chunk_logits = shift_logits[:, start:end, :].reshape(-1, vocab_size)
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.float(),
            chunk_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
    return total_loss / valid_tokens.to(dtype=total_loss.dtype).clamp_min(1)


def routing_loss(
    harmful_mass: torch.Tensor,
    split_name: str,
    *,
    harmful_mass_floor: float,
    std_mass_ceiling: float,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "softplus":
        penalty = F.softplus
    elif loss_type == "relu":
        penalty = F.relu
    else:
        raise ValueError(f"Unsupported routing_loss_type: {loss_type}")

    if split_name == "harmful":
        return penalty(harmful_mass.new_tensor(harmful_mass_floor) - harmful_mass)
    if split_name == "std":
        return penalty(harmful_mass - harmful_mass.new_tensor(std_mass_ceiling))
    return harmful_mass.new_zeros(())


def _split_routing_weight(
    step_kind: str,
    *,
    routing_loss_weight_std: float,
    routing_loss_weight_harmful: float,
) -> float:
    return (
        routing_loss_weight_std if step_kind == "std" else routing_loss_weight_harmful
    )


def _routing_loss_terms(
    *,
    step_kind: str,
    routing_collector: RoutingStatsCollector,
    anchor: torch.Tensor,
    routing_harmful_mass_floor: float,
    routing_std_mass_ceiling: float,
    routing_loss_type: str,
    routing_loss_weight_std: float,
    routing_loss_weight_harmful: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    harmful_mass = routing_collector.harmful_mass_for_loss(anchor)
    aux_routing_loss = routing_loss(
        harmful_mass,
        step_kind,
        harmful_mass_floor=routing_harmful_mass_floor,
        std_mass_ceiling=routing_std_mass_ceiling,
        loss_type=routing_loss_type,
    )
    split_routing_weight = _split_routing_weight(
        step_kind,
        routing_loss_weight_std=routing_loss_weight_std,
        routing_loss_weight_harmful=routing_loss_weight_harmful,
    )
    return aux_routing_loss, aux_routing_loss * split_routing_weight


@dataclass
class StepLossResult:
    loss: torch.Tensor
    lm_loss: torch.Tensor
    aux_routing_loss: torch.Tensor
    weighted_routing_loss: torch.Tensor
    rank_loss: torch.Tensor
    weighted_rank_loss: torch.Tensor
    rank_gap: torch.Tensor


def _std_rank_loss_terms(
    *,
    step_kind: str,
    routing_collector: RoutingStatsCollector,
    anchor: torch.Tensor,
    routing_std_rank_loss_weight: float,
    routing_std_rank_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = anchor.new_zeros(())
    if step_kind != "std" or routing_std_rank_loss_weight == 0.0:
        return zero, zero, zero
    rank_loss_value, rank_gap = routing_collector.std_rank_loss(
        anchor, margin=routing_std_rank_margin
    )
    return rank_loss_value, rank_loss_value * routing_std_rank_loss_weight, rank_gap


def compute_step_loss(
    *,
    step_kind: str,
    batch: dict[str, torch.Tensor],
    outputs: Any,
    routing_collector: RoutingStatsCollector,
    loss_chunk_size: int,
    routing_harmful_mass_floor: float,
    routing_std_mass_ceiling: float,
    routing_loss_type: str,
    routing_loss_weight_std: float,
    routing_loss_weight_harmful: float,
    routing_std_rank_loss_weight: float,
    routing_std_rank_margin: float,
) -> StepLossResult:
    if step_kind not in {"std", "harmful"}:
        raise ValueError(f"Unknown REX step_kind: {step_kind}")
    lm_loss = chunked_causal_lm_loss(
        outputs.logits, batch.get("labels"), chunk_size=loss_chunk_size
    )
    aux_routing_loss, weighted_routing_loss = _routing_loss_terms(
        step_kind=step_kind,
        routing_collector=routing_collector,
        anchor=lm_loss,
        routing_harmful_mass_floor=routing_harmful_mass_floor,
        routing_std_mass_ceiling=routing_std_mass_ceiling,
        routing_loss_type=routing_loss_type,
        routing_loss_weight_std=routing_loss_weight_std,
        routing_loss_weight_harmful=routing_loss_weight_harmful,
    )
    rank_loss_value, weighted_rank_loss, rank_gap = _std_rank_loss_terms(
        step_kind=step_kind,
        routing_collector=routing_collector,
        anchor=lm_loss,
        routing_std_rank_loss_weight=routing_std_rank_loss_weight,
        routing_std_rank_margin=routing_std_rank_margin,
    )
    loss = lm_loss + weighted_routing_loss + weighted_rank_loss
    return StepLossResult(
        loss=loss,
        lm_loss=lm_loss,
        aux_routing_loss=aux_routing_loss,
        weighted_routing_loss=weighted_routing_loss,
        rank_loss=rank_loss_value,
        weighted_rank_loss=weighted_rank_loss,
        rank_gap=rank_gap,
    )


def quota_step_kind(
    split_token_budgets: dict[str, int],
    split_tokens_seen: dict[str, int],
) -> str:
    if set(split_token_budgets) != {"std", "harmful"}:
        raise ValueError("REX quotas must define only std and harmful.")
    if set(split_token_budgets) != set(split_tokens_seen):
        raise ValueError(
            "split_token_budgets and split_tokens_seen must contain the same split names."
        )

    selected_name: str | None = None
    selected_remaining = 0
    selected_budget = 1
    for split_name, budget in split_token_budgets.items():
        seen = split_tokens_seen[split_name]
        if budget < 0 or seen < 0:
            raise ValueError("Split token budgets and counters must be non-negative.")
        remaining = budget - seen
        if remaining <= 0:
            continue
        if (
            selected_name is None
            or remaining * selected_budget > selected_remaining * budget
        ):
            selected_name = split_name
            selected_remaining = remaining
            selected_budget = budget

    if selected_name is None:
        raise RuntimeError("All split token quotas are already complete.")
    return selected_name
