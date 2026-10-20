from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from methods.rex.config import parse_gemma_harmful_expert_scale_lr
from moe.model_adapters import (
    GEMMA,
    RoutingOutput,
    model_adapter,
    parameter_role,
    route_logits,
)

ROUTING_STAT_KEYS = ("tokens", "dispatches", "harmful_dispatches", "harmful_mass_sum")


def set_rex_trainable_parameters(
    model: torch.nn.Module, config: dict | None = None
) -> None:
    scale_lr = parse_gemma_harmful_expert_scale_lr(config or {})
    if scale_lr is not None and model_adapter(model) != GEMMA:
        raise ValueError("Independent expert scales require Gemma")
    model._rex_harmful_expert_scale_lr = scale_lr
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        role = parameter_role(model, name)
        if scale_lr is not None and role.projection == "per_expert_scale":
            parameter.requires_grad_(role.role == "harmful_router" and scale_lr > 0)
        elif role.role in {
            "std_expert",
            "harmful_expert",
            "std_router",
            "harmful_router",
        }:
            parameter.requires_grad_(True)


def _iter_moe_blocks(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.nn.Module, torch.nn.Module]]:
    for layer in model_adapter(model).moe_layers(model):
        yield layer.owner, layer.router


def _unpack_router_output(
    output: Any, *, owner: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(output, RoutingOutput):
        return output.logits, output.weights, output.selected
    if (
        isinstance(output, tuple)
        and len(output) == 3
        and torch.is_tensor(output[0])
        and torch.is_tensor(output[1])
        and torch.is_tensor(output[2])
    ):
        return output
    raise RuntimeError(
        f"{owner} expects router output as (router_logits, topk_weights, selected)."
    )


def _tail_contiguous_start(
    num_experts: int, harmful_ids: tuple[int, ...]
) -> int | None:
    """Return the tail-start index if harmful_ids are a tail-contiguous suffix of range(num_experts).

    Returns None when there are no harmful experts to mask for this num_experts. Raises ValueError
    when harmful_ids are present but non-tail-contiguous (a real misconfiguration, not a skip).
    """
    if num_experts <= harmful_ids[-1]:
        return None
    tail_start = harmful_ids[0]
    expected = tuple(range(tail_start, num_experts))
    if harmful_ids != expected:
        raise ValueError(
            f"Requires tail-contiguous harmful experts: num_experts={num_experts}, "
            f"harmful_experts={list(harmful_ids)}."
        )
    return tail_start


def _cached_tail_contiguous_start(
    cache: dict[int, int | None],
    *,
    num_experts: int,
    harmful_ids: tuple[int, ...],
) -> int | None:
    if num_experts not in cache:
        cache[num_experts] = _tail_contiguous_start(num_experts, harmful_ids)
    return cache[num_experts]


def _empty_layer_stats(anchor: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: anchor.new_zeros((), dtype=torch.float32) for key in ROUTING_STAT_KEYS}


def _zero_layer_stats(routing_mass: torch.Tensor) -> dict[str, torch.Tensor]:
    stats = {"routing_mass": routing_mass}
    zero = routing_mass.detach().to(dtype=torch.float32)
    stats.update({key: zero for key in ROUTING_STAT_KEYS})
    return stats


class ActivationMasker:
    def __init__(self, model: torch.nn.Module, harmful_experts: list[int]) -> None:
        self._adapter = model_adapter(model)
        self._enabled = False
        self._harmful_ids = tuple(sorted({int(expert) for expert in harmful_experts}))
        if not self._harmful_ids:
            raise ValueError(
                "ActivationMasker requires at least one harmful expert id."
            )
        self._tail_start_cache: dict[int, int | None] = {}
        self._handles = []

        try:
            self._register_router_hooks(model)
        except Exception:
            self.close()
            raise
        if not self._handles:
            raise ValueError(
                "Could not find MoE modules for activation masking. "
                "Expected modules with both 'gate' and 'experts'."
            )

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def close(self) -> None:
        self.disable()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_router_hooks(self, model: torch.nn.Module) -> None:
        for _, gate in _iter_moe_blocks(model):
            spec = self._adapter.activation_mask_spec(gate)
            num_experts, top_k = spec.num_experts, spec.top_k
            if num_experts <= 0 or top_k <= 0:
                raise ValueError(
                    "ActivationMasker requires a router with positive expert count and top_k."
                )
            tail_start = self._harmful_tail_start(num_experts)
            if tail_start is None:
                continue
            if tail_start < top_k:
                raise ValueError(
                    "ActivationMasker requires at least top_k non-harmful experts: "
                    f"num_experts={num_experts}, harmful_experts={list(self._harmful_ids)}, top_k={top_k}."
                )
            self._handles.append(
                spec.target.register_forward_hook(
                    self._mask_projection_output
                    if spec.projection_output
                    else self._mask_router_output,
                    with_kwargs=False,
                )
            )

    def _mask_projection_output(
        self,
        _projection: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self._enabled:
            return None
        tail_start = self._harmful_tail_start(output.shape[-1])
        if tail_start is None:
            raise RuntimeError("ActivationMasker projection has no tail harmful experts.")
        return output[..., :tail_start]

    def _harmful_tail_start(self, num_experts: int) -> int | None:
        return _cached_tail_contiguous_start(
            self._tail_start_cache,
            num_experts=num_experts,
            harmful_ids=self._harmful_ids,
        )

    def _mask_router_output(
        self, _gate: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> Any | None:
        if not self._enabled:
            return None

        router_logits, original_topk_weights, selected = _unpack_router_output(
            output, owner="ActivationMasker"
        )
        top_k = int(selected.shape[-1])
        tail_start = self._harmful_tail_start(router_logits.shape[-1])
        if tail_start is None:
            raise RuntimeError(
                "ActivationMasker was invoked for a router without tail harmful experts."
            )
        if tail_start < top_k:
            raise RuntimeError(
                f"ActivationMasker requires top_k={top_k} non-harmful experts, got {tail_start}."
            )
        # output[0] is not consumed downstream: the expanded MoE forward discards it, routing_weights is
        # recomputed from the original router_logits[..., :tail_start] below, and RoutingStatsCollector
        # is registered before this masker so it already read the original logits. Return the original
        # router_logits instead of cloning+masking a tensor that nothing reads.
        gemma = bool(self._adapter.router_aux_keys)
        scale = _gate.per_expert_scale if gemma else None
        routed = route_logits(
            router_logits, top_k, gemma=gemma, scale=scale, allowed=tail_start
        )
        return routed if isinstance(output, RoutingOutput) else tuple(routed)


class RoutingStatsCollector:
    def __init__(self, model: torch.nn.Module, harmful_experts: list[int]) -> None:
        self._harmful_ids = tuple(sorted({int(expert) for expert in harmful_experts}))
        if not self._harmful_ids:
            raise ValueError(
                "RoutingStatsCollector requires at least one harmful expert id."
            )
        self._harmful_max = self._harmful_ids[-1]
        self._tail_start_cache: dict[int, int | None] = {}
        self._attention_mask: torch.Tensor | None = None
        self._handles = []
        self._layer_stats: list[dict[str, torch.Tensor] | None] = []
        self._layer_logits: list[torch.Tensor | None] = []
        self._layer_top_k: list[int | None] = []
        self._collect_stats = True
        self._suspended = False

        for module, gate in _iter_moe_blocks(model):
            layer_idx = len(self._layer_stats)
            self._layer_stats.append(None)
            self._layer_logits.append(None)
            self._layer_top_k.append(None)
            self._handles.append(
                gate.register_forward_hook(
                    self._make_hook(layer_idx, module), with_kwargs=False
                )
            )
        if not self._handles:
            raise ValueError(
                "Could not find Qwen3-MoE modules with both 'gate' and 'experts'."
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self._layer_stats = [None for _ in self._layer_stats]
        self._layer_logits = [None for _ in self._layer_logits]
        self._layer_top_k = [None for _ in self._layer_top_k]

    @contextlib.contextmanager
    def suspend(self):
        previous_suspended = self._suspended
        previous_collect_stats = self._collect_stats
        previous_attention_mask = self._attention_mask
        self.reset()
        self._suspended = True
        self._attention_mask = None
        try:
            yield
        finally:
            self.reset()
            self._suspended = previous_suspended
            self._collect_stats = previous_collect_stats
            self._attention_mask = previous_attention_mask

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self._attention_mask = (
            attention_mask.detach().to(dtype=torch.bool)
            if attention_mask is not None
            else None
        )

    def set_collect_stats(self, enabled: bool) -> None:
        self._collect_stats = enabled

    def _harmful_tail_start(self, num_experts: int) -> int | None:
        return _cached_tail_contiguous_start(
            self._tail_start_cache,
            num_experts=num_experts,
            harmful_ids=self._harmful_ids,
        )

    def harmful_mass_for_loss(self, anchor: torch.Tensor) -> torch.Tensor:
        masses = [
            stats["routing_mass"] for stats in self._layer_stats if stats is not None
        ]
        if not masses:
            return anchor.new_zeros(())
        return torch.stack(masses).mean()

    def layer_stats(self, anchor: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        empty = _empty_layer_stats(anchor)
        return [
            {
                key: value.detach().to(device=anchor.device, dtype=torch.float32)
                for key, value in stats.items()
                if key != "routing_mass"
            }
            if stats is not None
            else {key: value.clone() for key, value in empty.items()}
            for stats in self._layer_stats
        ]

    def std_rank_loss(
        self, anchor: torch.Tensor, *, margin: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses: list[torch.Tensor] = []
        gaps: list[torch.Tensor] = []
        for router_logits, top_k in zip(self._layer_logits, self._layer_top_k):
            if router_logits is None or top_k is None:
                continue
            if router_logits.shape[-1] <= self._harmful_max:
                continue

            logits = self._filter_valid_tokens(router_logits)
            if logits.numel() == 0:
                continue
            logits = logits.reshape(-1, logits.shape[-1])

            tail_start = self._harmful_tail_start(logits.shape[-1])
            if (
                tail_start is None
                or tail_start < top_k
                or tail_start >= logits.shape[-1]
            ):
                continue
            harmful_logits = logits[:, tail_start:]
            nonharmful_logits = logits[:, :tail_start]
            harmful_max = harmful_logits.max(dim=-1).values
            nonharmful_kth = torch.topk(nonharmful_logits, k=top_k, dim=-1).values[
                :, -1
            ]
            gap = nonharmful_kth - harmful_max
            losses.append(F.relu(logits.new_tensor(margin) - gap).mean())
            gaps.append(gap.mean())

        if not losses:
            zero = anchor.new_zeros(())
            return zero, zero
        return torch.stack(losses).mean(), torch.stack(gaps).mean()

    def _filter_valid_tokens(self, values: torch.Tensor) -> torch.Tensor:
        if self._attention_mask is None:
            return values

        mask = self._attention_mask.to(device=values.device)
        if values.ndim == 2 and values.shape[0] == mask.numel():
            return values[mask.reshape(-1)]
        if values.ndim >= 3 and values.shape[:2] == mask.shape:
            return values[mask]
        return values

    def _filter_selected_tokens(
        self,
        selected: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._attention_mask is None:
            return selected, topk_weights

        mask = self._attention_mask.to(device=selected.device)
        if selected.ndim == 2 and selected.shape[0] == mask.numel():
            valid = mask.reshape(-1)
            return selected[valid], topk_weights[valid]
        if selected.ndim == 3 and selected.shape[:2] == mask.shape:
            return selected[mask], topk_weights[mask]
        return selected, topk_weights

    def _make_hook(self, layer_idx: int, moe_block: torch.nn.Module):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            if self._suspended:
                return
            router_logits, topk_weights, selected = _unpack_router_output(
                output, owner="RoutingStatsCollector"
            )
            if router_logits.ndim < 2:
                return

            gate = _module
            num_experts = int(gate.num_experts)
            if num_experts <= self._harmful_max:
                return
            top_k = int(selected.shape[-1])
            self._layer_logits[layer_idx] = router_logits
            self._layer_top_k[layer_idx] = top_k
            if not self._collect_stats:
                return

            selected, topk_weights = self._filter_selected_tokens(
                selected, topk_weights
            )

            zero = router_logits.sum() * 0.0
            if selected.numel() == 0:
                self._layer_stats[layer_idx] = _zero_layer_stats(zero)
                return

            tail_start = self._harmful_tail_start(num_experts)
            if tail_start is None:
                return
            harmful_mask = selected >= tail_start
            harmful_token_mass = (
                topk_weights * harmful_mask.to(dtype=topk_weights.dtype)
            ).sum(dim=-1)
            routing_mass = harmful_token_mass.mean()
            self._layer_stats[layer_idx] = {
                "routing_mass": routing_mass,
                "tokens": selected.new_tensor(selected.shape[0], dtype=torch.float32),
                "dispatches": selected.new_tensor(
                    selected.numel(), dtype=torch.float32
                ),
                "harmful_dispatches": harmful_mask.sum().to(dtype=torch.float32),
                "harmful_mass_sum": harmful_token_mass.detach()
                .sum()
                .to(dtype=torch.float32),
            }

        return hook


class GradientMasker:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def parameter_match_summary(self) -> dict[str, int]:
        summary = {
            "trainable_params": 0,
            "expert_params": 0,
            "std_bank_expert_params": 0,
            "harmful_bank_expert_params": 0,
            "router_params": 0,
        }
        for name, parameter in self._model.named_parameters():
            if not parameter.requires_grad:
                continue
            summary["trainable_params"] += 1

            if parameter_role(self._model, name).role == "std_expert":
                summary["expert_params"] += 1
                summary["std_bank_expert_params"] += 1
                continue

            if parameter_role(self._model, name).role == "harmful_expert":
                summary["expert_params"] += 1
                summary["harmful_bank_expert_params"] += 1
                continue

            if parameter_role(self._model, name).role in {
                "std_router",
                "harmful_router",
            }:
                summary["router_params"] += 1
        return summary

    def mask_non_target_parameters(self, step_kind: str) -> None:
        if step_kind == "std":
            masked_roles = {"harmful_expert"}
        elif step_kind == "harmful":
            masked_roles = {"std_expert"}
        else:
            raise ValueError(f"Unknown step_kind: {step_kind}")

        for name, parameter in self._model.named_parameters():
            if parameter.grad is None:
                continue
            role = parameter_role(self._model, name)
            should_mask = role.role in masked_roles
            if (
                getattr(self._model, "_rex_harmful_expert_scale_lr", None) is not None
                and role.projection == "per_expert_scale"
                and step_kind != "harmful"
            ):
                should_mask = True
            if should_mask:
                parameter.grad = None
