from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_RMU_LOSS_REDUCTIONS = {"token_mean", "sequence_mean"}
_RMU_LOSS_COMPUTE_DTYPES = {"float32", "activation"}


@dataclass
class RmuLossResult:
    forget_loss: torch.Tensor
    forget_activation_norm: torch.Tensor
    target_norm: torch.Tensor
    forget_cosine: torch.Tensor
    forget_tokens: torch.Tensor


@dataclass
class RmuLayerTerms:
    forget_loss: torch.Tensor
    forget_activation_norm: torch.Tensor
    target_norm: torch.Tensor
    forget_cosine: torch.Tensor
    forget_tokens: torch.Tensor


@dataclass
class RmuRetainLossResult:
    retain_loss: torch.Tensor
    retain_activation_norm: torch.Tensor
    frozen_activation_norm: torch.Tensor
    retain_cosine: torch.Tensor
    retain_tokens: torch.Tensor


@dataclass
class RmuRetainLayerTerms:
    retain_loss: torch.Tensor
    retain_activation_norm: torch.Tensor
    frozen_activation_norm: torch.Tensor
    retain_cosine: torch.Tensor
    retain_tokens: torch.Tensor


class RmuControlVectorCache:
    def __init__(self, *, seed: int, steering_coeff: float) -> None:
        self._seed = int(seed)
        self._steering_coeff = float(steering_coeff)
        self._cache: dict[
            tuple[int, int, str, int | None, torch.dtype], torch.Tensor
        ] = {}

    def get(
        self, layer_idx: int, hidden_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cache_key = (int(layer_idx), int(hidden_size), device.type, device.index, dtype)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self._seed + int(layer_idx) * 1_000_003 + int(hidden_size) * 97
        )
        # Generate and normalize in the requested loss dtype. Safe_MoE's fixed
        # RMU target is sampled directly in the captured activation dtype; the
        # default float32 loss path remains bit-for-bit unchanged.
        target = torch.rand(1, 1, int(hidden_size), generator=generator, dtype=dtype)
        target = F.normalize(target, dim=-1) * self._steering_coeff
        target = target.to(device=device)
        self._cache[cache_key] = target
        return target


def normalize_rmu_loss_reduction(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    if normalized not in _RMU_LOSS_REDUCTIONS:
        expected = ", ".join(sorted(_RMU_LOSS_REDUCTIONS))
        raise ValueError(
            f"rmu_loss_reduction must be one of {expected}, got {value!r}."
        )
    return normalized


def normalize_rmu_loss_compute_dtype(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    if normalized not in _RMU_LOSS_COMPUTE_DTYPES:
        expected = ", ".join(sorted(_RMU_LOSS_COMPUTE_DTYPES))
        raise ValueError(
            f"rmu_loss_compute_dtype must be one of {expected}, got {value!r}."
        )
    return normalized


class RmuActivationRecorder:
    def __init__(
        self,
        model: torch.nn.Module,
        layer_indices: list[int] | tuple[int, ...],
        *,
        module_template: str | None = None,
    ) -> None:
        self._layer_indices = tuple(int(layer_idx) for layer_idx in layer_indices)
        if not self._layer_indices:
            raise ValueError("RmuActivationRecorder requires at least one layer index.")
        self._activations: dict[int, torch.Tensor] = {}
        self._enabled = False
        self._handles = []
        for layer_idx in self._layer_indices:
            module = resolve_layer_module(
                model, layer_idx, module_template=module_template
            )
            self._handles.append(
                module.register_forward_hook(
                    self._make_hook(layer_idx), with_kwargs=False
                )
            )

    def clear(self) -> None:
        self._activations.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    @contextmanager
    def recording(self):
        previous = self._enabled
        self._enabled = True
        self.clear()
        try:
            yield self
        finally:
            self._enabled = previous

    def activations(self) -> dict[int, torch.Tensor]:
        missing = [
            layer_idx
            for layer_idx in self._layer_indices
            if layer_idx not in self._activations
        ]
        if missing:
            raise RuntimeError(f"Missing RMU activations for layer indices: {missing}")
        return dict(self._activations)

    def _make_hook(self, layer_idx: int):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            if not self._enabled:
                return
            activation = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(activation):
                self._activations[layer_idx] = activation

        return hook


def resolve_layer_module(
    model: torch.nn.Module,
    layer_idx: int,
    *,
    module_template: str | None = None,
) -> torch.nn.Module:
    if module_template:
        return _resolve_module_template(model, module_template, layer_idx)

    from moe.model_adapters import model_adapter

    return model_adapter(model).text_layer(model, layer_idx)


def _resolve_module_template(
    model: torch.nn.Module, module_template: str, layer_idx: int
) -> torch.nn.Module:
    path = module_template.format(
        layer_idx=int(layer_idx), layer_id=int(layer_idx), model="", model_name=""
    )
    path = path.strip(".")
    current: Any = model
    if not path:
        return current
    for part in path.split("."):
        if not part:
            continue
        match = re.fullmatch(
            r"(?P<attr>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<index>\d+)\])?", part
        )
        if match is None:
            raise ValueError(f"Unsupported rmu_module_template component: {part!r}")
        current = getattr(current, match.group("attr"))
        if match.group("index") is not None:
            current = current[int(match.group("index"))]
    if not isinstance(current, torch.nn.Module):
        raise ValueError(
            f"rmu_module_template did not resolve to a torch.nn.Module: {module_template!r}"
        )
    return current


def rmu_token_mask_from_batch(
    batch: dict[str, torch.Tensor], mode: str
) -> torch.Tensor | None:
    normalized = str(mode).lower().replace("-", "_")
    if normalized in {"all", "none"}:
        return None
    if normalized == "labels":
        labels = batch.get("labels")
        if labels is None:
            return None
        return labels.ne(-100)
    if normalized in {"attention", "attention_mask"}:
        attention_mask = batch.get("attention_mask")
        return (
            attention_mask.to(dtype=torch.bool) if attention_mask is not None else None
        )
    if normalized == "last_nonpad":
        attention_mask = batch.get("attention_mask")
        if attention_mask is None or attention_mask.ndim != 2:
            return None
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).clamp_min(1) - 1
        mask.scatter_(1, lengths[:, None], True)
        return mask
    raise ValueError(
        "rmu_token_mask must be one of 'labels', 'attention', 'last_nonpad', or 'all'."
    )


def compute_rmu_loss(
    forget_activations: dict[int, torch.Tensor],
    layer_indices: tuple[int, ...],
    forget_token_mask: torch.Tensor | None,
    target_cache: RmuControlVectorCache,
    *,
    max_tokens_per_batch: int | None = None,
    loss_reduction: str = "token_mean",
    loss_compute_dtype: str = "float32",
) -> RmuLossResult:
    missing = [
        layer_idx for layer_idx in layer_indices if layer_idx not in forget_activations
    ]
    if missing:
        raise RuntimeError(
            f"Missing forget RMU activations for layer indices: {missing}"
        )

    anchor = next(iter(forget_activations.values())).sum() * 0.0
    zero = anchor.to(dtype=torch.float32)
    layer_terms: list[RmuLayerTerms] = []
    for layer_idx in layer_indices:
        terms = _rmu_layer_terms(
            forget_activations[layer_idx],
            layer_idx=layer_idx,
            token_mask=forget_token_mask,
            target_cache=target_cache,
            max_tokens_per_batch=max_tokens_per_batch,
            loss_reduction=loss_reduction,
            loss_compute_dtype=loss_compute_dtype,
        )
        if terms is not None:
            layer_terms.append(terms)

    return RmuLossResult(
        forget_loss=_mean_or_zero([terms.forget_loss for terms in layer_terms], zero),
        forget_activation_norm=_mean_or_zero(
            [terms.forget_activation_norm for terms in layer_terms], zero
        ),
        target_norm=_mean_or_zero([terms.target_norm for terms in layer_terms], zero),
        forget_cosine=_mean_or_zero(
            [terms.forget_cosine for terms in layer_terms], zero
        ),
        forget_tokens=_mean_or_zero(
            [terms.forget_tokens for terms in layer_terms], zero
        ),
    )


def compute_rmu_retain_loss(
    updated_activations: dict[int, torch.Tensor],
    frozen_activations: dict[int, torch.Tensor],
    layer_indices: tuple[int, ...],
    retain_token_mask: torch.Tensor | None,
    *,
    max_tokens_per_batch: int | None = None,
    loss_reduction: str = "token_mean",
    loss_compute_dtype: str = "float32",
) -> RmuRetainLossResult:
    missing_updated = [
        layer_idx for layer_idx in layer_indices if layer_idx not in updated_activations
    ]
    missing_frozen = [
        layer_idx for layer_idx in layer_indices if layer_idx not in frozen_activations
    ]
    if missing_updated:
        raise RuntimeError(
            f"Missing updated retain RMU activations for layer indices: {missing_updated}"
        )
    if missing_frozen:
        raise RuntimeError(
            f"Missing frozen retain RMU activations for layer indices: {missing_frozen}"
        )

    anchor = next(iter(updated_activations.values())).sum() * 0.0
    zero = anchor.to(dtype=torch.float32)
    layer_terms: list[RmuRetainLayerTerms] = []
    for layer_idx in layer_indices:
        terms = _rmu_retain_layer_terms(
            updated_activations[layer_idx],
            frozen_activations[layer_idx],
            token_mask=retain_token_mask,
            max_tokens_per_batch=max_tokens_per_batch,
            loss_reduction=loss_reduction,
            loss_compute_dtype=loss_compute_dtype,
        )
        if terms is not None:
            layer_terms.append(terms)

    return RmuRetainLossResult(
        retain_loss=_mean_or_zero([terms.retain_loss for terms in layer_terms], zero),
        retain_activation_norm=_mean_or_zero(
            [terms.retain_activation_norm for terms in layer_terms], zero
        ),
        frozen_activation_norm=_mean_or_zero(
            [terms.frozen_activation_norm for terms in layer_terms], zero
        ),
        retain_cosine=_mean_or_zero(
            [terms.retain_cosine for terms in layer_terms], zero
        ),
        retain_tokens=_mean_or_zero(
            [terms.retain_tokens for terms in layer_terms], zero
        ),
    )


def _rmu_layer_terms(
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    token_mask: torch.Tensor | None,
    target_cache: RmuControlVectorCache,
    max_tokens_per_batch: int | None,
    loss_reduction: str,
    loss_compute_dtype: str,
) -> RmuLayerTerms | None:
    loss_reduction = normalize_rmu_loss_reduction(loss_reduction)
    loss_compute_dtype = normalize_rmu_loss_compute_dtype(loss_compute_dtype)
    if loss_reduction == "sequence_mean" and max_tokens_per_batch is not None:
        raise ValueError(
            "rmu_max_tokens_per_batch must be disabled when rmu_loss_reduction=sequence_mean; "
            "token subsampling would change the reference per-sequence normalization."
        )

    selected_forget = select_hidden_states(
        hidden_states,
        token_mask,
        max_tokens=max_tokens_per_batch,
    )
    if selected_forget.numel() == 0:
        return None

    loss_hidden_states = (
        hidden_states if loss_compute_dtype == "activation" else hidden_states.float()
    )
    loss_dtype = loss_hidden_states.dtype
    target = target_cache.get(
        layer_idx,
        selected_forget.shape[-1],
        loss_hidden_states.device,
        loss_dtype,
    )
    if loss_reduction == "sequence_mean":
        forget_loss = _sequence_mean_activation_mse(
            loss_hidden_states, target, token_mask
        )
    else:
        loss_selected = selected_forget.to(dtype=loss_dtype)
        forget_loss = F.mse_loss(
            loss_selected, target.reshape(1, -1).expand_as(loss_selected)
        )

    selected_forget_float = selected_forget.float()
    metric_target = (
        target.float()
        .reshape(1, selected_forget.shape[-1])
        .expand_as(selected_forget_float)
    )
    return RmuLayerTerms(
        forget_loss=forget_loss,
        forget_activation_norm=selected_forget_float.detach().norm(dim=-1).mean(),
        target_norm=target.detach().float().norm(dim=-1).mean(),
        forget_cosine=F.cosine_similarity(
            selected_forget_float.detach(),
            metric_target.detach(),
            dim=-1,
        ).mean(),
        forget_tokens=selected_forget_float.new_tensor(
            selected_forget.shape[0], dtype=torch.float32
        ),
    )


def _sequence_mean_activation_mse(
    hidden_states: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError(
            "rmu_loss_reduction=sequence_mean requires activations shaped [batch, sequence, hidden], "
            f"got {tuple(hidden_states.shape)}."
        )
    batch_size, sequence_length, _ = hidden_states.shape
    if token_mask is None:
        mask = torch.ones(
            (batch_size, sequence_length),
            device=hidden_states.device,
            dtype=torch.bool,
        )
    else:
        mask = token_mask.detach().to(device=hidden_states.device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                "RMU token mask must match the first two activation dimensions for sequence_mean, "
                f"got activation={tuple(hidden_states.shape)} mask={tuple(mask.shape)}."
            )

    expanded_target = target.to(
        device=hidden_states.device, dtype=hidden_states.dtype
    ).expand_as(hidden_states)
    squared_diff = F.mse_loss(hidden_states, expanded_target, reduction="none")
    per_token_loss = squared_diff.mean(dim=-1) * mask.to(dtype=squared_diff.dtype)
    valid_tokens = mask.sum(dim=-1).clamp_min(1).to(dtype=per_token_loss.dtype)
    return (per_token_loss.sum(dim=-1) / valid_tokens).mean()


def _rmu_retain_layer_terms(
    updated_hidden_states: torch.Tensor,
    frozen_hidden_states: torch.Tensor,
    *,
    token_mask: torch.Tensor | None,
    max_tokens_per_batch: int | None,
    loss_reduction: str,
    loss_compute_dtype: str,
) -> RmuRetainLayerTerms | None:
    loss_reduction = normalize_rmu_loss_reduction(loss_reduction)
    loss_compute_dtype = normalize_rmu_loss_compute_dtype(loss_compute_dtype)
    if loss_reduction == "sequence_mean" and max_tokens_per_batch is not None:
        raise ValueError(
            "rmu_max_tokens_per_batch must be disabled when rmu_loss_reduction=sequence_mean; "
            "token subsampling would change the reference per-sequence normalization."
        )
    selected_updated, selected_frozen = select_hidden_state_pair(
        updated_hidden_states,
        frozen_hidden_states,
        token_mask,
        max_tokens=max_tokens_per_batch,
    )
    if selected_updated.numel() == 0:
        return None

    if loss_reduction == "sequence_mean":
        loss_updated = (
            updated_hidden_states
            if loss_compute_dtype == "activation"
            else updated_hidden_states.float()
        )
        loss_frozen = frozen_hidden_states.detach().to(dtype=loss_updated.dtype)
        retain_loss = _sequence_mean_activation_mse(
            loss_updated, loss_frozen, token_mask
        )
    else:
        loss_updated = (
            selected_updated
            if loss_compute_dtype == "activation"
            else selected_updated.float()
        )
        loss_frozen = selected_frozen.detach().to(dtype=loss_updated.dtype)
        retain_loss = F.mse_loss(loss_updated, loss_frozen)

    selected_updated = selected_updated.float()
    selected_frozen = selected_frozen.detach().float()
    return RmuRetainLayerTerms(
        retain_loss=retain_loss,
        retain_activation_norm=selected_updated.detach().norm(dim=-1).mean(),
        frozen_activation_norm=selected_frozen.detach().norm(dim=-1).mean(),
        retain_cosine=F.cosine_similarity(
            selected_updated.detach(), selected_frozen.detach(), dim=-1
        ).mean(),
        retain_tokens=selected_updated.new_tensor(
            selected_updated.shape[0], dtype=torch.float32
        ),
    )


def select_hidden_states(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
    *,
    max_tokens: int | None,
) -> torch.Tensor:
    return _subsample_rows(
        _flatten_hidden_states_with_mask(hidden_states, token_mask),
        max_tokens=max_tokens,
    )


def select_hidden_state_pair(
    updated_hidden_states: torch.Tensor,
    frozen_hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
    *,
    max_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(updated_hidden_states.shape) != tuple(frozen_hidden_states.shape):
        raise ValueError(
            "Updated and frozen RMU retain activations must have identical shapes, "
            f"got updated={tuple(updated_hidden_states.shape)} frozen={tuple(frozen_hidden_states.shape)}."
        )
    updated_flat = _flatten_hidden_states_with_mask(updated_hidden_states, token_mask)
    frozen_flat = _flatten_hidden_states_with_mask(frozen_hidden_states, token_mask)
    if updated_flat.shape[0] != frozen_flat.shape[0]:
        raise ValueError(
            "Updated and frozen RMU retain activations selected different token counts, "
            f"got updated={updated_flat.shape[0]} frozen={frozen_flat.shape[0]}."
        )
    return _subsample_row_pair(updated_flat, frozen_flat, max_tokens=max_tokens)


def _flatten_hidden_states_with_mask(
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    if hidden_states.ndim < 2:
        raise ValueError(
            f"RMU activations must have at least 2 dimensions, got shape={tuple(hidden_states.shape)}."
        )
    hidden_size = hidden_states.shape[-1]
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    if token_mask is not None:
        mask = token_mask.detach().to(device=hidden_states.device, dtype=torch.bool)
        if hidden_states.ndim >= 3 and tuple(hidden_states.shape[: mask.ndim]) == tuple(
            mask.shape
        ):
            flat_hidden = hidden_states[mask].reshape(-1, hidden_size)
        elif (
            mask.ndim == 2
            and hidden_states.ndim == 3
            and hidden_states.shape[:2] == (mask.shape[1], mask.shape[0])
        ):
            flat_hidden = hidden_states.transpose(0, 1)[mask].reshape(-1, hidden_size)
        elif mask.numel() == flat_hidden.shape[0]:
            flat_hidden = flat_hidden[mask.reshape(-1)]
    return flat_hidden


def _subsample_rows(values: torch.Tensor, *, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is None or max_tokens <= 0 or values.shape[0] <= max_tokens:
        return values
    indices = torch.randperm(values.shape[0], device=values.device)[:max_tokens]
    return values.index_select(0, indices)


def _subsample_row_pair(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    max_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_tokens is None or max_tokens <= 0 or first.shape[0] <= max_tokens:
        return first, second
    indices = torch.randperm(first.shape[0], device=first.device)[:max_tokens]
    return first.index_select(0, indices), second.index_select(
        0, indices.to(device=second.device)
    )


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero
