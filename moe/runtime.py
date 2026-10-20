from __future__ import annotations

import copy
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from common.io_utils import read_json
from common.manifest_names import EXPANSION_MANIFEST
from moe.model_adapters import MoeLayer, canonical_name, model_adapter, route_logits


@dataclass(frozen=True)
class ExpandedMoeRuntimeSummary:
    converted_layers: int
    old_num_experts: int
    harmful_experts: tuple[int, ...]
    num_harmful_experts: int
    new_num_experts: int


@dataclass(frozen=True)
class SplitPackedMoeWeights:
    std_gate_weight: torch.Tensor
    harmful_gate_weight: torch.Tensor
    std_gate_up_proj: torch.Tensor
    harmful_gate_up_proj: torch.Tensor
    std_down_proj: torch.Tensor
    harmful_down_proj: torch.Tensor


class RexHarmfulGate(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)


class MegatronExpertMasker:
    """Temporarily exclude the appended bank before MCore softmax/top-k.

    Keep all EP tensors and dispatch dimensions intact. This is a runtime mask,
    not a physical checkpoint transformation. Gemma's per-expert scaling stays
    in the native MoE layer, after routing.
    """

    def __init__(
        self,
        layers: list[MoeLayer],
        checkpoint: str | Path,
        *,
        adapter_name: str | None = None,
    ) -> None:
        self.old_num_experts, self.harmful_experts, self.new_num_experts = (
            _load_expanded_moe_runtime_metadata(checkpoint, layers, adapter_name=adapter_name)
        )
        self.routers = [layer.router for layer in layers]
        for router in self.routers:
            config = router.config
            if (
                int(config.num_moe_experts) != self.new_num_experts
                or int(router.weight.shape[0]) != self.new_num_experts
            ):
                raise ValueError("Megatron router expert count does not match expansion manifest.")
            if (
                not callable(getattr(router, "routing", None))
                or not 0 < int(router.topk) <= self.old_num_experts
            ):
                raise ValueError("Unsupported Megatron router or top-k exceeds retained experts.")
            if getattr(config, "moe_router_score_function", "softmax") != "softmax":
                raise ValueError("Runtime expert masking requires softmax routing.")
            if (
                getattr(router, "routing_type", None) == "sinkhorn"
                or getattr(config, "moe_router_num_groups", None) not in (None, 1)
                or getattr(config, "moe_expert_capacity_factor", None) is not None
                or getattr(config, "moe_router_force_load_balancing", False)
                or getattr(config, "moe_router_force_biased", None) is not None
            ):
                raise ValueError("Runtime masking does not support grouped, capacity-limited or forced routing.")

    @contextmanager
    def masked(self):
        originals = []
        try:
            for router in self.routers:
                method = router.routing
                originals.append((router, "routing" in vars(router), vars(router).get("routing")))

                def routing(logits, *args, _original=method, **kwargs):
                    logits = logits.clone()
                    logits[..., self.old_num_experts:] = -torch.inf
                    return _original(logits, *args, **kwargs)

                router.routing = routing
            yield
        finally:
            for router, had_instance_method, original in reversed(originals):
                if had_instance_method:
                    router.routing = original
                else:
                    del router.routing


def load_expanded_moe_manifest_metadata(
    initial_checkpoint_dir: str | Path, *, require_expert_list: bool = True,
) -> tuple[int, tuple[int, ...], int]:
    manifest_path, manifest = _read_expanded_moe_manifest(initial_checkpoint_dir)
    return _expanded_moe_manifest_metadata(
        manifest, manifest_path, require_expert_list=require_expert_list
    )


def _read_expanded_moe_manifest(checkpoint: str | Path) -> tuple[Path, dict]:
    manifest_path = Path(checkpoint) / EXPANSION_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing expert expansion manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expansion manifest must be a JSON object: {manifest_path}")
    return manifest_path, manifest


def _expanded_moe_manifest_metadata(
    manifest: dict, manifest_path: Path, *, require_expert_list: bool
) -> tuple[int, tuple[int, ...], int]:
    if manifest.get("expert_layout") not in {"packed", "split"}:
        raise ValueError(
            "Expanded MoE runtime requires a packed or split expanded checkpoint, "
            f"got expert_layout={manifest.get('expert_layout')!r}."
        )
    old_num_experts = int(manifest.get("old_num_experts", 0))
    new_num_experts = int(manifest.get("new_num_experts", 0))
    if old_num_experts <= 0 or new_num_experts <= old_num_experts:
        raise ValueError(f"Invalid expert counts in manifest: {manifest_path}")

    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"Manifest has no layers: {manifest_path}")

    expected = _manifest_layer_new_experts(layers[0], manifest_path, require_list=require_expert_list)
    for layer in layers[1:]:
        current = _manifest_layer_new_experts(layer, manifest_path, require_list=require_expert_list)
        if current != expected:
            raise ValueError("All manifest layers must have identical new_experts for expanded MoE runtime.")
    required = tuple(range(old_num_experts, new_num_experts))
    if expected != required:
        raise ValueError(
            "Expanded MoE runtime requires tail-contiguous harmful experts: "
            f"expected={list(required)} got={list(expected)}."
        )
    return old_num_experts, expected, new_num_experts


def _load_expanded_moe_runtime_metadata(
    checkpoint: str | Path,
    layers: list[MoeLayer],
    *,
    adapter_name: str | None = None,
) -> tuple[int, tuple[int, ...], int]:
    manifest_path, manifest = _read_expanded_moe_manifest(checkpoint)
    metadata = _expanded_moe_manifest_metadata(
        manifest, manifest_path, require_expert_list=True
    )
    expected = [int(layer.get("layer", i)) for i, layer in enumerate(manifest["layers"])]
    if (
        len(set(expected)) != len(expected)
        or sorted(expected) != sorted(layer.index for layer in layers)
    ):
        raise ValueError("Runtime text MoE layers do not match expansion manifest layers.")
    if adapter_name is not None and manifest.get("model_adapter", adapter_name) != adapter_name:
        raise ValueError("Expansion manifest model_adapter does not match model config.")
    return metadata


def _manifest_layer_new_experts(layer: object, manifest_path: Path, *, require_list: bool) -> tuple[int, ...]:
    # Offline excision historically accepted iterable expert ids; runtime
    # requires a JSON list. Keep that input-policy difference explicit.
    if not isinstance(layer, dict):
        raise ValueError(f"Manifest layer is not an object: {manifest_path}")
    new_experts = layer.get("new_experts", [])
    if not require_list:
        return tuple(int(expert) for expert in new_experts)
    if not isinstance(new_experts, list) or not new_experts:
        raise ValueError(f"Every manifest layer must define non-empty new_experts: {manifest_path}")
    return tuple(int(expert) for expert in new_experts)


def convert_packed_moe_to_expanded_runtime(
    model: torch.nn.Module,
    *,
    initial_checkpoint_dir: str | Path,
) -> ExpandedMoeRuntimeSummary:
    adapter = model_adapter(model)
    layers = list(adapter.moe_layers(model))
    old_num_experts, manifest_harmful_experts, new_num_experts = (
        _load_expanded_moe_runtime_metadata(
            initial_checkpoint_dir, layers, adapter_name=adapter.name
        )
    )
    converted = 0
    for layer in layers:
        adapter.convert_layer_to_expanded(layer.owner, old_num_experts, new_num_experts)
        converted += 1
    if converted == 0:
        raise ValueError("Could not find packed MoE blocks to convert to expanded MoE runtime.")
    setattr(model, "_rex_expanded_moe_runtime", True)
    setattr(model, "_rex_expanded_moe_old_num_experts", old_num_experts)
    setattr(model, "_rex_expanded_moe_new_num_experts", new_num_experts)
    return ExpandedMoeRuntimeSummary(
        converted_layers=converted,
        old_num_experts=old_num_experts,
        harmful_experts=manifest_harmful_experts,
        num_harmful_experts=len(manifest_harmful_experts),
        new_num_experts=new_num_experts,
    )


def merge_expanded_moe_state_dict_for_save(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not getattr(model, "_rex_expanded_moe_runtime", False):
        raise RuntimeError("Expected an expanded MoE runtime model before saving REX checkpoints.")

    merged = dict(state_dict)
    adapter = model_adapter(model)
    for layer in adapter.moe_layers(model):
        prefix = canonical_name(layer.name) + "."
        for std, harmful in adapter.runtime_merge_pairs():
            _append_harmful_rows_for_save(merged, std_key=prefix + std, harmful_key=prefix + harmful)

    return merged


def _append_harmful_rows_for_save(
    state_dict: dict[str, torch.Tensor],
    *,
    std_key: str,
    harmful_key: str,
) -> None:
    if std_key not in state_dict:
        raise KeyError(f"Missing std key for expanded MoE merge: {std_key}")
    std_tensor = state_dict[std_key]
    harmful_tensor = state_dict[harmful_key].to(device=std_tensor.device, dtype=std_tensor.dtype)
    state_dict[std_key] = torch.cat([std_tensor, harmful_tensor], dim=0)
    del state_dict[harmful_key]


def _convert_moe_block(module: torch.nn.Module, *, old_num_experts: int, new_num_experts: int) -> None:
    if getattr(module, "_rex_expanded_moe", False):
        return
    gate = module.gate
    experts = module.experts
    num_harmful_experts = int(new_num_experts - old_num_experts)
    weights = _split_packed_moe_weights(
        gate,
        experts,
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
    )

    _install_expanded_router(
        module,
        gate,
        weights,
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
        num_harmful_experts=num_harmful_experts,
    )
    _install_expanded_experts(
        module,
        experts,
        weights,
        old_num_experts=old_num_experts,
    )
    _mark_expanded_moe_module(
        module,
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
        num_harmful_experts=num_harmful_experts,
    )


def _split_packed_moe_weights(
    gate: torch.nn.Module,
    experts: torch.nn.Module,
    *,
    old_num_experts: int,
    new_num_experts: int,
) -> SplitPackedMoeWeights:
    std_gate_weight, harmful_gate_weight = _split_packed_expert_rows(
        "Router weight",
        gate.weight.detach(),
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
    )
    std_gate_up_proj, harmful_gate_up_proj = _split_packed_expert_rows(
        "gate_up_proj",
        experts.gate_up_proj.detach(),
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
    )
    std_down_proj, harmful_down_proj = _split_packed_expert_rows(
        "down_proj",
        experts.down_proj.detach(),
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
    )
    return SplitPackedMoeWeights(
        std_gate_weight=std_gate_weight,
        harmful_gate_weight=harmful_gate_weight,
        std_gate_up_proj=std_gate_up_proj,
        harmful_gate_up_proj=harmful_gate_up_proj,
        std_down_proj=std_down_proj,
        harmful_down_proj=harmful_down_proj,
    )


def _install_expanded_router(
    module: torch.nn.Module,
    gate: torch.nn.Module,
    weights: SplitPackedMoeWeights,
    *,
    old_num_experts: int,
    new_num_experts: int,
    num_harmful_experts: int,
) -> None:
    module.harmful_gate = RexHarmfulGate(weights.harmful_gate_weight)
    gate.weight = nn.Parameter(weights.std_gate_weight.clone())
    gate.num_experts = int(new_num_experts)
    gate.num_std_experts = int(old_num_experts)
    gate.num_harmful_experts = num_harmful_experts
    gate._rex_expanded_moe_router = True
    gate._rex_harmful_gate_ref = weakref.ref(module.harmful_gate)
    gate.forward = types.MethodType(_expanded_moe_router_forward, gate)


def _install_expanded_experts(
    module: torch.nn.Module,
    experts: torch.nn.Module,
    weights: SplitPackedMoeWeights,
    *,
    old_num_experts: int,
) -> None:
    module.harmful_experts = _make_harmful_experts_like(
        experts,
        gate_up_proj=weights.harmful_gate_up_proj,
        down_proj=weights.harmful_down_proj,
    )
    experts.gate_up_proj = nn.Parameter(weights.std_gate_up_proj.clone())
    experts.down_proj = nn.Parameter(weights.std_down_proj.clone())
    experts.num_experts = int(old_num_experts)
    if hasattr(experts, "config"):
        experts.config = copy.copy(experts.config)
        experts.config.num_experts = int(old_num_experts)


def _mark_expanded_moe_module(
    module: torch.nn.Module,
    *,
    old_num_experts: int,
    new_num_experts: int,
    num_harmful_experts: int,
) -> None:
    module.num_std_experts = int(old_num_experts)
    module.num_harmful_experts = num_harmful_experts
    module.num_experts = int(new_num_experts)
    module._rex_expanded_moe = True
    module.forward = types.MethodType(_expanded_moe_forward, module)


def _split_packed_expert_rows(
    name: str,
    tensor: torch.Tensor,
    *,
    old_num_experts: int,
    new_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.shape[0] != new_num_experts:
        raise ValueError(
            f"{name} expert count {tensor.shape[0]} does not match manifest new_num_experts={new_num_experts}."
        )
    return tensor[:old_num_experts], tensor[old_num_experts:new_num_experts]


def _expanded_moe_router_forward(gate: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = hidden_states.reshape(-1, int(gate.hidden_dim))
    std_logits = F.linear(hidden_states, gate.weight)
    harmful_gate_ref = getattr(gate, "_rex_harmful_gate_ref", None)
    harmful_gate = harmful_gate_ref() if harmful_gate_ref is not None else None
    if harmful_gate is None:
        raise RuntimeError("Expanded MoE router lost its harmful_gate reference.")
    harmful_logits = harmful_gate(hidden_states)
    router_logits = torch.cat([std_logits, harmful_logits], dim=-1)
    return route_logits(router_logits, int(gate.top_k))


def _expanded_moe_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = module.shared_expert(hidden_states_reshaped)
    _, routing_weights, selected_experts = module.gate(hidden_states_reshaped)
    expert_output = _expanded_moe_experts_forward(module, hidden_states_reshaped, selected_experts, routing_weights)
    shared_expert_output = torch.sigmoid(module.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
    expert_output = expert_output + shared_expert_output
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


def _expanded_moe_experts_forward(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    std_experts = module.experts
    harmful_experts = module.harmful_experts
    std_count = int(module.num_std_experts)
    num_experts = std_count + int(module.num_harmful_experts)

    num_top_k = int(top_k_index.size(-1))
    num_tokens, hidden_dim = hidden_states.shape
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[torch.div(perm, num_top_k, rounding_mode="floor")]
    sample_weights_g = sample_weights[perm]

    histc_input = expert_ids_g.float() if hidden_states.device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=num_experts, min=0, max=num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    std_total = offsets[std_count - 1]

    std_out = _expanded_moe_expert_bank_forward(
        std_experts,
        selected_hidden_states_g[:std_total],
        sample_weights_g[:std_total],
        offsets[:std_count],
    )
    harmful_out = _expanded_moe_expert_bank_forward(
        harmful_experts,
        selected_hidden_states_g[std_total:],
        sample_weights_g[std_total:],
        offsets[std_count:] - std_total,
    )
    weighted_out_g = torch.cat([std_out, harmful_out], dim=0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=hidden_states.device)
    weighted_out = weighted_out_g[inv_perm]
    return weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def _expanded_moe_expert_bank_forward(
    experts: torch.nn.Module,
    hidden_states: torch.Tensor,
    sample_weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    gate_up = _grouped_linear(hidden_states, experts.gate_up_proj, offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden_states = experts.act_fn(gate) * up
    hidden_states = _grouped_linear(hidden_states, experts.down_proj, offsets)
    hidden_states = hidden_states * sample_weights.unsqueeze(-1)
    return hidden_states.to(sample_weights.dtype)


def _grouped_linear(input: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return _grouped_mm(input, weight.transpose(-2, -1), offsets)


def _grouped_mm(input: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if input.numel() == 0:
        return input.new_empty((0, weight.size(-1)), dtype=weight.dtype)
    if input.device.type == "cuda":
        if hasattr(F, "grouped_mm"):
            return F.grouped_mm(input.to(weight.dtype), weight, offs=offsets)
        return torch._grouped_mm(input.to(weight.dtype), weight, offs=offsets)
    return _grouped_mm_fallback(input.to(weight.dtype), weight, offsets)


def _grouped_mm_fallback(input: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    output_parts: list[torch.Tensor] = []
    start = 0
    for expert_idx, end in enumerate(offsets.tolist()):
        if end > start:
            output_parts.append(input[start:end].matmul(weight[expert_idx]))
        start = end
    if not output_parts:
        return input.new_empty((0, weight.size(-1)), dtype=weight.dtype)
    return torch.cat(output_parts, dim=0)


def _make_harmful_experts_like(
    experts: torch.nn.Module,
    *,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.nn.Module:
    # A bank only needs weights and the native activation, not a model config.
    harmful_experts = nn.Module()
    harmful_experts.act_fn = experts.act_fn
    harmful_experts.gate_up_proj = nn.Parameter(gate_up_proj.detach().clone())
    harmful_experts.down_proj = nn.Parameter(down_proj.detach().clone())
    harmful_experts.num_experts = int(gate_up_proj.shape[0])
    return harmful_experts


def _convert_gemma_layer(layer: nn.Module, *, old_num_experts: int, new_num_experts: int) -> None:
    old, new = old_num_experts, new_num_experts
    if getattr(layer, "_rex_expanded_moe", False):
        return
    router = layer.router
    weights = _split_packed_moe_weights(router.proj, layer.experts, old_num_experts=old, new_num_experts=new)
    std_scale, harmful_scale = _split_packed_expert_rows(
        "per_expert_scale", router.per_expert_scale.detach(), old_num_experts=old, new_num_experts=new)
    layer.harmful_gate = RexHarmfulGate(weights.harmful_gate_weight)
    layer.harmful_gate.per_expert_scale = nn.Parameter(harmful_scale.clone())
    router.proj.weight = nn.Parameter(weights.std_gate_weight.clone())
    router.proj.out_features = old
    router.per_expert_scale = nn.Parameter(std_scale.clone())
    router.num_experts, router.top_k = new, int(router.config.top_k_experts)
    router._rex_harmful_gate_ref = weakref.ref(layer.harmful_gate)
    router.forward = types.MethodType(_gemma_router_forward, router)
    _install_expanded_experts(layer, layer.experts, weights, old_num_experts=old)
    layer.num_std_experts, layer.num_harmful_experts = old, new - old
    layer.num_experts = new
    layer._rex_expanded_moe = True
    # Keep the entire native decoder forward (norms, dense MLP and residuals).
    layer.experts._rex_owner_ref = weakref.ref(layer)
    layer.experts.forward = types.MethodType(_gemma_experts_forward, layer.experts)


def _gemma_router_forward(router: nn.Module, hidden_states: torch.Tensor):
    hidden_states = router.norm(hidden_states) * router.scale * router.scalar_root_size
    harmful = router._rex_harmful_gate_ref()
    if harmful is None:
        raise RuntimeError("Gemma router lost harmful bank reference.")
    logits = torch.cat((router.proj(hidden_states), harmful(hidden_states)), -1)
    scale = torch.cat((router.per_expert_scale, harmful.per_expert_scale))
    return route_logits(logits, router.top_k, gemma=True, scale=scale)


def _gemma_experts_forward(experts: nn.Module, hidden_states, top_k_index, top_k_weights):
    owner = experts._rex_owner_ref()
    if owner is None:
        raise RuntimeError("Gemma expert bank lost decoder reference.")
    return _expanded_moe_experts_forward(owner, hidden_states, top_k_index, top_k_weights)
