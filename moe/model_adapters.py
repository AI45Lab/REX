"""Architecture contracts shared by training and checkpoint transformations.

Only text decoder MoE parameters belong to these contracts. In particular MTP
and vision layers are never discovered by an unanchored ``layers`` match.
"""

from __future__ import annotations

from safetensors import safe_open

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

LAYER_PREFIX_RE = r"(?:model|model\.language_model)\.layers"
SPLIT_EXPERT_KEY_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(?P<layer>\d+)\.(?:mlp\.)?experts\."
    r"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
PACKED_EXPERT_KEY_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(?P<layer>\d+)\.(?:mlp\.)?experts\.(?P<proj>gate_up_proj|down_proj)$"
)
GATE_KEY_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(?P<layer>\d+)\.(?:mlp\.gate|router\.proj)\.weight$"
)
EXPERT_SCALE_KEY_RE = re.compile(
    rf"^{LAYER_PREFIX_RE}\.(?P<layer>\d+)\.router\.per_expert_scale$"
)


def canonical_name(name: str) -> str:
    return ".".join(
        p
        for p in name.split(".")
        if p
        not in {
            "_fsdp_wrapped_module",
            "_checkpoint_wrapped_module",
            "_orig_mod",
            "module",
        }
    )


@dataclass(frozen=True)
class ParameterRole:
    role: str = "shared"
    layer: int | None = None
    projection: str | None = None


@dataclass(frozen=True)
class MoeLayer:
    name: str
    index: int
    owner: torch.nn.Module
    router: torch.nn.Module
    experts: torch.nn.Module


@dataclass(frozen=True)
class ActivationMaskSpec:
    target: torch.nn.Module
    num_experts: int
    top_k: int
    projection_output: bool = False


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    decoder_class: str
    moe_suffix: str
    router_name: str
    top_k_field: str
    expert_prefix: str = "mlp."
    router_weight_tail: str = "gate.weight"
    router_aux_keys: tuple[str, ...] = ()
    expert_count_config_keys: tuple[str, ...] = ("num_experts", "num_local_experts", "n_routed_experts")
    expert_projection_names: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
    packed_projection_names: tuple[str, ...] = ("gate_up_proj", "down_proj")
    runtime_converter: str = "_convert_moe_block"

    def router_weight_key(self, layer: int, *, layer_prefix: str = "model.layers") -> str:
        return f"{layer_prefix}.{layer}{self.moe_suffix}.{self.router_weight_tail}"

    def split_expert_key(self, layer: int, expert: int, proj: str, *, layer_prefix: str = "model.layers") -> str:
        return f"{layer_prefix}.{layer}{self.moe_suffix}.experts.{expert}.{proj}.weight"

    def packed_expert_key(self, layer: int, proj: str, *, layer_prefix: str = "model.layers") -> str:
        return f"{layer_prefix}.{layer}{self.moe_suffix}.experts.{proj}"

    def config_num_experts(self, config: Any) -> int:
        return config_num_experts(config, keys=self.expert_count_config_keys)

    def set_config_num_experts(self, config: dict, count: int, *, old_value: int | None = None) -> None:
        set_config_num_experts(config, count, old_value=old_value, keys=self.expert_count_config_keys)

    def convert_layer_to_expanded(self, layer, old_num_experts: int, new_num_experts: int) -> None:
        from moe import runtime as expanded_moe_runtime
        converter = getattr(expanded_moe_runtime, self.runtime_converter)
        converter(layer, old_num_experts=old_num_experts, new_num_experts=new_num_experts)

    def activation_mask_spec(self, router: torch.nn.Module) -> ActivationMaskSpec:
        """Describe where activation masking can intercept routing.

        Native Gemma returns probabilities from its router. Mask the projection
        before that softmax; expanded routers already expose raw routing logits.
        """
        if self.router_aux_keys and not hasattr(router, "_rex_harmful_gate_ref"):
            projection = router.proj
            return ActivationMaskSpec(
                target=projection,
                num_experts=int(projection.out_features),
                top_k=int(getattr(router.config, self.top_k_field)),
                projection_output=True,
            )
        return ActivationMaskSpec(
            target=router,
            num_experts=int(getattr(router, "num_experts", 0)),
            top_k=int(getattr(router, "top_k", 0)),
        )

    def runtime_merge_pairs(self):
        return ([(self.router_weight_tail, "harmful_gate.weight")]
                + [(f"experts.{p}", f"harmful_experts.{p}") for p in self.packed_projection_names]
                + [(f"{self.router_name}.{p}", f"harmful_gate.{p}") for p in self.router_aux_keys])

    def parameter_role(self, name: str) -> ParameterRole:
        name = canonical_name(name)
        # The final alternative supports standalone MoE modules in unit tests.
        match = re.fullmatch(
            r"(?:(?:model\.(?:language_model\.)?)?layers\.(?P<layer>\d+)\.)?(?P<tail>.+)",
            name,
        )
        layer = int(match["layer"]) if match["layer"] is not None else None
        tail = match["tail"]
        prefix = self.expert_prefix
        projections = r"(?:gate_up_proj|gate_proj|up_proj|down_proj)(?:\.weight)?"
        for bank, role in (
            ("experts", "std_expert"),
            ("harmful_experts", "harmful_expert"),
        ):
            m = re.fullmatch(
                re.escape(prefix + bank + ".") + rf"(?:\d+\.)?(?P<proj>{projections})",
                tail,
            )
            if m:
                return ParameterRole(role, layer, m["proj"].split(".")[0])
        router_tails = (
            {"mlp.gate.weight", "mlp.gate.bias"}
            if prefix
            else {"router.proj.weight", "router.per_expert_scale"}
        )
        harmful_tails = (
            {"mlp.harmful_gate.weight", "mlp.harmful_gate.bias"}
            if prefix
            else {"harmful_gate.weight", "harmful_gate.per_expert_scale"}
        )
        if tail in router_tails:
            return ParameterRole("std_router", layer, "per_expert_scale" if tail == "router.per_expert_scale" else None)
        if tail in harmful_tails:
            return ParameterRole("harmful_router", layer, "per_expert_scale" if tail == "harmful_gate.per_expert_scale" else None)
        shared_prefix = "mlp.shared_expert." if prefix else "mlp."
        if re.fullmatch(re.escape(shared_prefix) + projections, tail):
            return ParameterRole("shared_mlp", layer, tail.split(".")[-2])
        return ParameterRole(layer=layer)

    def moe_layers(self, model: torch.nn.Module) -> Iterator[MoeLayer]:
        pattern = re.compile(
            r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)"
            + re.escape(self.moe_suffix)
            + r"$"
        )
        for name, owner in model.named_modules():
            match = pattern.fullmatch(canonical_name(name))
            if match is None:
                continue
            router = owner._modules.get(self.router_name)
            experts = owner._modules.get("experts")
            if isinstance(router, torch.nn.Module) and isinstance(
                experts, torch.nn.Module
            ):
                yield MoeLayer(name, int(match[1]), owner, router, experts)

    def text_layer(self, model: torch.nn.Module, index: int) -> torch.nn.Module:
        for name, module in model.named_modules():
            if canonical_name(name) in {
                f"model.language_model.layers.{index}",
                f"model.layers.{index}",
                f"layers.{index}",
            }:
                return module
        raise ValueError(f"Missing text decoder layer {index} for {self.name}.")

    def auto_model_class(self, config: Any):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        return (
            AutoModelForCausalLM
            if _value(config, "model_type", "").endswith("_text")
            else AutoModelForImageTextToText
        )


QWEN = ModelAdapter(
    "qwen3_5_moe", "Qwen3_5MoeDecoderLayer", ".mlp", "gate", "num_experts_per_tok"
)
GEMMA = ModelAdapter("gemma4", "Gemma4TextDecoderLayer", "", "router", "top_k_experts",
                     expert_prefix="", router_weight_tail="router.proj.weight",
                     router_aux_keys=("per_expert_scale",), runtime_converter="_convert_gemma_layer")


def megatron_moe_layers(model: torch.nn.Module, adapter: ModelAdapter) -> list[MoeLayer]:
    """Text-only MCore/bridge layout used by the independent Swift SFT workflow.

    TP/PP=1 callers pass the unwrapped model. Gemma's always-on dense MLP is
    separate from ``experts_mlp``; vision and MTP must never be discovered here.
    """
    gpt = getattr(model, "language_model", model)
    decoder = getattr(gpt, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise ValueError("Expected a Megatron text decoder with layers.")
    owner_name = "experts_mlp" if adapter == GEMMA else "mlp"
    found: list[MoeLayer] = []
    for offset, layer in enumerate(decoder.layers):
        owner = getattr(layer, owner_name, None)
        router = getattr(owner, "router", None)
        experts = getattr(owner, "experts", None)
        if not isinstance(router, torch.nn.Module) or not isinstance(
            experts, torch.nn.Module
        ):
            continue
        index = int(getattr(layer, "layer_number", offset + 1)) - 1
        found.append(
            MoeLayer(f"decoder.layers.{offset}.{owner_name}", index, owner, router, experts)
        )
    if not found or len({layer.index for layer in found}) != len(found):
        raise ValueError("Missing or duplicate Megatron text MoE layers.")
    return found


def _value(config: Any, key: str, default=None):
    return (
        config.get(key, default)
        if isinstance(config, dict)
        else getattr(config, key, default)
    )


def adapter_from_config(config: Any) -> ModelAdapter:
    kind = _value(config, "model_type")
    if kind in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return QWEN
    if kind in {"gemma4", "gemma4_text"}:
        text = _value(config, "text_config", config)
        if not _value(text, "enable_moe_block", False):
            raise ValueError("Gemma4 requires enable_moe_block=True.")
        return GEMMA
    raise ValueError(
        f"Unsupported model_type={kind!r}; expected Qwen3.5 MoE or Gemma4 MoE."
    )


def adapter_from_checkpoint(path: str | Path) -> ModelAdapter:
    return adapter_from_config(json.loads((Path(path) / "config.json").read_text()))


def tied_weight_aliases(config: Any) -> dict[str, str]:
    """HF aliases that may be omitted by a tied-weight checkpoint export.

    These describe the saved HF layout, not frozen passthrough parameters.
    Reconstruct aliases only from the exported, trained canonical tensor.
    """
    kind = _value(config, "model_type", "")
    if kind not in {"gemma4", "gemma4_text"}:
        return {}
    text = _value(config, "text_config", config)
    if not _value(config, "tie_word_embeddings", False) or not _value(text, "tie_word_embeddings", False):
        return {}
    prefix = "model.language_model" if kind == "gemma4" else "model"
    return {"lm_head.weight": f"{prefix}.embed_tokens.weight"}


def model_adapter(model: torch.nn.Module) -> ModelAdapter:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model must declare config.model_type for adapter selection.")
    return adapter_from_config(config)


def parameter_role(model: torch.nn.Module, name: str) -> ParameterRole:
    return model_adapter(model).parameter_role(name)


def checkpoint_metadata(config: dict[str, Any]) -> dict[str, Any]:
    if "model_type" not in config:  # historical synthetic checkpoints
        return {}
    adapter = adapter_from_config(config)
    return {
        "model_adapter": adapter.name,
        "layout_schema_version": 1,
        "router_auxiliary_rows": list(adapter.router_aux_keys),
    }


def read_weight_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if (path / "model.safetensors.index.json").is_file():
        return json.loads((path / "model.safetensors.index.json").read_text())
    if (path / "model.safetensors").is_file():
        header = read_safetensors_header(path / "model.safetensors")
        return {"metadata": {}, "weight_map": {k: "model.safetensors" for k in header}}
    raise FileNotFoundError(f"Missing safetensors index or model.safetensors: {path}")


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only JSON metadata; safe_open would mmap a whole 50 GB Gemma shard."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if length > 64 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def validate_checkpoint_layout(path, config, weight_map, count, layers):
    """Validate headers before allocating or writing a transformed checkpoint."""
    text = config.get("text_config", config)
    if int(text.get("num_experts", count)) != count:
        raise ValueError("Config expert count does not match manifest.")
    adapter = adapter_from_config(config) if "model_type" in config else None
    if adapter:
        top_k = int(text.get(adapter.top_k_field, 0))
        if top_k <= 0 or top_k > count:
            raise ValueError("Invalid top-k relative to expert count.")
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate manifest layer ids.")
    if adapter and "num_hidden_layers" in text:
        expected = set(range(int(text["num_hidden_layers"])))
        if set(layers) != expected:
            raise ValueError("Text MoE layer coverage does not match config.")
    gate_layers, scale_layers, packed, split = set(), set(), {}, {}
    by_shard = {}
    for key, shard in weight_map.items():
        match = (
            GATE_KEY_RE.match(key)
            or EXPERT_SCALE_KEY_RE.match(key)
            or PACKED_EXPERT_KEY_RE.match(key)
            or SPLIT_EXPERT_KEY_RE.match(key)
        )
        if match:
            layer = int(match["layer"])
            if adapter:
                is_gemma = ".router." in key or (
                    ".experts." in key and ".mlp.experts." not in key
                )
                if is_gemma != (adapter == GEMMA):
                    raise ValueError(f"Weight layout disagrees with model_type: {key}")
            if GATE_KEY_RE.match(key):
                gate_layers.add(layer)
            elif EXPERT_SCALE_KEY_RE.match(key):
                scale_layers.add(layer)
            elif PACKED_EXPERT_KEY_RE.match(key):
                packed.setdefault(layer, set()).add(match["proj"])
            else:
                split.setdefault(layer, {}).setdefault(int(match["expert"]), set()).add(
                    match["proj"]
                )
            by_shard.setdefault(shard, []).append(key)
    if gate_layers != set(layers):
        raise ValueError("Router layer coverage does not match experts.")
    if adapter == GEMMA and scale_layers != set(layers):
        raise ValueError("Missing Gemma per_expert_scale rows.")
    if packed and split:
        raise ValueError("Mixed packed and split text MoE layout.")
    if packed:
        if set(packed) != set(layers) or any(
            p != {"gate_up_proj", "down_proj"} for p in packed.values()
        ):
            raise ValueError("Incomplete packed expert projections.")
    elif split:
        if set(split) != set(layers) or any(
            set(e) != set(range(count))
            or any(p != {"gate_proj", "up_proj", "down_proj"} for p in e.values())
            for e in split.values()
        ):
            raise ValueError("Incomplete split expert projections or ids.")
    else:
        raise ValueError("No text MoE expert tensors.")
    for shard, keys in by_shard.items():
        header = read_safetensors_header(Path(path) / shard)
        for key in keys:
            shape = header[key]["shape"]
            if not SPLIT_EXPERT_KEY_RE.match(key) and (not shape or shape[0] != count):
                raise ValueError(
                    f"Expert/router shape disagrees with config: {key}: {shape}, expected {count} rows"
                )
            if adapter and "hidden_size" in text and "moe_intermediate_size" in text:
                hidden, intermediate = (
                    int(text["hidden_size"]),
                    int(text["moe_intermediate_size"]),
                )
                if GATE_KEY_RE.match(key):
                    expected = [count, hidden]
                elif EXPERT_SCALE_KEY_RE.match(key):
                    expected = [count]
                elif match := PACKED_EXPERT_KEY_RE.match(key):
                    expected = (
                        [count, hidden, intermediate]
                        if match["proj"] == "down_proj"
                        else [count, 2 * intermediate, hidden]
                    )
                else:
                    match = SPLIT_EXPERT_KEY_RE.match(key)
                    expected = (
                        [hidden, intermediate]
                        if match["proj"] == "down_proj"
                        else [intermediate, hidden]
                    )
                if shape != expected:
                    raise ValueError(
                        f"Projection shape disagrees with config: {key}: {shape}, expected {expected}"
                    )


@dataclass
class RoutingOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    weights: torch.Tensor  # normalized top-k weights before expert scaling
    selected: torch.Tensor
    dispatch_weights: torch.Tensor

    # Native MoE forwards consume a 3-tuple; keep that interface intact.
    def __iter__(self):
        return iter((self.logits, self.dispatch_weights, self.selected))

    def __getitem__(self, index):
        return (self.logits, self.dispatch_weights, self.selected)[index]


def diagnostic_topk_weights(adapter: ModelAdapter, output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Observe normalized top-k probabilities before dispatch casting/scaling.

    Native Gemma exposes probabilities; native Qwen exposes logits. This
    read-only diagnostic contract does not change runtime dispatch or REX's
    existing statistics precision.
    """
    if isinstance(output, RoutingOutput):
        probabilities, selected = output.probabilities.float(), output.selected
    elif isinstance(output, tuple) and len(output) == 3:
        scores, _, selected = output
        if adapter == GEMMA:
            probabilities = scores.float()
        elif adapter == QWEN:
            probabilities = scores.float().softmax(-1)
        else:
            raise ValueError(f"Unsupported diagnostic router: {adapter.name}")
    else:
        raise ValueError("Expected native router output or RoutingOutput")
    selected_probabilities = probabilities.gather(-1, selected)
    denominator = selected_probabilities.sum(-1, keepdim=True)
    if not torch.isfinite(selected_probabilities).all() or not (denominator > 0).all():
        raise ValueError("Invalid diagnostic top-k probabilities")
    return selected_probabilities / denominator, selected


def route_logits(
    logits: torch.Tensor,
    top_k: int,
    *,
    gemma: bool = False,
    scale: torch.Tensor | None = None,
    allowed: int | None = None,
) -> RoutingOutput:
    scores = logits if allowed is None else logits[..., :allowed]
    probabilities = torch.softmax(
        scores, dim=-1, dtype=None if gemma else torch.float32
    )
    weights, selected = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    # Qwen's native router casts normalized weights back before both dispatch
    # and REX statistics. Preserve that precision contract for existing runs.
    weights = weights.to(logits.dtype)
    dispatch = weights if scale is None else weights * scale[selected]
    return RoutingOutput(
        logits, probabilities, weights, selected, dispatch.to(logits.dtype)
    )


def split_expert_key(
    weight_map: dict[str, str],
    layer: int,
    expert: int,
    proj: str,
    *,
    source_key: str | None = None,
) -> str:
    candidates = [
        f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
        f"model.layers.{layer}.experts.{expert}.{proj}.weight",
        f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight",
    ]
    for candidate in candidates:
        if candidate in weight_map:
            return candidate
    if source_key is not None:
        return source_key.replace(f".experts.{source_key.split('.experts.', 1)[1].split('.', 1)[0]}.", f".experts.{expert}.")
    raise KeyError(f"Missing split expert tensor for layer={layer}, expert={expert}, proj={proj}")


class TensorReader:
    def __init__(self, path):
        self.path = path
        self.weight_map = read_weight_index(path)["weight_map"]

    def read(self, key):
        with safe_open(self.path / self.weight_map[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def expert(self, stem, projection, expert):
        """Return a split projection from either on-disk representation."""
        split = split_key_from_stem(stem, expert, projection)
        if split in self.weight_map:
            return self.read(split)
        packed_proj = "down_proj" if projection == "down_proj" else "gate_up_proj"
        packed = self.read(f"{stem}.{packed_proj}")[expert]
        if projection == "down_proj":
            return packed
        return packed.chunk(2, dim=0)[0 if projection == "gate_proj" else 1]

    def packed_tail(self, key, start, end):
        if key in self.weight_map:
            return self.read(key)[start:end]
        stem, projection = key.rsplit(".", 1)
        rows = []
        for expert in range(start, end):
            if projection == "down_proj":
                row = self.expert(stem, projection, expert)
            else:
                row = torch.cat([self.expert(stem, p, expert) for p in ("gate_proj", "up_proj")], 0)
            rows.append(row)
        return torch.stack(rows)


EXPERT_COUNT_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
SPLIT_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PACKED_EXPERT_PROJECTIONS = ("gate_up_proj", "down_proj")


def config_value(config: Any, key: str, default=None):
    value = _value(config, key)
    return value if value is not None else _value(_value(config, "text_config", {}), key, default)


def config_num_experts(config: Any, *, keys=EXPERT_COUNT_CONFIG_KEYS) -> int:
    for key in keys:
        value = config_value(config, key)
        if value is not None:
            return int(value)
    return 0


def set_config_num_experts(config: dict, count: int, *, old_value: int | None = None,
                           keys=EXPERT_COUNT_CONFIG_KEYS) -> None:
    for section in (config, config.get("text_config")):
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key not in section:
                continue
            if old_value is not None:
                try:
                    if int(section[key]) != old_value:
                        continue
                except (ValueError, TypeError):
                    continue
            section[key] = count


def expert_stem(key: str) -> str:
    if SPLIT_EXPERT_KEY_RE.match(key) is None:
        raise ValueError(f"Not a split expert key: {key}")
    return key.split(".experts.", 1)[0] + ".experts"


def split_key_from_stem(stem: str, expert: int, projection: str) -> str:
    return f"{stem}.{expert}.{projection}.weight"


def is_expert_bank_key(key: str) -> bool:
    return any(pattern.match(key) for pattern in
               (GATE_KEY_RE, EXPERT_SCALE_KEY_RE, PACKED_EXPERT_KEY_RE, SPLIT_EXPERT_KEY_RE))
