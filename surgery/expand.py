#!/usr/bin/env python3
"""Expand a MoE checkpoint into F0 by cloning source experts in every sparse layer.

Source experts are ranked per layer by target_freq - control_weight *
control_freq (dispatch frequencies from surgery.routing_stats). The clones and
their router rows get epsilon Gaussian noise; the original source experts are
scaled by source_expert_scale after their copies are made.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from common.io_utils import load_shard, prepare_output_dir, read_json, tensor_nbytes, write_json_atomic
from common.manifest_names import EXPANSION_MANIFEST, LEGACY_RECORD_FILES, RUN_RECORD
from common.run_record import write_run_record
# Shared text-decoder layout contract; excludes vision and MTP.
from moe.model_adapters import (
    EXPERT_SCALE_KEY_RE,
    GATE_KEY_RE,
    PACKED_EXPERT_KEY_RE,
    PACKED_EXPERT_PROJECTIONS,
    SPLIT_EXPERT_KEY_RE,
    SPLIT_EXPERT_PROJECTIONS,
    checkpoint_metadata,
    config_num_experts,
    read_weight_index,
    set_config_num_experts,
    split_expert_key,
    validate_checkpoint_layout,
)

COPY_FILE_SUFFIXES = {
    ".jinja",
    ".json",
    ".model",
    ".py",
    ".txt",
    ".md",
}
# Records of the input describe the input, not F0; the input's run.json is
# embedded in F0's run.json instead.
SKIP_SIDE_FILES = {RUN_RECORD, *LEGACY_RECORD_FILES}
FREQUENCY_SCORE_FORMULA = "target_dispatch_freq - control_weight * control_dispatch_freq"


@dataclass(frozen=True)
class ExpansionRecipe:
    num_new_experts: int
    control_weight: float
    seed: int
    epsilon: float
    source_expert_scale: float


@dataclass(frozen=True)
class CheckpointDiscovery:
    weight_map: dict[str, str]
    experts_by_layer: dict[int, list[int]]
    expert_layout: str


@dataclass(frozen=True)
class ExpansionPlan:
    old_num_experts: int
    num_new_experts: int
    new_num_experts: int
    layers: list[int]
    source_by_layer: dict[int, list[int]]
    source_scores_by_layer: dict[int, list[float]]
    discovery: CheckpointDiscovery


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Expansion recipe YAML.")
    parser.add_argument("--base", required=True, type=Path, help="Input standard checkpoint (B).")
    parser.add_argument("--target-frequency-file", required=True, type=Path, help="Target dispatch_freq.csv.")
    parser.add_argument("--control-frequency-file", required=True, type=Path, help="Control dispatch_freq.csv.")
    parser.add_argument("--output", required=True, type=Path, help="New output directory for F0.")
    return parser


def load_recipe(path: Path) -> tuple[dict[str, Any], ExpansionRecipe]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    recipe = ExpansionRecipe(
        num_new_experts=int(config["num_new_experts"]),
        control_weight=float(config["control_weight"]),
        seed=int(config["seed"]),
        epsilon=float(config["epsilon"]),
        source_expert_scale=float(config["source_expert_scale"]),
    )
    if recipe.num_new_experts <= 0:
        raise ValueError("num_new_experts must be positive.")
    if not math.isfinite(recipe.epsilon) or recipe.epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative.")
    if not math.isfinite(recipe.source_expert_scale) or recipe.source_expert_scale < 0:
        raise ValueError("source_expert_scale must be finite and non-negative.")
    if not math.isfinite(recipe.control_weight) or recipe.control_weight < 0:
        raise ValueError("control_weight must be finite and non-negative.")
    return config, recipe


def scan_moe_weight_keys(
    weight_map: dict[str, str],
) -> tuple[dict[int, set[int]], dict[int, set[str]]]:
    split_experts_by_layer: dict[int, set[int]] = defaultdict(set)
    packed_projections_by_layer: dict[int, set[str]] = defaultdict(set)
    for key in weight_map:
        match = SPLIT_EXPERT_KEY_RE.match(key)
        if match:
            split_experts_by_layer[int(match.group("layer"))].add(int(match.group("expert")))
            continue
        packed_match = PACKED_EXPERT_KEY_RE.match(key)
        if packed_match:
            packed_projections_by_layer[int(packed_match.group("layer"))].add(packed_match.group("proj"))
    return split_experts_by_layer, packed_projections_by_layer


def discovery_from_packed_layout(
    *,
    weight_map: dict[str, str],
    packed_projections_by_layer: dict[int, set[str]],
    split_experts_by_layer: dict[int, set[int]],
    old_num_experts: int,
) -> CheckpointDiscovery:
    missing = {
        layer: sorted(set(PACKED_EXPERT_PROJECTIONS) - projections)
        for layer, projections in packed_projections_by_layer.items()
        if projections != set(PACKED_EXPERT_PROJECTIONS)
    }
    if missing:
        raise ValueError(f"Packed MoE layers are missing projections: {missing}")
    if split_experts_by_layer:
        raise ValueError("Found both packed and split MoE expert layouts in decoder layers; this is unsupported.")
    experts_by_layer = {
        layer: list(range(old_num_experts))
        for layer in sorted(packed_projections_by_layer)
    }
    return CheckpointDiscovery(
        weight_map=weight_map,
        experts_by_layer=experts_by_layer,
        expert_layout="packed",
    )


def discovery_from_split_layout(
    *,
    weight_map: dict[str, str],
    split_experts_by_layer: dict[int, set[int]],
) -> CheckpointDiscovery:
    if not split_experts_by_layer:
        raise ValueError("No supported MoE expert keys were found in the checkpoint.")

    sorted_experts_by_layer = {layer: sorted(experts) for layer, experts in split_experts_by_layer.items()}
    expert_counts = {len(experts) for experts in sorted_experts_by_layer.values()}
    if len(expert_counts) != 1:
        raise ValueError(f"Expected all MoE layers to have the same expert count, got {sorted(expert_counts)}.")
    return CheckpointDiscovery(
        weight_map=weight_map,
        experts_by_layer=sorted_experts_by_layer,
        expert_layout="split",
    )


def discover_checkpoint(
    checkpoint: Path,
    *,
    old_num_experts: int,
) -> CheckpointDiscovery:
    index = read_weight_index(checkpoint)
    weight_map: dict[str, str] = dict(index["weight_map"])
    split_experts_by_layer, packed_projections_by_layer = scan_moe_weight_keys(weight_map)

    if packed_projections_by_layer:
        return discovery_from_packed_layout(
            weight_map=weight_map,
            packed_projections_by_layer=packed_projections_by_layer,
            split_experts_by_layer=split_experts_by_layer,
            old_num_experts=old_num_experts,
        )
    return discovery_from_split_layout(
        weight_map=weight_map,
        split_experts_by_layer=split_experts_by_layer,
    )


def load_frequency_values(
    frequency_file: Path,
    layers: list[int],
    num_experts: int,
) -> dict[int, dict[int, float]]:
    expected_keys = {(layer, expert) for layer in layers for expert in range(num_experts)}
    values: dict[int, dict[int, float]] = defaultdict(dict)
    seen_keys: set[tuple[int, int]] = set()
    with frequency_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "expert", "dispatch_freq"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Frequency CSV {frequency_file} is missing required columns: {sorted(missing)}")
        for row in reader:
            layer = int(row["layer"])
            expert = int(row["expert"])
            dispatch_freq = float(row["dispatch_freq"])
            key = (layer, expert)
            if key in seen_keys:
                raise ValueError(f"Frequency CSV {frequency_file} has duplicate row for layer={layer}, expert={expert}.")
            if not math.isfinite(dispatch_freq):
                raise ValueError(
                    f"Frequency CSV {frequency_file} has non-finite dispatch_freq "
                    f"for layer={layer}, expert={expert}."
                )
            seen_keys.add(key)
            values[layer][expert] = dispatch_freq

    missing_keys = expected_keys - seen_keys
    unexpected_keys = seen_keys - expected_keys
    if missing_keys or unexpected_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing={sorted(missing_keys)[:8]} (total={len(missing_keys)})")
        if unexpected_keys:
            details.append(f"unexpected={sorted(unexpected_keys)[:8]} (total={len(unexpected_keys)})")
        raise ValueError(f"Frequency CSV {frequency_file} layer/expert rows do not match the checkpoint: {'; '.join(details)}")
    return {layer: dict(values[layer]) for layer in layers}


def select_source_experts(
    *,
    layers: list[int],
    num_experts: int,
    num_new_experts: int,
    target_frequency_file: Path,
    control_frequency_file: Path,
    control_weight: float,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Top-ranked experts per layer (repeated cyclically if more clones than experts)."""
    target_values = load_frequency_values(target_frequency_file, layers, num_experts)
    control_values = load_frequency_values(control_frequency_file, layers, num_experts)

    sources: dict[int, list[int]] = {}
    scores: dict[int, list[float]] = {}
    for layer in layers:
        rows = [
            (expert, target_values[layer][expert] - control_weight * control_values[layer][expert])
            for expert in range(num_experts)
        ]
        rows.sort(key=lambda item: (-item[1], item[0]))
        sources[layer] = repeat_to_length([expert for expert, _score in rows], num_new_experts)
        scores[layer] = repeat_to_length([score for _expert, score in rows], num_new_experts)
    return sources, scores


def repeat_to_length(values: list, length: int) -> list:
    return [values[idx % len(values)] for idx in range(length)]


def copy_checkpoint_side_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.name in SKIP_SIDE_FILES:
            continue
        if path.name.startswith("model-") and path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        if path.is_file() and (path.suffix in COPY_FILE_SUFFIXES or path.name.startswith("tokenizer")):
            shutil.copy2(path, output_dir / path.name)


def add_noise(tensor: torch.Tensor, epsilon: float, generator: torch.Generator) -> torch.Tensor:
    clone = tensor.detach().clone()
    if epsilon == 0:
        return clone
    noisy = clone.to(dtype=torch.float32)
    noisy = noisy + torch.randn(noisy.shape, generator=generator, dtype=torch.float32) * epsilon
    return noisy.to(dtype=clone.dtype)


def shard_metadata(input_dir: Path, shard_name: str) -> dict[str, str] | None:
    with safe_open(input_dir / shard_name, framework="pt", device="cpu") as f:
        return f.metadata()


def assign_new_expert_keys(
    weight_map: dict[str, str],
    source_by_layer: dict[int, list[int]],
    old_num_experts: int,
) -> dict[str, str]:
    updated = dict(weight_map)
    for source_key, new_key in iter_new_split_expert_key_pairs(weight_map, source_by_layer, old_num_experts):
        updated[new_key] = weight_map[source_key]
    return updated


def iter_new_split_expert_key_pairs(
    weight_map: dict[str, str],
    source_by_layer: dict[int, list[int]],
    old_num_experts: int,
) -> Iterator[tuple[str, str]]:
    for layer, sources in source_by_layer.items():
        for offset, source_expert in enumerate(sources):
            new_expert = old_num_experts + offset
            for proj in SPLIT_EXPERT_PROJECTIONS:
                source_key = split_expert_key(weight_map, layer, source_expert, proj)
                new_key = split_expert_key(weight_map, layer, new_expert, proj, source_key=source_key)
                yield source_key, new_key


def build_shard_to_new_experts(
    original_weight_map: dict[str, str],
    source_by_layer: dict[int, list[int]],
    old_num_experts: int,
) -> dict[str, list[tuple[str, str]]]:
    shard_to_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source_key, new_key in iter_new_split_expert_key_pairs(
        original_weight_map,
        source_by_layer,
        old_num_experts,
    ):
        shard_to_pairs[original_weight_map[source_key]].append((source_key, new_key))
    return shard_to_pairs


def _expand_rows_for_layer_if_needed(
    key: str,
    tensor: torch.Tensor,
    source_by_layer: dict[int, list[int]],
    epsilon: float,
    generator: torch.Generator,
    *,
    key_pattern: re.Pattern[str],
    source_expert_scale: float = 1.0,
) -> torch.Tensor:
    match = key_pattern.match(key)
    if not match:
        return tensor
    layer = int(match.group("layer"))
    sources = source_by_layer.get(layer)
    if not sources:
        return tensor

    new_rows = [add_noise(tensor[source_expert], epsilon, generator) for source_expert in sources]
    original_rows = tensor
    if source_expert_scale != 1.0:
        original_rows = tensor.clone()
        unique_sources = sorted(set(sources))
        original_rows[unique_sources] = original_rows[unique_sources] * source_expert_scale
    return torch.cat([original_rows, torch.stack(new_rows, dim=0)], dim=0)


def expand_gate_if_needed(
    key: str,
    tensor: torch.Tensor,
    source_by_layer: dict[int, list[int]],
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return _expand_rows_for_layer_if_needed(
        key,
        tensor,
        source_by_layer,
        epsilon,
        generator,
        key_pattern=GATE_KEY_RE,
    )


def expand_packed_expert_if_needed(
    key: str,
    tensor: torch.Tensor,
    source_by_layer: dict[int, list[int]],
    epsilon: float,
    generator: torch.Generator,
    source_expert_scale: float = 1.0,
) -> torch.Tensor:
    return _expand_rows_for_layer_if_needed(
        key,
        tensor,
        source_by_layer,
        epsilon,
        generator,
        key_pattern=PACKED_EXPERT_KEY_RE,
        source_expert_scale=source_expert_scale,
    )


def validate_checkpoint_paths(checkpoint: Path, output_dir: Path) -> None:
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"Input checkpoint directory does not exist: {checkpoint}")
    if output_dir == checkpoint or output_dir.is_relative_to(checkpoint) or checkpoint.is_relative_to(output_dir):
        raise ValueError("Output and input checkpoint directories must be disjoint.")


def load_config_and_discovery(checkpoint: Path) -> tuple[dict[str, Any], int, CheckpointDiscovery]:
    config = read_json(checkpoint / "config.json")
    old_num_experts = config_num_experts(config)
    if old_num_experts <= 0:
        raise ValueError("config must define num_experts either at top level or under text_config.")
    discovery = discover_checkpoint(
        checkpoint,
        old_num_experts=old_num_experts,
    )
    discovered_num_experts = len(next(iter(discovery.experts_by_layer.values())))
    if old_num_experts != discovered_num_experts:
        raise ValueError(
            f"config num_experts={old_num_experts} does not match checkpoint experts={discovered_num_experts}."
        )
    validate_checkpoint_layout(checkpoint, config, discovery.weight_map, old_num_experts, list(discovery.experts_by_layer))
    return config, old_num_experts, discovery


def build_expansion_plan(
    recipe: ExpansionRecipe,
    *,
    target_frequency_file: Path,
    control_frequency_file: Path,
    old_num_experts: int,
    discovery: CheckpointDiscovery,
) -> ExpansionPlan:
    layers = sorted(discovery.experts_by_layer)
    source_by_layer, source_scores_by_layer = select_source_experts(
        layers=layers,
        num_experts=old_num_experts,
        num_new_experts=recipe.num_new_experts,
        target_frequency_file=target_frequency_file,
        control_frequency_file=control_frequency_file,
        control_weight=recipe.control_weight,
    )
    return ExpansionPlan(
        old_num_experts=old_num_experts,
        num_new_experts=recipe.num_new_experts,
        new_num_experts=old_num_experts + recipe.num_new_experts,
        layers=layers,
        source_by_layer=source_by_layer,
        source_scores_by_layer=source_scores_by_layer,
        discovery=discovery,
    )


def build_weight_map_and_shard_plan(
    plan: ExpansionPlan,
) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    weight_map = plan.discovery.weight_map
    if plan.discovery.expert_layout == "split":
        updated_weight_map = assign_new_expert_keys(weight_map, plan.source_by_layer, plan.old_num_experts)
        shard_to_new_experts = build_shard_to_new_experts(weight_map, plan.source_by_layer, plan.old_num_experts)
    else:
        updated_weight_map = dict(weight_map)
        shard_to_new_experts = {}
    return updated_weight_map, shard_to_new_experts


def expand_shard_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    shard_name: str,
    source_by_layer: dict[int, list[int]],
    shard_to_new_experts: dict[str, list[tuple[str, str]]],
    epsilon: float,
    noise_generator: torch.Generator,
    source_expert_scale: float = 1.0,
) -> None:
    for key, tensor in list(tensors.items()):
        if match := EXPERT_SCALE_KEY_RE.match(key):
            sources = source_by_layer[int(match["layer"])]
            tensors[key] = torch.cat((tensor, tensor[sources]), dim=0)
            continue
        expanded = expand_gate_if_needed(
            key=key,
            tensor=tensor,
            source_by_layer=source_by_layer,
            epsilon=epsilon,
            generator=noise_generator,
        )
        tensors[key] = expand_packed_expert_if_needed(
            key=key,
            tensor=expanded,
            source_by_layer=source_by_layer,
            epsilon=epsilon,
            generator=noise_generator,
            source_expert_scale=source_expert_scale,
        )

    for source_key, new_key in shard_to_new_experts.get(shard_name, []):
        tensors[new_key] = add_noise(tensors[source_key], epsilon, noise_generator)
    if source_expert_scale != 1.0:
        source_keys = {
            source_key
            for source_key, _new_key in shard_to_new_experts.get(shard_name, [])
        }
        for source_key in source_keys:
            tensors[source_key] = tensors[source_key].mul(source_expert_scale)


def write_expanded_shards(
    *,
    checkpoint: Path,
    output_dir: Path,
    plan: ExpansionPlan,
    shard_to_new_experts: dict[str, list[tuple[str, str]]],
    epsilon: float,
    seed: int,
    source_expert_scale: float = 1.0,
) -> int:
    weight_map = plan.discovery.weight_map
    shards = sorted(set(weight_map.values()))
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(seed)

    total_size = 0
    for shard_idx, shard_name in enumerate(shards, start=1):
        print(f"Writing shard {shard_idx}/{len(shards)}: {shard_name}")
        tensors = load_shard(checkpoint, shard_name)
        expand_shard_tensors(
            tensors,
            shard_name=shard_name,
            source_by_layer=plan.source_by_layer,
            shard_to_new_experts=shard_to_new_experts,
            epsilon=epsilon,
            noise_generator=noise_generator,
            source_expert_scale=source_expert_scale,
        )

        for tensor in tensors.values():
            total_size += tensor_nbytes(tensor)

        metadata = shard_metadata(checkpoint, shard_name)
        save_file(tensors, output_dir / shard_name, metadata=metadata)
    return total_size


def build_expansion_manifest(
    *,
    checkpoint: Path,
    output_dir: Path,
    target_frequency_file: Path,
    control_frequency_file: Path,
    recipe: ExpansionRecipe,
    plan: ExpansionPlan,
) -> dict[str, Any]:
    """Structural description of the removable bank, read by loaders and excision.

    The selection fields are kept for parity with existing F0 checkpoints.
    """
    layers_manifest = [
        {
            "layer": layer,
            "new_experts": list(range(plan.old_num_experts, plan.new_num_experts)),
            "source_experts": plan.source_by_layer[layer],
            "source_scores": plan.source_scores_by_layer[layer],
        }
        for layer in plan.layers
    ]
    return {
        **checkpoint_metadata(read_json(checkpoint / "config.json")),
        "source_checkpoint": str(checkpoint),
        "output_checkpoint": str(output_dir),
        "strategy": "frequency",
        "frequency_file": str(target_frequency_file),
        "control_frequency_file": str(control_frequency_file),
        "control_weight": recipe.control_weight,
        "frequency_score_formula": FREQUENCY_SCORE_FORMULA,
        "seed": recipe.seed,
        "epsilon": recipe.epsilon,
        "source_expert_scale": recipe.source_expert_scale,
        "old_num_experts": plan.old_num_experts,
        "num_new_experts": plan.num_new_experts,
        "new_num_experts": plan.new_num_experts,
        "expert_layout": plan.discovery.expert_layout,
        "layers": layers_manifest,
    }


def expand_checkpoint(
    *,
    recipe: ExpansionRecipe,
    checkpoint: Path,
    target_frequency_file: Path,
    control_frequency_file: Path,
    output_dir: Path,
) -> ExpansionPlan:
    validate_checkpoint_paths(checkpoint, output_dir)
    config, old_num_experts, discovery = load_config_and_discovery(checkpoint)
    plan = build_expansion_plan(
        recipe,
        target_frequency_file=target_frequency_file,
        control_frequency_file=control_frequency_file,
        old_num_experts=old_num_experts,
        discovery=discovery,
    )
    updated_weight_map, shard_to_new_experts = build_weight_map_and_shard_plan(plan)

    prepare_output_dir(output_dir)
    copy_checkpoint_side_files(checkpoint, output_dir)
    total_size = write_expanded_shards(
        checkpoint=checkpoint,
        output_dir=output_dir,
        plan=plan,
        shard_to_new_experts=shard_to_new_experts,
        epsilon=recipe.epsilon,
        seed=recipe.seed,
        source_expert_scale=recipe.source_expert_scale,
    )

    updated_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(updated_weight_map.items())),
    }
    write_json_atomic(output_dir / "model.safetensors.index.json", updated_index)

    set_config_num_experts(config, plan.new_num_experts)
    write_json_atomic(output_dir / "config.json", config)

    manifest = build_expansion_manifest(
        checkpoint=checkpoint,
        output_dir=output_dir,
        target_frequency_file=target_frequency_file,
        control_frequency_file=control_frequency_file,
        recipe=recipe,
        plan=plan,
    )
    write_json_atomic(output_dir / EXPANSION_MANIFEST, manifest)
    return plan


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path, checkpoint, target_csv, control_csv, output_dir = (
        path.resolve()
        for path in (args.config, args.base, args.target_frequency_file, args.control_frequency_file, args.output)
    )
    config, recipe = load_recipe(config_path)
    plan = expand_checkpoint(
        recipe=recipe,
        checkpoint=checkpoint,
        target_frequency_file=target_csv,
        control_frequency_file=control_csv,
        output_dir=output_dir,
    )
    write_run_record(
        output_dir,
        step="expand_experts",
        inputs={
            "config": config_path,
            "base": checkpoint,
            "target_frequency_file": target_csv,
            "control_frequency_file": control_csv,
        },
        config=config,
        results={
            "expert_layout": plan.discovery.expert_layout,
            "old_num_experts": plan.old_num_experts,
            "new_num_experts": plan.new_num_experts,
            "layers": plan.layers,
            "source_experts_by_layer": {str(layer): plan.source_by_layer[layer] for layer in plan.layers},
        },
    )
    print(f"Expanded checkpoint written to {output_dir}")


if __name__ == "__main__":
    main()
