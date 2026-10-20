#!/usr/bin/env python3
"""Physically remove the tail-contiguous added expert bank from an expanded checkpoint.

Supports split and packed expert tensors. With --random-original-experts-seed
the same number of original experts is removed instead in every layer and the
added bank is kept (the random-excision selectivity control).
"""
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from common.io_utils import load_shard, prepare_output_dir, read_json, tensor_nbytes, write_json_atomic
from common.manifest_names import EXPANSION_MANIFEST, LEGACY_RECORD_FILES, RUN_RECORD
from common.run_record import write_run_record
from moe.runtime import load_expanded_moe_manifest_metadata
# Shared text-decoder layout contract; excludes vision and MTP.
from moe.model_adapters import (
    EXPERT_SCALE_KEY_RE,
    GATE_KEY_RE,
    PACKED_EXPERT_KEY_RE,
    PACKED_EXPERT_PROJECTIONS,
    SPLIT_EXPERT_KEY_RE,
    SPLIT_EXPERT_PROJECTIONS,
    read_weight_index,
    set_config_num_experts,
    validate_checkpoint_layout,
)

write_sorted_json = partial(write_json_atomic, sort_keys=True)

INDEX_NAME = "model.safetensors.index.json"
# The excised checkpoint is standard: it drops the expansion manifest. Records
# of the input are embedded in the output's run.json instead of copied.
SKIP_SIDE_FILES = {INDEX_NAME, EXPANSION_MANIFEST, RUN_RECORD, "README.md", "README.zh-CN.md",
                   "args.json", "logging.jsonl", "latest_checkpointed_iteration.txt",
                   "reload_probe.pt", *LEGACY_RECORD_FILES}
RANDOM_SAMPLING = "Python random.Random(seed).sample, ascending layer order, without replacement"


@dataclass(frozen=True)
class AblationSpec:
    layout: str
    old_num_experts: int
    new_num_experts: int
    harmful_experts: tuple[int, ...]
    layers: tuple[int, ...]
    manifest_layout: str | None = None
    removed_experts_by_layer: tuple[tuple[int, tuple[int, ...]], ...] = ()
    random_original_experts_seed: int | None = None

    @property
    def num_harmful_experts(self) -> int:
        return self.new_num_experts - self.old_num_experts

    def removed_for_layer(self, layer: int) -> tuple[int, ...]:
        if self.removed_experts_by_layer:
            return dict(self.removed_experts_by_layer)[layer]
        return self.harmful_experts

    def kept_for_layer(self, layer: int) -> tuple[int, ...]:
        removed = set(self.removed_for_layer(layer))
        return tuple(e for e in range(self.new_num_experts) if e not in removed)


def random_original_excision_spec(spec: AblationSpec, seed: int) -> AblationSpec:
    """Same layers/count as the designated bank, retaining every added expert."""
    if type(seed) is not int or seed < 0:
        raise ValueError('random-original-experts seed must be a nonnegative integer')
    if spec.num_harmful_experts > spec.old_num_experts:
        raise ValueError('Cannot remove more original experts than exist')
    generator = random.Random(seed)
    removed = tuple((layer, tuple(sorted(generator.sample(range(spec.old_num_experts), spec.num_harmful_experts)))) for layer in sorted(spec.layers))
    return replace(spec, removed_experts_by_layer=removed, random_original_experts_seed=seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Input expanded checkpoint directory.")
    parser.add_argument("--output", required=True, type=Path, help="New output checkpoint directory.")
    parser.add_argument(
        "--random-original-experts-seed",
        type=int,
        help="Control: remove K original experts in each selected layer, preserving the added bank.",
    )
    return parser


def validate_paths(model_dir: Path, output_dir: Path) -> None:
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Input model directory does not exist: {model_dir}")
    if output_dir == model_dir or output_dir.is_relative_to(model_dir) or model_dir.is_relative_to(output_dir):
        raise ValueError("Output and input checkpoint directories must be disjoint.")


def load_ablation_spec(model_dir: Path) -> AblationSpec:
    manifest_path = model_dir / EXPANSION_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing expansion manifest: {manifest_path}")
    manifest = read_json(manifest_path)

    old_num_experts, expected_harmful, new_num_experts = load_expanded_moe_manifest_metadata(model_dir, require_expert_list=False)
    layout = manifest["expert_layout"]
    # Runtime validates the same nonempty list and identical contiguous tail.
    # Layer-id defaulting remains offset-based, with duplicate/coverage checking
    # in validate_checkpoint_layout, as before this shared parser.
    layer_ids = [int(layer.get("layer", offset)) for offset, layer in enumerate(manifest["layers"])]

    return AblationSpec(
        layout=layout,
        old_num_experts=old_num_experts,
        new_num_experts=new_num_experts,
        harmful_experts=expected_harmful,
        layers=tuple(layer_ids),
        manifest_layout=layout,
    )


def scan_layout(
    weight_map: dict[str, str],
) -> tuple[dict[int, dict[int, set[str]]], dict[int, set[str]]]:
    split_experts_by_layer: dict[int, dict[int, set[str]]] = {}
    packed_projections_by_layer: dict[int, set[str]] = {}
    for key in weight_map:
        if split_match := SPLIT_EXPERT_KEY_RE.match(key):
            layer = int(split_match.group("layer"))
            expert = int(split_match.group("expert"))
            projection = split_match.group("proj")
            split_experts_by_layer.setdefault(layer, {}).setdefault(
                expert,
                set(),
            ).add(projection)
            continue
        if packed_match := PACKED_EXPERT_KEY_RE.match(key):
            layer = int(packed_match.group("layer"))
            packed_projections_by_layer.setdefault(layer, set()).add(packed_match.group("proj"))
    return split_experts_by_layer, packed_projections_by_layer


def require_gate_keys(weight_map: dict[str, str], layers: tuple[int, ...]) -> None:
    present_layers = {int(match.group("layer")) for key in weight_map if (match := GATE_KEY_RE.match(key))}
    missing = sorted(set(layers) - present_layers)
    if missing:
        raise KeyError(f"Missing router gate weights for layers: {missing}")


def validate_split_layout(
    split_experts_by_layer: dict[int, dict[int, set[str]]],
    spec: AblationSpec,
) -> None:
    if set(split_experts_by_layer) != set(spec.layers):
        raise ValueError(
            "Split expert layers do not match manifest layers: "
            f"checkpoint={sorted(split_experts_by_layer)}, manifest={sorted(spec.layers)}"
        )
    expected_experts = set(range(spec.new_num_experts))
    bad_layers = {
        layer: {
            "missing": sorted(expected_experts - set(experts)),
            "extra": sorted(set(experts) - expected_experts),
        }
        for layer, experts in split_experts_by_layer.items()
        if set(experts) != expected_experts
    }
    if bad_layers:
        raise ValueError(f"Split expert ids must be exactly 0..{spec.new_num_experts - 1}: {bad_layers}")
    incomplete_projections = {
        layer: {
            expert: {
                "missing": sorted(set(SPLIT_EXPERT_PROJECTIONS) - projections),
                "extra": sorted(projections - set(SPLIT_EXPERT_PROJECTIONS)),
            }
            for expert, projections in experts.items()
            if projections != set(SPLIT_EXPERT_PROJECTIONS)
        }
        for layer, experts in split_experts_by_layer.items()
    }
    incomplete_projections = {
        layer: experts
        for layer, experts in incomplete_projections.items()
        if experts
    }
    if incomplete_projections:
        raise ValueError(
            "Split experts are missing required projections: "
            f"{incomplete_projections}"
        )


def validate_packed_layout(packed_projections_by_layer: dict[int, set[str]], spec: AblationSpec) -> None:
    if set(packed_projections_by_layer) != set(spec.layers):
        raise ValueError(
            "Packed expert layers do not match manifest layers: "
            f"checkpoint={sorted(packed_projections_by_layer)}, manifest={sorted(spec.layers)}"
        )
    missing = {
        layer: sorted(set(PACKED_EXPERT_PROJECTIONS) - projections)
        for layer, projections in packed_projections_by_layer.items()
        if projections != set(PACKED_EXPERT_PROJECTIONS)
    }
    if missing:
        raise ValueError(f"Packed expert layers are missing projections: {missing}")


def validate_layout(weight_map: dict[str, str], spec: AblationSpec) -> AblationSpec:
    split_experts_by_layer, packed_projections_by_layer = scan_layout(weight_map)
    if split_experts_by_layer and packed_projections_by_layer:
        raise ValueError("Found both split and packed MoE expert keys in one checkpoint; this is unsupported.")

    if split_experts_by_layer:
        validate_split_layout(split_experts_by_layer, spec)
        actual_layout = "split"
    elif packed_projections_by_layer:
        validate_packed_layout(packed_projections_by_layer, spec)
        actual_layout = "packed"
    else:
        raise ValueError("No supported MoE expert tensors found in checkpoint.")

    require_gate_keys(weight_map, spec.layers)
    manifest_layout = spec.manifest_layout or spec.layout
    if actual_layout != manifest_layout:
        # Intern-S2's Hugging Face save_pretrained override materializes the
        # logically packed expert tensors as one key per expert. The checkpoint
        # remains packed-compatible when loaded, but physical ablation must
        # follow the tensor layout that is actually present on disk.
        if manifest_layout == "packed" and actual_layout == "split":
            print(
                "Manifest uses logical packed layout, while the saved Hugging Face "
                "checkpoint materializes split expert tensors; ablating the split tensors."
            )
            return replace(spec, layout=actual_layout, manifest_layout=manifest_layout)
        raise ValueError(
            f"Manifest expert_layout={manifest_layout!r} does not match "
            f"checkpoint tensor layout={actual_layout!r}."
        )
    return replace(spec, layout=actual_layout, manifest_layout=manifest_layout)


def shard_names(weight_map: dict[str, str]) -> list[str]:
    return sorted(set(weight_map.values()))


def trim_expert_rows(key: str, tensor: torch.Tensor, spec: AblationSpec) -> torch.Tensor:
    if tensor.ndim == 0 or tensor.shape[0] != spec.new_num_experts:
        raise ValueError(
            f"{key} has shape {tuple(tensor.shape)}, expected first dimension {spec.new_num_experts}."
        )
    if not spec.removed_experts_by_layer:
        return tensor[: spec.old_num_experts].contiguous()
    match = GATE_KEY_RE.match(key) or EXPERT_SCALE_KEY_RE.match(key) or PACKED_EXPERT_KEY_RE.match(key)
    if match is None:
        raise ValueError(f'Unsupported expert-row key: {key}')
    kept = spec.kept_for_layer(int(match.group('layer')))
    return tensor.index_select(0, torch.tensor(kept, device=tensor.device)).contiguous()


def transformed_key(key: str, spec: AblationSpec) -> str | None:
    match = SPLIT_EXPERT_KEY_RE.match(key)
    if not match:
        return key
    layer, expert = int(match.group('layer')), int(match.group('expert'))
    if expert in spec.removed_for_layer(layer):
        return None
    if not spec.removed_experts_by_layer:
        return key
    new_expert = spec.kept_for_layer(layer).index(expert)
    start, end = match.span('expert')
    return key[:start] + str(new_expert) + key[end:]


def transform_tensor(key: str, tensor: torch.Tensor, spec: AblationSpec) -> torch.Tensor | None:
    if GATE_KEY_RE.match(key) or EXPERT_SCALE_KEY_RE.match(key):
        return trim_expert_rows(key, tensor, spec)

    if spec.layout == "packed":
        if PACKED_EXPERT_KEY_RE.match(key):
            return trim_expert_rows(key, tensor, spec)
        return tensor

    split_match = SPLIT_EXPERT_KEY_RE.match(key)
    if split_match and int(split_match.group("expert")) in spec.removed_for_layer(int(split_match.group('layer'))):
        return None
    return tensor


def output_weight_map(weight_map: dict[str, str], spec: AblationSpec) -> dict[str, str]:
    if spec.layout == "packed":
        return dict(weight_map)
    return {
        new_key: shard_name
        for key, shard_name in weight_map.items()
        if (new_key := transformed_key(key, spec)) is not None
    }


def write_ablated_shards(
    *,
    model_dir: Path,
    output_dir: Path,
    weight_map: dict[str, str],
    spec: AblationSpec,
) -> int:
    total_size = 0
    for shard_idx, shard_name in enumerate(shard_names(weight_map), start=1):
        print(f"Writing shard {shard_idx}: {shard_name}")
        tensors, metadata = load_shard(model_dir, shard_name, with_metadata=True)
        transformed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            new_tensor = transform_tensor(key, tensor, spec)
            if new_tensor is None:
                continue
            new_key = transformed_key(key, spec)
            assert new_key is not None
            transformed[new_key] = new_tensor
            total_size += tensor_nbytes(new_tensor)
        if transformed:
            save_file(transformed, output_dir / shard_name, metadata=metadata)
    return total_size


def copy_side_files(model_dir: Path, output_dir: Path) -> None:
    # Only top-level files (config, tokenizer, processor, remote code) make the
    # output loadable. Subdirectories of a node hold its own records, evaluations
    # and training state (eval/, online_eval/, logs/, ...): copying them would
    # make the excised node look already evaluated.
    for path in model_dir.iterdir():
        if (
            not path.is_file()
            or path.name in SKIP_SIDE_FILES
            or path.suffix == ".safetensors"
        ):
            continue
        shutil.copy2(path, output_dir / path.name)


def build_ablated_config(model_dir: Path, spec: AblationSpec) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Input checkpoint is missing config.json: {config_path}")
    config = read_json(config_path)
    set_config_num_experts(config, spec.old_num_experts, old_value=spec.new_num_experts)
    return config


def write_index(output_dir: Path, index: dict[str, Any], weight_map: dict[str, str], total_size: int) -> None:
    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = total_size
    write_sorted_json(
        output_dir / INDEX_NAME,
        {
            **index,
            "metadata": metadata,
            "weight_map": dict(sorted(weight_map.items())),
        },
    )


def excision_results(spec: AblationSpec) -> dict[str, Any]:
    results: dict[str, Any] = {
        "tensor_layout": spec.layout,
        "manifest_layout": spec.manifest_layout or spec.layout,
        "source_num_experts": spec.new_num_experts,
        "output_num_experts": spec.old_num_experts,
        "num_removed_experts": spec.num_harmful_experts,
        "layers": list(spec.layers),
    }
    if not spec.removed_experts_by_layer:
        results.update(
            operation="added_bank_excision",
            removed_experts=list(spec.harmful_experts),
            kept_experts=list(range(spec.old_num_experts)),
        )
        return results
    results.update(
        operation="random_original_excision",
        random_seed=spec.random_original_experts_seed,
        sampling=RANDOM_SAMPLING,
        removed_experts_by_layer={str(layer): list(spec.removed_for_layer(layer)) for layer in spec.layers},
        kept_experts_by_layer={str(layer): list(spec.kept_for_layer(layer)) for layer in spec.layers},
        added_experts_preserved=list(spec.harmful_experts),
    )
    return results


def excise_checkpoint(
    *,
    model_dir: Path,
    output_dir: Path,
    random_original_experts_seed: int | None = None,
) -> AblationSpec:
    validate_paths(model_dir, output_dir)
    spec = load_ablation_spec(model_dir)
    index = read_weight_index(model_dir)
    weight_map = dict(index["weight_map"])
    spec = validate_layout(weight_map, spec)
    if random_original_experts_seed is not None:
        spec = random_original_excision_spec(spec, random_original_experts_seed)
    validate_checkpoint_layout(model_dir, read_json(model_dir / "config.json"), weight_map, spec.new_num_experts, list(spec.layers))
    config = build_ablated_config(model_dir, spec)

    prepare_output_dir(output_dir)
    copy_side_files(model_dir, output_dir)
    new_weight_map = output_weight_map(weight_map, spec)
    total_size = write_ablated_shards(
        model_dir=model_dir,
        output_dir=output_dir,
        weight_map=weight_map,
        spec=spec,
    )
    write_index(output_dir, index, new_weight_map, total_size)
    write_sorted_json(output_dir / "config.json", config)
    if spec.removed_experts_by_layer:
        validate_checkpoint_layout(output_dir, config, new_weight_map, spec.old_num_experts, list(spec.layers))
    return spec


def main() -> None:
    args = build_arg_parser().parse_args()
    model_dir = args.model.resolve()
    output_dir = args.output.resolve()
    spec = excise_checkpoint(
        model_dir=model_dir,
        output_dir=output_dir,
        random_original_experts_seed=args.random_original_experts_seed,
    )
    write_run_record(
        output_dir,
        step="excise_experts",
        inputs={"model": model_dir},
        args={"random_original_experts_seed": args.random_original_experts_seed},
        results=excision_results(spec),
    )
    print(f"Wrote excised checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
