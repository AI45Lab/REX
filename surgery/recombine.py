#!/usr/bin/env python3
"""Rebuild F2 from A2's retained parameters and the removed bank of its F1.

F1 must be the expanded checkpoint whose excision produced A2's parent A1
(pipelines/f1_to_f2.sh guarantees this); it is not inferred here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.checkpoint_io import copy_runtime_side_files
from common.io_utils import ShardedSafetensorsWriter, write_json_atomic
from common.manifest_names import EXPANSION_MANIFEST
from common.run_record import write_run_record
from moe.model_adapters import (
    EXPERT_SCALE_KEY_RE,
    GATE_KEY_RE,
    PACKED_EXPERT_KEY_RE,
    SPLIT_EXPERT_KEY_RE,
    SPLIT_EXPERT_PROJECTIONS,
    TensorReader,
    checkpoint_metadata,
    expert_stem,
    is_expert_bank_key,
    set_config_num_experts,
    split_key_from_stem,
    validate_checkpoint_layout,
)
from surgery.excise import load_ablation_spec, validate_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f1", required=True, type=Path, help="Expanded checkpoint providing the removed bank.")
    parser.add_argument("--a2", required=True, type=Path, help="Standard checkpoint providing retained parameters.")
    parser.add_argument("--output", required=True, type=Path, help="New output directory for F2.")
    return parser.parse_args()


def recombine(f1_dir: Path, a2_dir: Path, output_dir: Path) -> dict:
    original, ablated, output = (p.resolve() for p in (f1_dir, a2_dir, output_dir))
    for source in (original, ablated):
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError("Recombination input and output paths must be disjoint.")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if (ablated / EXPANSION_MANIFEST).exists():
        raise ValueError("A2 must be a standard checkpoint.")

    f1, a2 = TensorReader(original), TensorReader(ablated)
    spec = validate_layout(f1.weight_map, load_ablation_spec(original))
    old, new = spec.old_num_experts, spec.new_num_experts
    config = json.loads((ablated / "config.json").read_text())
    full_config = json.loads((original / "config.json").read_text())
    if checkpoint_metadata(config) != checkpoint_metadata(full_config):
        raise ValueError("F1/A2 model architectures disagree.")
    validate_checkpoint_layout(
        original, full_config, f1.weight_map, new, list(spec.layers)
    )
    validate_checkpoint_layout(ablated, config, a2.weight_map, old, list(spec.layers))

    # All retained non-bank tensors must exist on both sides. HF may omit one
    # alias of tied embeddings; allow only that documented alias difference.
    tied = bool(config.get("tie_word_embeddings", False))
    unmatched = set(f1.weight_map) ^ set(a2.weight_map)
    invalid = [
        k
        for k in unmatched
        if not is_expert_bank_key(k) and not (tied and k == "lm_head.weight")
    ]
    if invalid:
        raise ValueError(f"F1/A2 non-bank keys disagree: {sorted(invalid)[:8]}")
    for key in set(f1.weight_map) & set(a2.weight_map):
        if not is_expert_bank_key(key) and f1.read(key).shape != a2.read(key).shape:
            raise ValueError(f"F1/A2 retained shape mismatch: {key}")

    output.mkdir(parents=True)
    copy_runtime_side_files(output, [ablated])
    expansion = json.loads((original / EXPANSION_MANIFEST).read_text())
    expansion["inherited_expert_layout"] = expansion["expert_layout"]
    expansion["expert_layout"] = (
        "packed"
        if any(PACKED_EXPERT_KEY_RE.match(k) for k in a2.weight_map)
        else "split"
    )
    write_json_atomic(output / EXPANSION_MANIFEST, expansion, ensure_ascii=True)
    set_config_num_experts(config, new)
    write_json_atomic(output / "config.json", config, ensure_ascii=True)

    writer = ShardedSafetensorsWriter(output, prefix="model-recombined", start_index=1)
    split_stems = set()
    for key in sorted(a2.weight_map):
        tensor = a2.read(key)
        if GATE_KEY_RE.match(key) or EXPERT_SCALE_KEY_RE.match(key):
            tensor = torch.cat((tensor, f1.read(key)[old:new]), 0)
        elif PACKED_EXPERT_KEY_RE.match(key):
            tensor = torch.cat((tensor, f1.packed_tail(key, old, new)), 0)
        elif SPLIT_EXPERT_KEY_RE.match(key):
            split_stems.add(expert_stem(key))
        writer.add(key, tensor)
    for stem in sorted(split_stems):
        for expert in range(old, new):
            for projection in SPLIT_EXPERT_PROJECTIONS:
                writer.add(
                    split_key_from_stem(stem, expert, projection),
                    f1.expert(stem, projection, expert),
                )
    writer.flush()
    write_json_atomic(
        output / "model.safetensors.index.json",
        {
            "metadata": {"total_size": writer.total_size},
            "weight_map": writer.weight_map,
        },
        ensure_ascii=True,
    )
    print(f"Recombined checkpoint: {output}")
    return {
        "restored_expert_indices": list(range(old, new)),
        "output_num_experts": new,
        "expert_layout": expansion["expert_layout"],
        "inherited_expert_layout": expansion["inherited_expert_layout"],
        "layers": list(spec.layers),
    }


def main():
    args = parse_args()
    results = recombine(args.f1, args.a2, args.output)
    write_run_record(
        args.output,
        step="recombine_experts",
        inputs={"f1": args.f1, "a2": args.a2},
        results=results,
    )


if __name__ == "__main__":
    main()
