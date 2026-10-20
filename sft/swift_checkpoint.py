"""Swift-independent pieces of the SFT plugin, importable on CPU.

``sft/swift_plugin.py`` wires these into ms-swift's Megatron trainer:

* ``square_loss_terms`` reproduces XTuner's ``loss_reduction="square"``.
* ``finalize_checkpoint`` turns a swift HF save into a repository checkpoint
  (same key set as the initial model, runtime side files, the structural
  expansion manifest, ``run.json``, optional verified object-store mirror)
  and moves it to its place in the node directory: the endpoint at the node
  root, any intermediate save milestone under ``milestones/tokens-<N>/``.
"""
from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from common.io_utils import ShardedSafetensorsWriter, write_json_atomic
from common.manifest_names import EXPANSION_MANIFEST, LEGACY_RECORD_FILES, RUN_RECORD
from common.run_record import write_run_record
from moe.model_adapters import read_safetensors_header, read_weight_index, tied_weight_aliases

# Parameters that `--language_model_only` neither trains nor exports. They are
# frozen, so the initial checkpoint's tensors are the trained model's tensors.
PASSTHROUGH_COMPONENTS = ("visual", "vision_tower", "embed_vision", "mtp")
# Runtime assets swift may not re-export (remote code, chat template, processor).
RUNTIME_SIDE_SUFFIXES = {".jinja", ".json", ".model", ".py", ".txt"}
NEVER_COPY = {"config.json", "model.safetensors.index.json", EXPANSION_MANIFEST, RUN_RECORD,
              *LEGACY_RECORD_FILES}
PASSTHROUGH_SHARD_BYTES = 2_000_000_000


def square_loss_terms(losses: torch.Tensor, loss_mask: torch.Tensor, cu_seqlens: torch.Tensor | None,
                      scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerator and integer denominator of XTuner's square loss reduction.

    Each supervised token of sample ``i`` is weighted ``1/sqrt(n_i)``, where
    ``n_i`` is that sample's supervised-token count, and the global loss is
    ``sum(w * l) / sum(w)``. Megatron sums the numerators and the integer
    denominators over micro-batches and data-parallel ranks before dividing,
    so both are multiplied by ``scale`` and the denominator rounded; the
    rounding error is below ``1/scale`` per micro-batch.

    ``losses``/``loss_mask`` are ``[batch, seq]``; ``cu_seqlens`` gives sample
    boundaries inside the single packed row, or ``None`` for one sample per row.
    """
    mask = loss_mask.to(losses.dtype)
    weights = torch.zeros_like(losses)
    if cu_seqlens is None:
        counts = mask.sum(dim=1, keepdim=True)
        weights = mask * counts.clamp(min=1).rsqrt()
    else:
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            segment = mask[:, start:end]
            weights[:, start:end] = segment * segment.sum().clamp(min=1).rsqrt()
    numerator = (losses * weights).sum() * scale
    denominator = torch.round(weights.sum() * scale).to(torch.int)
    return numerator, denominator


def milestone_steps(milestones: tuple[int, ...], tokens_per_step: int, max_tokens: int) -> dict[int, int]:
    """Map optimizer step -> nominal token milestone, validating the schedule."""
    if not milestones or list(milestones) != sorted(set(milestones)):
        raise ValueError(f"Milestones must be strictly increasing: {milestones}")
    if milestones[-1] != max_tokens:
        raise ValueError("The final checkpoint token milestone must equal the token budget")
    if max_tokens % tokens_per_step:
        raise ValueError(f"Token budget {max_tokens} is not divisible by tokens per step {tokens_per_step}")
    bad = [m for m in milestones if m % tokens_per_step]
    if bad:
        raise ValueError(f"Milestones {bad} are not divisible by tokens per step {tokens_per_step}")
    return {m // tokens_per_step: m for m in milestones}


def _component(key: str) -> str | None:
    for part in key.split("."):
        if part in PASSTHROUGH_COMPONENTS:
            return part
    return None


def restore_passthrough_weights(checkpoint: Path, initial: Path) -> dict[str, Any]:
    """Make ``checkpoint`` carry exactly the initial model's weight keys.

    Keys missing from the swift save are copied from ``initial`` only when they
    belong to a component the run never trains. Config-declared tied aliases
    are restored from the *trained export*, never from initial weights. All
    other key differences are errors. Returns what was restored.
    """
    source_map = read_weight_index(initial)["weight_map"]
    single = checkpoint / "model.safetensors"
    if single.is_file():
        shard = "model-00001-of-00001.safetensors"
        single.replace(checkpoint / shard)
        index = {"metadata": {}, "weight_map": {k: shard for k in read_safetensors_header(checkpoint / shard)}}
    else:
        index = read_weight_index(checkpoint)
    target_map = index["weight_map"]
    missing = sorted(set(source_map) - set(target_map))
    extra = sorted(set(target_map) - set(source_map))
    initial_aliases = tied_weight_aliases(json.loads((initial / "config.json").read_text()))
    saved_config = checkpoint / "config.json"
    saved_aliases = tied_weight_aliases(json.loads(saved_config.read_text())) if saved_config.is_file() else {}
    aliases = {key: canonical for key, canonical in initial_aliases.items()
               if key in missing and saved_aliases.get(key) == canonical and canonical in target_map}
    passthrough = [k for k in missing if _component(k) is not None]
    unexpected = [k for k in missing if k not in passthrough and k not in aliases]
    if extra or unexpected:
        raise RuntimeError(
            f"Saved weight keys differ from the initial model {initial}: "
            f"{len(extra)} unexpected (e.g. {extra[:3]}), "
            f"{len(unexpected)} missing trainable (e.g. {unexpected[:3]})")
    if not missing:
        return {}

    from safetensors import safe_open

    writer = ShardedSafetensorsWriter(checkpoint, prefix="model-passthrough",
                                     max_shard_bytes=PASSTHROUGH_SHARD_BYTES, weight_map=target_map)
    for key in missing:
        origin = checkpoint if key in aliases else initial
        tensor_key = aliases.get(key, key)
        shard = target_map[tensor_key] if key in aliases else source_map[key]
        with safe_open(origin / shard, framework="pt", device="cpu") as handle:
            writer.add(key, handle.get_tensor(tensor_key))
    writer.flush()
    added_bytes = writer.total_size
    metadata = index.setdefault("metadata", {})
    if "total_size" in metadata:
        metadata["total_size"] += added_bytes
    index["weight_map"] = dict(sorted(target_map.items()))
    write_json_atomic(checkpoint / "model.safetensors.index.json", index)
    return {
        "frozen_components_from_initial": sorted({_component(k) for k in passthrough}),
        "frozen_keys_from_initial": len(passthrough),
        "tied_weight_aliases_from_saved_checkpoint": aliases,
    }


def copy_runtime_side_files(checkpoint: Path, initial: Path) -> list[str]:
    copied = []
    for path in sorted(initial.iterdir()):
        if (path.is_file() and path.name not in NEVER_COPY and not (checkpoint / path.name).exists()
                and (path.suffix in RUNTIME_SIDE_SUFFIXES or path.name.startswith("tokenizer"))
                and not path.name.endswith(".object_store.json")):  # legacy upload receipts
            shutil.copy2(path, checkpoint / path.name)
            copied.append(path.name)
    return copied


MILESTONE_SUBDIR = "milestones"


def milestone_name(nominal_tokens: int) -> str:
    return f"tokens-{nominal_tokens:012d}"


def place_checkpoint(checkpoint: Path, node_dir: Path, *, nominal_tokens: int, final: bool) -> Path:
    """Move a finalized swift save (``<node>/checkpoint-<step>``) into the node layout.

    An intermediate milestone becomes ``<node>/milestones/tokens-<N>/`` by one
    rename. The endpoint's files move to the node root with ``run.json`` last.
    Swift copies its run-level ``args.json`` into every save; a copy identical
    to the root file is dropped. Any other name already present at the root is
    left in ``checkpoint`` (which is then kept) rather than overwritten.
    Returns the checkpoint's new location.
    """
    if not final:
        target = node_dir / MILESTONE_SUBDIR / milestone_name(nominal_tokens)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Milestone checkpoint already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.rename(target)
        return target
    entries = sorted(checkpoint.iterdir(), key=lambda path: path.name == RUN_RECORD)
    for path in entries:
        destination = node_dir / path.name
        if destination.exists() or destination.is_symlink():
            if path.is_file() and destination.is_file() and filecmp.cmp(path, destination, shallow=False):
                path.unlink()
            continue
        path.rename(destination)
    if not any(checkpoint.iterdir()):
        checkpoint.rmdir()
    return node_dir


def finalize_checkpoint(
    checkpoint: Path, *, node_dir: Path, initial_model: Path, data_path: Path, recipe: dict[str, Any],
    args: dict[str, Any], results: dict[str, Any], nominal_tokens: int, final: bool,
) -> Path:
    """Record the checkpoint and move it to its place in the node; returns that location."""
    restored = restore_passthrough_weights(checkpoint, initial_model)
    copy_runtime_side_files(checkpoint, initial_model)
    # Loaders find the removable bank of an expanded model through this file.
    if (initial_model / EXPANSION_MANIFEST).is_file():
        shutil.copy2(initial_model / EXPANSION_MANIFEST, checkpoint / EXPANSION_MANIFEST)
    write_run_record(checkpoint, step="train_sft", inputs={"model": initial_model, "data": data_path},
                     config=recipe, args=args, results={**results, "restored_weights": restored})
    return place_checkpoint(checkpoint, node_dir, nominal_tokens=nominal_tokens, final=final)
