"""Shared checkpoint serialization for the REX and RMU trainers.

Every trainer writes the same physical layout: packed HF shards capped at 2GB,
the tokenizer and the runtime-side files inherited from the parent checkpoint.
Only the ``run.json`` record differs, so callers pass it as ``write_record``.
"""

from __future__ import annotations


from common.manifest_names import EXPANSION_MANIFEST, LEGACY_RECORD_FILES, RUN_RECORD

from common.io_utils import ShardedSafetensorsWriter

import gc
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from common.io_utils import write_json_atomic
from moe.runtime import merge_expanded_moe_state_dict_for_save
from moe.model_adapters import read_weight_index, read_safetensors_header, canonical_name, PACKED_EXPERT_KEY_RE

# Files copied from the parent checkpoint so the output stays directly loadable:
# tokenizer assets, chat template, remote code and README.
RUNTIME_SIDE_FILE_SUFFIXES = {".jinja", ".json", ".md", ".model", ".py", ".txt"}

# Weight-side or run-local files that must come from this run's own save, never
# from the parent. Records are listed because each step writes its own.
SKIP_RUNTIME_SIDE_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "optimizer.bin",
    "pytorch_model_fsdp.bin",
    "random_states_0.pkl",
    "scheduler.bin",
    RUN_RECORD,
    *LEGACY_RECORD_FILES,
}

# The Intern-S2 save override otherwise emits roughly 50 GB shards, which make a
# later FSDP continuation briefly materialize too much host memory across eight
# local ranks. Match the base checkpoint's small-shard layout so continuation and
# evaluation loading stay bounded.
MAX_SHARD_SIZE = "2GB"


# A tokenizer/processor directory can be another checkpoint. Its expert
# structure is unrelated to the weights being saved; only the parent owns it.
PARENT_SIDE_MANIFESTS = {EXPANSION_MANIFEST}


def cuda_empty_cache_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_synchronize_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reduce_tokens(accelerator: Any, local_tokens: int) -> int:
    """Sum a per-rank token count across the process group."""
    tensor = torch.tensor(local_tokens, device=accelerator.device, dtype=torch.long)
    return int(accelerator.reduce(tensor, reduction="sum").item())


def is_weight_side_file(path: Path) -> bool:
    return (
        path.name in SKIP_RUNTIME_SIDE_FILES
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
        or path.suffix == ".bin"
    )


def is_runtime_side_file(path: Path) -> bool:
    return path.suffix in RUNTIME_SIDE_FILE_SUFFIXES or path.name.startswith("tokenizer")


def runtime_source_dirs(config: Mapping[str, Any]) -> list[Path]:
    return [
        Path(str(config["initial_checkpoint_dir"])),
        Path(str(config["tokenizer_dir"])),
    ]


def copy_runtime_side_files(output_dir: Path, source_dirs: Sequence[Path]) -> None:
    seen: set[Path] = set()
    for source_index, source_dir in enumerate(source_dirs):
        resolved = Path(source_dir).resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        for path in resolved.iterdir():
            if source_index and path.name in PARENT_SIDE_MANIFESTS:
                continue
            if path.is_file() and is_runtime_side_file(path) and not is_weight_side_file(path):
                shutil.copy2(path, output_dir / path.name)


def save_checkpoint(
    *,
    accelerator: Any,
    tokenizer: Any,
    model: torch.nn.Module,
    output_dir: Path,
    source_dirs: Sequence[Path],
    merge_expanded_moe: bool = True,
    write_record: Callable[[Path], None] | None = None,
) -> None:
    """Save one HF checkpoint plus its runtime-side files and run record.

    ``merge_expanded_moe`` folds runtime-only harmful expert keys back into the
    standard packed layout; pass ``False`` when the model is already standard.
    ``write_record`` runs on the main process after the weights land.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = bool(model.training)
    model.eval()
    accelerator.wait_for_everyone()
    gc.collect()
    cuda_empty_cache_if_available()

    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process and merge_expanded_moe:
        state_dict = merge_expanded_moe_state_dict_for_save(unwrapped, state_dict)
    if accelerator.is_main_process:
        frozen_names = {canonical_name(n) for n, p in unwrapped.named_parameters() if not p.requires_grad}
        load_dtype = getattr(unwrapped, "_rex_parameter_load_dtype", None)
        loaded_dtypes = (
            {name: load_dtype for name in frozen_names} if load_dtype else None
        )
        frozen_names.update(canonical_name(n) for n, _ in unwrapped.named_buffers())
        preserve_frozen_parent_precision(
            state_dict, Path(source_dirs[0]), frozen_names, loaded_dtypes=loaded_dtypes
        )
    unwrapped.save_pretrained(
        str(output_dir),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        max_shard_size=MAX_SHARD_SIZE,
        # Transformers 5.8 otherwise reverses Qwen's packed tensors into tens
        # of thousands of legacy expert keys, forcing every rank to repack.
        save_original_format=False,
    )
    del state_dict

    if accelerator.is_main_process:
        tokenizer.save_pretrained(str(output_dir))
        copy_runtime_side_files(output_dir, source_dirs)
        preserve_passthrough_weights(output_dir, Path(source_dirs[0]))
        expansion_path = output_dir / EXPANSION_MANIFEST
        if merge_expanded_moe and expansion_path.is_file():
            expansion = json.loads(expansion_path.read_text())
            weight_map = read_weight_index(output_dir)["weight_map"]
            actual_layout = "packed" if any(PACKED_EXPERT_KEY_RE.match(k) for k in weight_map) else "split"
            expansion["inherited_expert_layout"] = expansion["expert_layout"]
            expansion["expert_layout"] = actual_layout
            write_json_atomic(expansion_path, expansion)
        if write_record is not None:
            write_record(output_dir)

    accelerator.wait_for_everyone()
    gc.collect()
    cuda_empty_cache_if_available()
    if was_training:
        model.train()


def preserve_frozen_parent_precision(
    state_dict, parent: Path, frozen_names: set[str], *, loaded_dtypes=None
) -> list[str]:
    """Keep frozen high-precision source values when the loader rounded them.

    Restoring is allowed only if the source cast through the loader and runtime
    dtypes is exactly the current value. A changed frozen tensor is an error,
    never hidden by copy. Buffers must not inherit parameter-loader conversions.
    """
    from safetensors import safe_open
    source_map = read_weight_index(parent)["weight_map"]
    headers = {}
    dtype_sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
    aliases = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value) and value.numel():
            identity = (value.device, value.data_ptr(), value.shape, value.stride(), value.dtype)
            aliases.setdefault(identity, []).append(name)
    restored = []
    processed = set()
    for key in sorted(frozen_names & state_dict.keys() & source_map.keys()):
        if key in processed:
            continue
        tensor = state_dict[key]
        if not tensor.is_floating_point():
            continue
        shard = source_map[key]
        if shard not in headers:
            headers[shard] = read_safetensors_header(parent/shard)
        load_dtype = (loaded_dtypes or {}).get(key, tensor.dtype)
        effective_size = min(
            tensor.element_size(), torch.empty((), dtype=load_dtype).element_size()
        )
        if dtype_sizes.get(headers[shard][key]["dtype"], 0) <= effective_size:
            continue
        with safe_open(parent/shard, framework="pt", device="cpu") as handle:
            source = handle.get_tensor(key)
        expected = source.to(dtype=load_dtype).to(
            device=tensor.device, dtype=tensor.dtype
        )
        if not torch.equal(expected, tensor):
            raise ValueError(f"Frozen parameter changed beyond loader dtype conversion: {key}")
        identity = (tensor.device, tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype)
        replacement = source.clone()
        for alias in aliases.get(identity, [key]):
            state_dict[alias] = replacement
            processed.add(alias)
        restored.append(key)
    return restored


def preserve_passthrough_weights(output_dir: Path, parent: Path) -> None:
    """Preserve native-loader-ignored MTP tensors, bounded to one small shard.

    They retain their original shapes/config semantics and never participate in
    the text decoder's expansion. Unknown ignored keys must not be silently
    classified as MTP by a broad expert-name pattern.
    """
    from safetensors import safe_open
    source_map = read_weight_index(parent)["weight_map"]
    keys = sorted(k for k in source_map if k.startswith("mtp."))
    if not keys:
        return
    single_file = output_dir / "model.safetensors"
    if single_file.is_file():
        # HF gives model.safetensors precedence over the index. Move the native
        # save into the index so ignored tensors remain part of the checkpoint,
        # including when a small checkpoint is saved repeatedly to one folder.
        header = read_safetensors_header(single_file)
        shard_name = "model-serialized.safetensors"
        single_file.replace(output_dir / shard_name)
        index = {"metadata": {}, "weight_map": {k: shard_name for k in header}}
    else:
        index = read_weight_index(output_dir)
    target_map = index["weight_map"]
    writer = ShardedSafetensorsWriter(output_dir, prefix="model-passthrough", weight_map=target_map)
    for key in keys:
        if key in target_map:
            continue
        with safe_open(parent / source_map[key], framework="pt", device="cpu") as handle:
            writer.add(key, handle.get_tensor(key))
    writer.flush()
    written, total = writer.shards, writer.total_size
    # Unsharded native saves may not have supplied total_size.
    if "total_size" not in index.get("metadata", {}):
        original_bytes = 0
        for shard in set(target_map.values()) - set(written):
            with safe_open(output_dir / shard, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    t = handle.get_tensor(key)
                    original_bytes += t.numel() * t.element_size()
        index.setdefault("metadata", {})["total_size"] = original_bytes
    index["metadata"]["total_size"] += total
    write_json_atomic(output_dir / "model.safetensors.index.json", index)



def record_token_milestone_metrics(
    *,
    checkpoint_dir: Path,
    milestone: int,
    tokens_seen: int,
    log_step: int,
    metrics: dict[str, float],
    token_basis: str = "total",
    split_tokens_seen: Mapping[str, int] | None = None,
) -> bool:
    # LMEvalRunner returns metrics only on the main process, so non-main ranks
    # naturally skip this filesystem write.
    if not metrics:
        return False
    result_path = checkpoint_dir / "lm_eval_token_milestones.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{result_path} must contain a JSON object.")
    else:
        payload = {}
    payload[str(milestone)] = {
        "milestone_tokens": milestone,
        "main_tokens_seen": tokens_seen,
        "log_step": log_step,
        "metrics": metrics,
    }
    if split_tokens_seen is not None:
        payload[str(milestone)].update(
            token_basis=token_basis,
            split_tokens_seen=dict(split_tokens_seen),
        )
    write_json_atomic(result_path, payload, sort_keys=True)
    return True


def write_final_validation_metrics(
    *,
    checkpoint_dir: Path,
    log_step: int,
    metrics: dict[str, float],
) -> Path | None:
    if not metrics:
        return None
    result_path = checkpoint_dir / "final_lm_eval_metrics.json"
    write_json_atomic(
        result_path,
        {
            "log_step": log_step,
            "lm_eval_limit": None,
            "metrics": metrics,
        },
        sort_keys=True,
    )
    return result_path
