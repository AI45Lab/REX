from __future__ import annotations

from moe.model_adapters import GEMMA, adapter_from_checkpoint
from common.tokenizers import ensure_pad_token

import argparse
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from dataset.xtuner_vendor.datasets.jsonl import JsonlDataset
from dataset.xtuner_vendor.datasets.packing import ExpandSoftPackDataset, HardPackDataset
from dataset.xtuner_vendor.datasets.pt_tokenize_fn import (
    LongTextPretrainTokenizeFunction,
)
from dataset.xtuner_vendor.datasets.sampler import LengthGroupedSampler, ParallelSampler
from dataset.xtuner_vendor.datasets.utils import tokenizer_xxhash

from .hf_dataset import HfPretrainDataset
from .recipe import DatasetSource, load_dataset_sources

IGNORE_INDEX = -100
SEQUENCE_KEYS = ("input_ids", "attention_mask", "labels")


@dataclass(frozen=True)
class SourceBuildContext:
    tokenizer: Any
    config: dict[str, Any]
    cache_dir: Path | None
    tokenizer_hash: str | None


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in value]


def _has_only_jsonl_files(source: DatasetSource) -> bool:
    return all(path.suffix.lower() == ".jsonl" for path in source.files)


def _token_cache_dir(config: dict[str, Any]) -> Path | None:
    value = (config.get("xtuner") or {}).get("cache_dir")
    return Path(value).expanduser() if value else None


@contextlib.contextmanager
def _temporary_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        yield
        return
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


@contextlib.contextmanager
def _xtuner_jsonl_single_process_mode():
    """Build vendored JSONL caches on rank0 after torch.distributed is initialized.

    Vendored XTuner datasets call barriers and all-gathers when torch.distributed
    is initialized. Prewarm runs only on accelerator main process, so those calls
    must be disabled while the cache is being materialized.
    """

    import dataset.xtuner_vendor.datasets.jsonl as jsonl_module

    original_get_rank = jsonl_module.get_rank
    original_barrier = jsonl_module.barrier
    original_get_local_world_size = jsonl_module.get_local_world_size
    original_get_local_process_group = jsonl_module.get_local_process_group
    original_is_local_rank0 = jsonl_module.is_local_rank0
    original_dist_is_initialized = jsonl_module.dist.is_initialized

    jsonl_module.get_rank = lambda: 0
    jsonl_module.barrier = lambda group=None: None
    jsonl_module.get_local_world_size = lambda: 1
    jsonl_module.get_local_process_group = lambda: None
    jsonl_module.is_local_rank0 = lambda: True
    jsonl_module.dist.is_initialized = lambda: False
    try:
        yield
    finally:
        jsonl_module.get_rank = original_get_rank
        jsonl_module.barrier = original_barrier
        jsonl_module.get_local_world_size = original_get_local_world_size
        jsonl_module.get_local_process_group = original_get_local_process_group
        jsonl_module.is_local_rank0 = original_is_local_rank0
        jsonl_module.dist.is_initialized = original_dist_is_initialized


@contextlib.contextmanager
def _prewarm_lock(cache_dir: Path, *, stale_seconds: float = 6 * 3600, wait_seconds: float = 24 * 3600):
    lock_path = cache_dir / ".rex-prewarm.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > stale_seconds
            except FileNotFoundError:
                continue
            if stale:
                with contextlib.suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() - start > wait_seconds:
                raise TimeoutError(f"Timed out waiting for XTuner prewarm lock: {lock_path}")
            time.sleep(2.0)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def _source_cache_tag(source: DatasetSource, config: dict[str, Any]) -> str:
    # Shortcut key for the vendored JSONL cache; the cache entry itself is also
    # keyed by file content and tokenizer hash.
    xtuner_cfg = config.get("xtuner") or {}
    payload = {
        "source": source.name,
        "data_format_type": source.data_format_type,
        "text_column": source.text_column,
        "add_eos_token": bool(xtuner_cfg.get("add_eos_token", True)),
        "add_bos_token": bool(xtuner_cfg.get("add_bos_token", False)),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _build_pretrain_tokenize_fn(
    tokenizer: Any,
    config: dict[str, Any],
    *,
    tokenizer_hash: str | None = None,
) -> LongTextPretrainTokenizeFunction:
    # Long documents are split into pack-sized chunks so no tokens are dropped.
    xtuner_cfg = config.get("xtuner") or {}
    return LongTextPretrainTokenizeFunction(
        tokenizer=tokenizer,
        chunk_size=int(xtuner_cfg.get("pack_max_length", config["max_seq_length"])),
        min_chunk_tokens=int(xtuner_cfg.get("min_chunk_tokens", 0)),
        max_length=None,
        add_eos_token=bool(xtuner_cfg.get("add_eos_token", True)),
        add_bos_token=bool(xtuner_cfg.get("add_bos_token", False)),
        tokenizer_hash=tokenizer_hash,
    )


def _build_pretrain_source_datasets(
    source: DatasetSource,
    *,
    context: SourceBuildContext,
) -> list[Any]:
    sample_ratio = source.sample_ratio * source.label_weight
    use_jsonl_long_text = (
        _has_only_jsonl_files(source)
        and source.data_format_type == "pretrain"
        and not source.text_column
        and source.row_split_manifest is None
    )
    tokenize_fn = _build_pretrain_tokenize_fn(
        context.tokenizer,
        context.config,
        tokenizer_hash=context.tokenizer_hash,
    )
    if use_jsonl_long_text:
        return [
            JsonlDataset(
                tokenize_fn=tokenize_fn,
                anno_path=path,
                sample_ratio=sample_ratio,
                enable_sequential_sampler=False,
                name=source.name,
                cache_dir=context.cache_dir,
                cache_tag=_source_cache_tag(source, context.config),
            )
            for path in source.files
        ]
    return [
        HfPretrainDataset(
            source=source,
            tokenize_fn=tokenize_fn,
            sample_ratio=sample_ratio,
            cache_dir=context.cache_dir,
        )
    ]


def _build_source_datasets(
    sources: list[DatasetSource],
    *,
    role: str,
    tokenizer: Any,
    config: dict[str, Any],
    cache_dir: Path | None,
    tokenizer_hash: str | None = None,
) -> list[Any]:
    context = SourceBuildContext(
        tokenizer=tokenizer,
        config=config,
        cache_dir=cache_dir,
        tokenizer_hash=tokenizer_hash,
    )
    datasets: list[Any] = []
    for source in sources:
        if source.role != role:
            continue
        datasets.extend(_build_pretrain_source_datasets(source, context=context))
    if not datasets:
        raise ValueError(f"No dataset sources found for role={role!r}.")
    return datasets


def _prewarm_role_filter(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        roles = [value]
    elif isinstance(value, list):
        roles = value
    else:
        raise ValueError("xtuner.prewarm_roles must be a string or list when provided.")
    normalized = {str(role) for role in roles}
    invalid = normalized - {"std", "harmful"}
    if invalid:
        raise ValueError(f"xtuner.prewarm_roles contains unsupported roles: {sorted(invalid)}")
    return normalized


def _require_xtuner_config(config: dict[str, Any]) -> dict[str, Any]:
    xtuner_cfg = config.get("xtuner") or {}
    if not isinstance(xtuner_cfg, dict):
        raise ValueError("xtuner config must be a mapping.")
    return xtuner_cfg


def _prewarm_sources(config: dict[str, Any], xtuner_cfg: dict[str, Any]) -> list[DatasetSource]:
    role_filter = _prewarm_role_filter(xtuner_cfg.get("prewarm_roles"))
    sources = load_dataset_sources(config)
    if role_filter is not None:
        sources = [source for source in sources if source.role in role_filter]
    return sources


def _load_tokenizer(tokenizer_dir: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )
    ensure_pad_token(tokenizer)
    return tokenizer


def _prewarm_one_source(
    source: DatasetSource,
    *,
    tokenizer: Any,
    config: dict[str, Any],
    cache_dir: Path,
    tokenizer_hash: str,
) -> None:
    try:
        source_datasets = _build_source_datasets(
            [source],
            role=source.role,
            tokenizer=tokenizer,
            config=config,
            cache_dir=cache_dir,
            tokenizer_hash=tokenizer_hash,
        )
        for dataset in source_datasets:
            len(dataset)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prewarm XTuner source {source.name!r} "
            f"role={source.role!r} files={source.files[:3]}"
        ) from exc


def _log_prewarm_progress(
    *,
    idx: int,
    total: int,
    source: DatasetSource,
    source_start: float,
    start: float,
    log_interval: int,
) -> None:
    if log_interval <= 0 or not (idx == 1 or idx % log_interval == 0 or idx == total):
        return
    elapsed = time.perf_counter() - start
    source_elapsed = time.perf_counter() - source_start
    print(
        "[xtuner-data] prewarm "
        f"source={idx}/{total} role={source.role} name={source.name!r} "
        f"source_elapsed={source_elapsed:.1f}s elapsed={elapsed:.1f}s",
        flush=True,
    )


def warmup_xtuner_offset_cache(
    config: dict[str, Any],
    *,
    max_workers: int = 16,
    log_interval: int = 100,
) -> tuple[int, float]:
    xtuner_cfg = _require_xtuner_config(config)

    cache_dir = _token_cache_dir(config)
    if cache_dir is None:
        return 0, 0.0
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources = _prewarm_sources(config, xtuner_cfg)

    tokenizer = _load_tokenizer(str(config["tokenizer_dir"]))

    start = time.perf_counter()
    warmed = 0
    original_tokenize_workers = os.environ.get("XTUNER_TOKENIZE_WORKERS")
    worker_value = str(max(1, int(max_workers))) if max_workers > 0 else original_tokenize_workers
    tokenizer_hash = tokenizer_xxhash(tokenizer)[:16]

    with _prewarm_lock(cache_dir), _temporary_env("XTUNER_TOKENIZE_WORKERS", worker_value):
        with _xtuner_jsonl_single_process_mode():
            for idx, source in enumerate(sources, start=1):
                source_start = time.perf_counter()
                _prewarm_one_source(
                    source,
                    tokenizer=tokenizer,
                    config=config,
                    cache_dir=cache_dir,
                    tokenizer_hash=tokenizer_hash,
                )
                warmed += 1
                _log_prewarm_progress(
                    idx=idx,
                    total=len(sources),
                    source=source,
                    source_start=source_start,
                    start=start,
                    log_interval=log_interval,
                )

    return warmed, time.perf_counter() - start


def _pack_dataset(datasets: list[Any], config: dict[str, Any], *, seed: int) -> Any:
    xtuner_cfg = config.get("xtuner") or {}
    pack_level = str(xtuner_cfg.get("pack_level", "soft")).lower()
    pack_max_length = int(xtuner_cfg.get("pack_max_length", config["max_seq_length"]))
    pack_chunk_size = int(xtuner_cfg.get("pack_chunk_size", 10_000))
    pack_workers = int(xtuner_cfg.get("pack_workers", 8))
    pack_extra_buffer_size = int(xtuner_cfg.get("pack_extra_buffer_size", 100))
    global_pack = bool(xtuner_cfg.get("global_pack", False))

    if pack_level == "soft":
        return ExpandSoftPackDataset(
            datasets,
            pack_max_length=pack_max_length,
            pack_chunk_size=pack_chunk_size,
            pack_workers=pack_workers,
            global_pack=global_pack,
            pack_extra_buffer_size=pack_extra_buffer_size,
            seed=seed,
        )
    if pack_level == "hard":
        return HardPackDataset(
            datasets,
            pack_max_length=pack_max_length,
            pack_chunk_size=pack_chunk_size,
            pack_workers=pack_workers,
            global_pack=global_pack,
            seed=seed,
        )
    if pack_level == "none":
        return ConcatDataset(datasets)
    raise ValueError("xtuner.pack_level must be one of 'none', 'soft', 'hard'.")


def _sampler_for_dataset(dataset: Any, config: dict[str, Any], *, seed: int, global_batch_size: int) -> Any:
    xtuner_cfg = config.get("xtuner") or {}
    pack_level = str(xtuner_cfg.get("pack_level", "soft")).lower()
    group_by_length = bool(xtuner_cfg.get("group_by_length", pack_level != "none"))
    round_up = bool(xtuner_cfg.get("round_up", True))
    if group_by_length and pack_level != "none":
        return LengthGroupedSampler(
            dataset=dataset,
            global_batch_size=global_batch_size,
            seed=seed,
            round_up=round_up,
        )
    return ParallelSampler(
        dataset=dataset,
        global_batch_size=global_batch_size,
        shuffle=True,
        seed=seed,
        round_up=round_up,
    )


class RexXtunerCollator:
    """Merge each packed list of text chunks into one padded training row."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_length: int,
        pack_to_max_length: bool,
        bos_per_pack: bool = False,
        bos_token_id: int | None = None,
    ) -> None:
        self.bos_per_pack = bos_per_pack
        self.bos_token_id = bos_token_id
        if bos_per_pack:
            if bos_token_id is None or bos_token_id < 0:
                raise ValueError("xtuner.bos_per_pack requires a tokenizer BOS token ID.")
            if max_length < 2:
                raise ValueError("xtuner.bos_per_pack requires max_length >= 2.")
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)
        self.pack_to_max_length = bool(pack_to_max_length)

    def _fit_instance(self, instance: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_tokens = sum(len(item["input_ids"]) for item in instance)
        if total_tokens > self.max_length:
            item_locations = [
                {
                    "source": item.get("_rex_source"),
                    "row_index": item.get("_rex_row_index"),
                    "dataset_index": item.get("_rex_dataset_index"),
                    "tokens": len(item["input_ids"]),
                }
                for item in instance
            ]
            raise ValueError(
                "Packed instance exceeds max_length; long-text chunking or packing metadata is inconsistent: "
                f"tokens={total_tokens}, max_length={self.max_length}, items={item_locations}."
            )
        return instance

    def _combine_instance(self, instance: Any) -> dict[str, Any]:
        items = instance if isinstance(instance, list) else [instance]
        items = self._fit_instance([dict(item) for item in items])
        combined: dict[str, Any] = {key: [] for key in SEQUENCE_KEYS}

        for idx, item in enumerate(items):
            input_ids = _as_int_list(item["input_ids"])
            labels = _as_int_list(item["labels"])
            attention_mask = _as_int_list(item.get("attention_mask", [1] * len(input_ids)))
            if idx > 0 and labels:
                labels[0] = IGNORE_INDEX
            combined["input_ids"].extend(input_ids)
            combined["attention_mask"].extend(attention_mask)
            combined["labels"].extend(labels)

        if not combined["input_ids"]:
            combined["input_ids"] = [self.pad_token_id]
            combined["attention_mask"] = [1]
            combined["labels"] = [IGNORE_INDEX]

        for key in SEQUENCE_KEYS:
            combined[key] = combined[key][: self.max_length]
        if self.bos_per_pack:
            # Apply after fitting/merging, not to documents or chunks: keep pack
            # membership/order and displace only the last token of a full pack.
            if combined["input_ids"][0] != self.bos_token_id:
                prefixes = (self.bos_token_id, 1, IGNORE_INDEX)
                for key, prefix in zip(SEQUENCE_KEYS, prefixes):
                    combined[key] = [prefix] + combined[key][: self.max_length - 1]
            combined["labels"][0] = IGNORE_INDEX
            combined["attention_mask"][0] = 1
        return combined

    def _batch_sequence_length(self, rows: list[list[int]]) -> int:
        if self.pack_to_max_length:
            return self.max_length
        return min(self.max_length, max(len(row) for row in rows))

    def _pad_1d(self, rows: list[list[int]], *, pad_value: int, max_length: int) -> torch.Tensor:
        output = torch.full((len(rows), max_length), pad_value, dtype=torch.long)
        for idx, row in enumerate(rows):
            row = row[:max_length]
            if row:
                output[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
        return output

    def __call__(self, rows: list[Any]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty batch.")
        combined_rows = [self._combine_instance(row) for row in rows]
        input_rows = [row["input_ids"] for row in combined_rows]
        max_length = self._batch_sequence_length(input_rows)
        return {
            "input_ids": self._pad_1d(input_rows, pad_value=self.pad_token_id, max_length=max_length),
            "attention_mask": self._pad_1d(
                [row["attention_mask"] for row in combined_rows], pad_value=0, max_length=max_length
            ),
            "labels": self._pad_1d(
                [row["labels"] for row in combined_rows], pad_value=IGNORE_INDEX, max_length=max_length
            ),
        }


def _build_collator(tokenizer: Any, config: dict[str, Any]) -> RexXtunerCollator:
    xtuner_cfg = _require_xtuner_config(config)
    bos_per_pack = xtuner_cfg.get("bos_per_pack", False)
    if not isinstance(bos_per_pack, bool):
        raise ValueError("xtuner.bos_per_pack must be a boolean.")
    if bos_per_pack:
        if xtuner_cfg.get("add_bos_token", False):
            raise ValueError("xtuner.bos_per_pack and xtuner.add_bos_token are mutually exclusive.")
        # Read declared model architecture, never infer it from a directory name
        # or tokenizer class (Gemma also uses generic fast tokenizer classes).
        model_dir = config.get("initial_checkpoint_dir") or config.get("tokenizer_dir")
        if not model_dir or adapter_from_checkpoint(model_dir) != GEMMA:
            raise ValueError("xtuner.bos_per_pack only supports Gemma 4 pure text pretrain.")
    return RexXtunerCollator(
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=int(xtuner_cfg.get("pack_max_length", config["max_seq_length"])),
        pack_to_max_length=bool(xtuner_cfg.get("pack_to_max_length", True)),
        bos_per_pack=bos_per_pack,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
    )


def _build_packed_role_dataset(
    *,
    sources: list[DatasetSource],
    role: str,
    split_seed: int,
    tokenizer: Any,
    tokenizer_hash: str,
    config: dict[str, Any],
    cache_dir: Path | None,
    source_build_context: Any,
) -> Any:
    context_factory = source_build_context or contextlib.nullcontext
    with context_factory():
        with _xtuner_jsonl_single_process_mode():
            role_datasets = _build_source_datasets(
                sources,
                role=role,
                tokenizer=tokenizer,
                config=config,
                cache_dir=cache_dir,
                tokenizer_hash=tokenizer_hash,
            )
    return _pack_dataset(role_datasets, config, seed=split_seed)


def _build_train_loader(
    *,
    dataset: Any,
    config: dict[str, Any],
    collator: RexXtunerCollator,
    train_batch_size: int,
    global_batch_size: int,
    split_seed: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Any,
) -> DataLoader:
    sampler = _sampler_for_dataset(
        dataset,
        config,
        seed=split_seed,
        global_batch_size=global_batch_size,
    )
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": train_batch_size,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "sampler": sampler,
    }
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **dataloader_kwargs)


def create_xtuner_dataloaders(
    config: dict[str, Any],
    *,
    train_batch_size: int,
    global_batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: Any = None,
    source_build_context: Any = None,
) -> tuple[DataLoader, DataLoader]:
    _require_xtuner_config(config)
    sources = load_dataset_sources(config)

    tokenizer = _load_tokenizer(str(config["tokenizer_dir"]))
    tokenizer_hash = tokenizer_xxhash(tokenizer)[:16]
    cache_dir = _token_cache_dir(config)
    collator = _build_collator(tokenizer, config)

    def make_loader(role: str, split_seed: int) -> DataLoader:
        packed_dataset = _build_packed_role_dataset(
            sources=sources,
            role=role,
            split_seed=split_seed,
            tokenizer=tokenizer,
            tokenizer_hash=tokenizer_hash,
            config=config,
            cache_dir=cache_dir,
            source_build_context=source_build_context,
        )
        return _build_train_loader(
            dataset=packed_dataset,
            config=config,
            collator=collator,
            train_batch_size=train_batch_size,
            global_batch_size=global_batch_size,
            split_seed=split_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )

    return (
        make_loader("std", seed),
        make_loader("harmful", seed + 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill the XTuner tokenization/metadata cache of a recipe without GPUs."
    )
    parser.add_argument("config", type=Path, help="REX or RMU recipe with xtuner and datasets sections.")
    parser.add_argument("tokenizer", type=Path, help="Checkpoint directory whose tokenizer the run will use.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["tokenizer_dir"] = str(args.tokenizer.resolve())
    xtuner_cfg = _require_xtuner_config(config)
    if _token_cache_dir(config) is None:
        raise ValueError("The recipe defines no xtuner.cache_dir to fill.")
    workers = int(xtuner_cfg.get("prewarm_workers", 16))
    warmed_sources, elapsed = warmup_xtuner_offset_cache(
        config,
        max_workers=workers,
        log_interval=int(xtuner_cfg.get("prewarm_log_interval", 100)),
    )
    print(
        "[xtuner-data] prewarm_complete "
        f"sources={warmed_sources} workers={workers} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
