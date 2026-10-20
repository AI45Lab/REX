from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from dataset.xtuner_vendor.datasets.data_item import DataItem
from dataset.xtuner_vendor.datasets.pt_tokenize_fn import (
    LongTextPretrainTokenizeFunction,
    PretrainTokenizeFunction,
)

from .recipe import DatasetSource


FORMAT_BY_SUFFIX = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".txt": "text",
}
_METADATA_SCHEMA_VERSION = 3
_METADATA_WORKER_DATASET: Dataset | None = None
_METADATA_WORKER_TOKENIZE_FN: PretrainTokenizeFunction | None = None
_METADATA_WORKER_TEXT_COLUMN: str | None = None


def _metadata_tokenizer_reload_spec(
    tokenize_fn: PretrainTokenizeFunction,
) -> dict[str, Any] | None:
    """Describe how a spawned worker can recreate its tokenizer safely.

    Intern-S2's remote-code tokenizer is not spawn-pickle safe: its nested
    SentencePiece processors lose their dynamically assigned ``offset``
    attributes. Recreating the tokenizer inside each worker preserves those
    runtime attributes and avoids forking a process after CUDA initialization.
    """

    tokenizer_dir = getattr(tokenize_fn.tokenizer, "name_or_path", None)
    if not tokenizer_dir:
        return None
    common = {
        "tokenizer_dir": str(tokenizer_dir),
        "add_eos_token": bool(tokenize_fn.add_eos_token),
        "add_bos_token": bool(tokenize_fn.add_bos_token),
        "tokenizer_hash": getattr(tokenize_fn, "_tokenizer_hash", None),
    }
    if isinstance(tokenize_fn, LongTextPretrainTokenizeFunction):
        return {
            **common,
            "kind": "long_text",
            "chunk_size": int(tokenize_fn.chunk_size),
            "tokenizer_chunk_chars": int(tokenize_fn.tokenizer_chunk_chars),
            "overlap_chars": int(tokenize_fn.overlap_chars),
            "min_chunk_tokens": int(tokenize_fn.min_chunk_tokens),
            "max_length": tokenize_fn.max_length,
        }
    return {**common, "kind": "pretrain"}


def _reload_metadata_worker_tokenize_fn(spec: dict[str, Any]) -> PretrainTokenizeFunction:
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_dir"],
        trust_remote_code=True,
    )
    common = {
        "tokenizer": tokenizer,
        "add_eos_token": spec["add_eos_token"],
        "add_bos_token": spec["add_bos_token"],
        "tokenizer_hash": spec["tokenizer_hash"],
    }
    if spec["kind"] == "long_text":
        return LongTextPretrainTokenizeFunction(
            **common,
            chunk_size=spec["chunk_size"],
            tokenizer_chunk_chars=spec["tokenizer_chunk_chars"],
            overlap_chars=spec["overlap_chars"],
            min_chunk_tokens=spec["min_chunk_tokens"],
            max_length=spec["max_length"],
        )
    if spec["kind"] == "pretrain":
        return PretrainTokenizeFunction(**common)
    raise ValueError(f"Unsupported HF metadata worker tokenize kind: {spec['kind']!r}")


def _initialize_metadata_worker(
    dataset: Dataset,
    tokenize_fn: PretrainTokenizeFunction | None,
    tokenizer_reload_spec: dict[str, Any] | None,
    text_column: str,
) -> None:
    global _METADATA_WORKER_DATASET
    global _METADATA_WORKER_TOKENIZE_FN
    global _METADATA_WORKER_TEXT_COLUMN
    _METADATA_WORKER_DATASET = dataset
    if tokenizer_reload_spec is not None:
        tokenize_fn = _reload_metadata_worker_tokenize_fn(tokenizer_reload_spec)
    if tokenize_fn is None:
        raise RuntimeError("HF metadata worker has neither a tokenizer nor a reload spec.")
    _METADATA_WORKER_TOKENIZE_FN = tokenize_fn
    _METADATA_WORKER_TEXT_COLUMN = text_column
    _METADATA_WORKER_TOKENIZE_FN.set_state("cache")


def _tokenize_metadata_row(row_index: int) -> dict[str, Any]:
    if (
        _METADATA_WORKER_DATASET is None
        or _METADATA_WORKER_TOKENIZE_FN is None
        or _METADATA_WORKER_TEXT_COLUMN is None
    ):
        raise RuntimeError("HF metadata worker was not initialized.")
    row = dict(_METADATA_WORKER_DATASET[int(row_index)])
    value = row.get(_METADATA_WORKER_TEXT_COLUMN)
    item = {"content": "" if value is None else str(value)}
    metadata = _METADATA_WORKER_TOKENIZE_FN(item)
    if not isinstance(metadata, dict):
        raise TypeError(
            f"HF metadata tokenizer returned {type(metadata).__name__}, expected dict."
        )
    return metadata


def _read_metadata_cache(cache_path: Path) -> dict[str, np.ndarray]:
    cached = np.load(cache_path, mmap_mode="r")
    schema_version = int(cached["schema_version"].reshape(-1)[0])
    if schema_version != _METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported HF pretrain metadata cache schema in {cache_path}: "
            f"expected={_METADATA_SCHEMA_VERSION} actual={schema_version}."
        )
    return {key: cached[key] for key in cached.files if key != "schema_version"}


def _write_metadata_cache(
    cache_path: Path,
    *,
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray([_METADATA_SCHEMA_VERSION], dtype=np.int64),
            **metadata,
        )
    tmp_path.replace(cache_path)
    return _read_metadata_cache(cache_path)


def _dataset_format(files: tuple[Path, ...]) -> str:
    suffixes = {path.suffix.lower() for path in files}
    if len(suffixes) != 1:
        raise ValueError(f"Mixed source file formats are not supported in one source: {sorted(suffixes)}")
    suffix = next(iter(suffixes))
    if suffix not in FORMAT_BY_SUFFIX:
        raise ValueError(f"Unsupported source file extension for HF dataset path: {suffix}")
    return FORMAT_BY_SUFFIX[suffix]


def _default_text_column(dataset: Dataset, source: DatasetSource) -> str:
    if source.text_column:
        return source.text_column
    for candidate in ("content", "text"):
        if candidate in dataset.column_names:
            return candidate
    for column in dataset.column_names:
        value = dataset[0][column] if len(dataset) else None
        if isinstance(value, str):
            return str(column)
    raise ValueError(f"Could not infer text column for source {source.name!r}; set text_column explicitly.")


class HfPretrainDataset:
    def __init__(
        self,
        *,
        source: DatasetSource,
        tokenize_fn: PretrainTokenizeFunction,
        sample_ratio: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.source = source
        self.tokenize_fn = tokenize_fn
        self.sample_ratio = float(source.sample_ratio if sample_ratio is None else sample_ratio)
        dataset_format = _dataset_format(source.files)
        self.dataset = load_dataset(
            dataset_format,
            data_files=[str(path) for path in source.files],
            split="train",
            cache_dir=str(cache_dir / "hf_raw") if cache_dir is not None else None,
        )
        if not isinstance(self.dataset, Dataset):
            raise TypeError(f"Expected datasets.Dataset for {source.name!r}, got {type(self.dataset)}")
        self._source_row_indices = np.arange(len(self.dataset), dtype=np.int64)
        self._source_sample_ids: list[str] | None = None
        if source.row_split_manifest is not None:
            split_indices, sample_ids = self._row_split_selection()
            self.dataset = self.dataset.select(split_indices)
            self._source_row_indices = self._source_row_indices[split_indices]
            self._source_sample_ids = sample_ids
        if self.sample_ratio != 1.0:
            ratio_indices = self._sample_ratio_indices(len(self.dataset))
            self.dataset = self.dataset.select(ratio_indices)
            self._source_row_indices = self._source_row_indices[ratio_indices]
            if self._source_sample_ids is not None:
                self._source_sample_ids = [
                    self._source_sample_ids[index] for index in ratio_indices
                ]
        self.text_column = _default_text_column(self.dataset, source)
        self._has_chunks = isinstance(tokenize_fn, LongTextPretrainTokenizeFunction)
        metadata = self._load_or_build_metadata(cache_dir)
        self.num_tokens = metadata["num_tokens"]
        self.proxy_attn_flops = metadata["proxy_attn_flops"]
        self._row_indices = metadata.get("row_indices")
        self._chunks = metadata.get("chunks")
        if self._has_chunks:
            if self._row_indices is None or self._chunks is None:
                raise ValueError(
                    f"Chunked HF pretrain metadata is incomplete for source {source.name!r}."
                )
            if not (
                len(self.num_tokens)
                == len(self.proxy_attn_flops)
                == len(self._row_indices)
                == len(self._chunks)
            ):
                raise ValueError(
                    f"Chunked HF pretrain metadata lengths disagree for source {source.name!r}."
                )

    def _sample_ratio_indices(self, length: int) -> list[int]:
        target = int(length * self.sample_ratio)
        if target <= 0:
            return []
        base_repeats = int(self.sample_ratio)
        indices: list[int] = []
        if base_repeats > 0:
            indices.extend(list(range(length)) * base_repeats)
        remaining = target - len(indices)
        if remaining > 0:
            rng = random.Random(0)
            indices.extend(rng.sample(range(length), remaining))
        return indices[:target]

    def _row_split_selection(self) -> tuple[list[int], list[str]]:
        manifest_path = Path(self.source.row_split_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = manifest["splits"][self.source.row_split]
        manifest_files = {
            str((Path(manifest["source_root"]) / record["relative_path"]).resolve()): record
            for record in manifest["files"]
        }
        selected: list[int] = []
        sample_ids: list[str] = []
        offset = 0
        for source_file in self.source.files:
            record = manifest_files[str(source_file.resolve())]
            relative_path = record["relative_path"]
            for row_index in split["row_indices_by_file"][relative_path]:
                selected.append(offset + int(row_index))
                sample_ids.append(f"{relative_path}:{int(row_index)}")
            offset += int(record["rows"])
        return selected, sample_ids

    def _cache_key(self) -> str:
        payload = [
            f"hf-pretrain-meta-v{_METADATA_SCHEMA_VERSION}",
            self.source.name,
            self.text_column,
            str(self.sample_ratio),
            self.tokenize_fn.hash(),
            f"chunked={self._has_chunks}",
        ]
        for path in self.source.files:
            stat = path.stat()
            payload.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        if self.source.row_split_manifest is not None:
            stat = self.source.row_split_manifest.stat()
            payload.append(
                f"row-split={self.source.row_split_manifest.resolve()}:"
                f"{self.source.row_split}:{stat.st_mtime_ns}"
            )
        return hashlib.sha1("|".join(payload).encode("utf-8")).hexdigest()

    def _metadata_cache_path(self, cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None
        return cache_dir / "hf_pretrain_meta" / f"{self._cache_key()}.npz"

    def _row_as_pretrain_item(self, row: dict[str, Any]) -> dict[str, str]:
        value = row.get(self.text_column)
        return {"content": "" if value is None else str(value)}

    def _build_tokenized_metadata(self) -> dict[str, np.ndarray]:
        num_tokens: list[int] = []
        proxy_attn_flops: list[float] = []
        row_indices: list[int] = []
        chunks: list[tuple[int, int, int]] = []
        worker_count = max(1, int(os.environ.get("XTUNER_TOKENIZE_WORKERS", "1")))
        self.tokenize_fn.set_state("cache")
        try:
            metadata_rows: Any
            pool = None
            if worker_count > 1 and len(self.dataset) > 1:
                tokenizer_reload_spec = _metadata_tokenizer_reload_spec(self.tokenize_fn)
                worker_tokenize_fn = (
                    None if tokenizer_reload_spec is not None else self.tokenize_fn
                )
                context = multiprocessing.get_context("spawn")
                pool = context.Pool(
                    processes=min(worker_count, len(self.dataset)),
                    initializer=_initialize_metadata_worker,
                    initargs=(
                        self.dataset,
                        worker_tokenize_fn,
                        tokenizer_reload_spec,
                        self.text_column,
                    ),
                )
                metadata_rows = pool.imap(
                    _tokenize_metadata_row,
                    range(len(self.dataset)),
                    chunksize=8,
                )
            else:
                metadata_rows = (
                    self.tokenize_fn(self._row_as_pretrain_item(dict(row)))
                    for row in self.dataset
                )
            if pool is not None:
                try:
                    for row_index, meta in enumerate(metadata_rows):
                        self._append_metadata_row(
                            row_index=row_index,
                            meta=meta,
                            num_tokens=num_tokens,
                            proxy_attn_flops=proxy_attn_flops,
                            row_indices=row_indices,
                            chunks=chunks,
                        )
                except BaseException:
                    pool.terminate()
                    pool.join()
                    raise
                else:
                    pool.close()
                    pool.join()
            else:
                for row_index, meta in enumerate(metadata_rows):
                    self._append_metadata_row(
                        row_index=row_index,
                        meta=meta,
                        num_tokens=num_tokens,
                        proxy_attn_flops=proxy_attn_flops,
                        row_indices=row_indices,
                        chunks=chunks,
                    )
        finally:
            self.tokenize_fn.set_state("runtime")
        metadata = {
            "num_tokens": np.asarray(num_tokens, dtype=np.int64),
            "proxy_attn_flops": np.asarray(proxy_attn_flops, dtype=np.float64),
        }
        if self._has_chunks:
            metadata["row_indices"] = np.asarray(row_indices, dtype=np.int64)
            metadata["chunks"] = np.asarray(chunks, dtype=np.int64).reshape(-1, 3)
        return metadata

    def _append_metadata_row(
        self,
        *,
        row_index: int,
        meta: dict[str, Any],
        num_tokens: list[int],
        proxy_attn_flops: list[float],
        row_indices: list[int],
        chunks: list[tuple[int, int, int]],
    ) -> None:
        if self._has_chunks:
            chunk_tokens = list(meta["num_tokens"])
            chunk_flops = list(meta.get("proxy_attn_flops", chunk_tokens))
            chunk_metadata = list(meta["chunks"])
            if not len(chunk_tokens) == len(chunk_flops) == len(chunk_metadata):
                raise ValueError(
                    f"Long-text tokenizer metadata lengths disagree for "
                    f"source={self.source.name!r} row={row_index}."
                )
            num_tokens.extend(int(value) for value in chunk_tokens)
            proxy_attn_flops.extend(float(value) for value in chunk_flops)
            row_indices.extend([row_index] * len(chunk_tokens))
            chunks.extend(
                (
                    int(chunk["char_start"]),
                    int(chunk["char_end"]),
                    int(chunk["token_start_offset"]),
                )
                for chunk in chunk_metadata
            )
        else:
            num_tokens.append(int(meta["num_tokens"]))
            proxy_attn_flops.append(
                float(meta.get("proxy_attn_flops", meta["num_tokens"]))
            )
        completed = row_index + 1
        total = len(self.dataset)
        if completed == 1 or completed % 5_000 == 0 or completed == total:
            print(
                "[xtuner-data][hf-metadata] "
                f"source={self.source.name!r} rows={completed}/{total} "
                f"chunks={len(num_tokens)} workers={max(1, int(os.environ.get('XTUNER_TOKENIZE_WORKERS', '1')))}",
                flush=True,
            )

    def _load_or_build_metadata(self, cache_dir: Path | None) -> dict[str, np.ndarray]:
        cache_path = self._metadata_cache_path(cache_dir)
        if cache_path is not None and cache_path.is_file():
            return _read_metadata_cache(cache_path)

        metadata = self._build_tokenized_metadata()

        if cache_path is not None:
            return _write_metadata_cache(cache_path, metadata=metadata)
        return metadata

    def __len__(self) -> int:
        return len(self.num_tokens) if self._has_chunks else len(self.dataset)

    def __getitem__(self, index: int) -> DataItem:
        dataset_index = int(index)
        row_index = (
            int(self._row_indices[dataset_index])
            if self._has_chunks and self._row_indices is not None
            else dataset_index
        )
        row = dict(self.dataset[row_index])
        kwargs: dict[str, int] = {}
        if self._has_chunks and self._chunks is not None:
            char_start, char_end, token_start_offset = self._chunks[dataset_index]
            kwargs = {
                "char_start": int(char_start),
                "char_end": int(char_end),
                "token_start_offset": int(token_start_offset),
            }
        item = self.tokenize_fn(self._row_as_pretrain_item(row), **kwargs)
        if not isinstance(item, dict):
            raise TypeError(f"Expected tokenized dict for {self.source.name!r}:{index}, got {type(item)}")
        item["_rex_source"] = self.source.name
        item["_rex_row_index"] = int(self._source_row_indices[row_index])
        if self._source_sample_ids is not None:
            item["_rex_sample_id"] = self._source_sample_ids[row_index]
        item["_rex_dataset_index"] = dataset_index
        return item

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None

    def get_state_dict(self) -> dict[str, Any]:
        return {}
