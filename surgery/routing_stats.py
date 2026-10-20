#!/usr/bin/env python3
"""Count per-layer, per-expert router dispatches of a MoE model over a corpus.

Collection is data parallel: one worker per visible GPU holds a full model
replica and processes a contiguous shard of the corpus. Workers write raw
counts; the parent sums them and derives the frequencies once. Only counts and
valid-token totals are shard-additive; averaging per-shard frequencies would
silently weight shards by their own token totals.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer, PreTrainedTokenizerBase

from common.run_record import write_run_record
from common.tokenizers import ensure_pad_token
from moe.model_adapters import RoutingOutput, config_value, model_adapter


DISPATCH_FREQ_CSV = "dispatch_freq.csv"
COUNTS_DIR = "counts"
TORCH_DTYPE = torch.bfloat16
DISPATCH_FREQ_CSV_FIELDS = [
    "layer",
    "module",
    "expert",
    "dispatch_count",
    "valid_tokens",
    "top_k",
    "dispatches_seen",
    "dispatch_freq",
]
DATASET_BUILDER_BY_SUFFIX = {
    ".parquet": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".txt": "text",
    ".text": "text",
}
SPLIT = "train"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="MoE checkpoint directory.")
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="save_to_disk dataset directory, a directory of one file type, or a single data file.",
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--long-document-policy",
        choices=["chunk", "truncate"],
        default="chunk",
        help=(
            "chunk splits documents longer than --max-length into consecutive windows "
            "so every token is counted; truncate keeps only the first --max-length "
            "tokens and biases corpora with long documents. Corpora compared against "
            "each other must use the same policy."
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on corpus rows (applied before sharding).")
    return parser


def config_int(config: Any, key: str) -> int:
    return int(config_value(config, key, 0) or 0)


def _dataset_from_disk_dir(path: Path) -> Dataset | None:
    try:
        loaded = load_from_disk(str(path))
    except (FileNotFoundError, ValueError):
        return None

    if isinstance(loaded, DatasetDict):
        if SPLIT in loaded:
            return loaded[SPLIT]
        available_splits = ", ".join(sorted(loaded.keys()))
        raise ValueError(
            f"Split {SPLIT!r} not found in dataset {str(path)!r}. "
            f"Available splits: {available_splits}"
        )
    return loaded


def _dataset_files_by_builder(path: Path) -> dict[str, list[str]]:
    files_by_builder: dict[str, list[str]] = {}
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        builder = DATASET_BUILDER_BY_SUFFIX.get(file_path.suffix.lower())
        if builder is not None:
            files_by_builder.setdefault(builder, []).append(str(file_path))
    return files_by_builder


def load_corpus(path: Path) -> Dataset:
    if path.is_dir():
        disk_dataset = _dataset_from_disk_dir(path)
        if disk_dataset is not None:
            return disk_dataset
        files_by_builder = _dataset_files_by_builder(path)
        if len(files_by_builder) == 1:
            builder, files = next(iter(files_by_builder.items()))
            return load_dataset(builder, data_files={SPLIT: files}, split=SPLIT)
        if not files_by_builder:
            raise ValueError(f"No supported dataset files found under: {path}")
        raise ValueError(
            f"Found multiple dataset file types under {path}: {sorted(files_by_builder)}. "
            "Pass a single file or a save_to_disk dataset directory."
        )
    builder = DATASET_BUILDER_BY_SUFFIX.get(path.suffix.lower())
    if builder is None:
        raise ValueError(f"Unsupported dataset file type: {path}")
    return load_dataset(builder, data_files={SPLIT: str(path)}, split=SPLIT)


def chunk_token_ids(input_ids: list[int], max_length: int) -> list[list[int]]:
    """Split one document into consecutive windows of at most max_length tokens."""
    if not input_ids:
        return []
    return [input_ids[start : start + max_length] for start in range(0, len(input_ids), max_length)]


def make_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    max_length: int,
    long_document_policy: str,
):
    def pad_features(sequences: list[list[int]]) -> dict[str, torch.Tensor]:
        features = [
            {"input_ids": sequence, "attention_mask": [1] * len(sequence)}
            for sequence in sequences
            if sequence
        ]
        if not features:
            return {}
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    def collate_text(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = ["" if row[text_column] is None else str(row[text_column]) for row in rows]
        if long_document_policy == "chunk":
            encoded = tokenizer(texts, truncation=False, add_special_tokens=True)
            sequences = []
            for input_ids in encoded["input_ids"]:
                sequences.extend(chunk_token_ids(list(input_ids), max_length))
            return pad_features(sequences)

        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate_text


def filter_selected_by_attention_mask(selected: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return selected

    mask = attention_mask
    if selected.ndim == 2 and selected.shape[0] == mask.numel():
        return selected[mask.reshape(-1)]
    if selected.ndim == 3 and selected.shape[:2] == mask.shape:
        return selected[mask]
    if selected.ndim == 2 and mask.shape[0] == 1 and selected.shape[0] == int(mask.sum()):
        return selected
    raise RuntimeError(
        "Could not align router logits with attention_mask: "
        f"selected_shape={tuple(selected.shape)} mask_shape={tuple(mask.shape)}"
    )


class ExpertActivationCounter:
    def __init__(self, moe_gates: list[tuple[str, torch.nn.Module]], top_k: int, num_experts: int):
        self.layer_names = [name for name, _ in moe_gates]
        self.top_k = top_k
        self.num_experts = num_experts
        self.counts = torch.zeros(len(moe_gates), num_experts, dtype=torch.long)
        self.valid_tokens_per_layer = torch.zeros(len(moe_gates), dtype=torch.long)
        self._attention_mask: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for layer_idx, (_, gate) in enumerate(moe_gates):
            self._handles.append(gate.register_forward_hook(self._make_hook(layer_idx)))

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        self._attention_mask = attention_mask.detach().to("cpu", dtype=torch.bool)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            # Routers that return (logits, weights, indices) report the experts
            # actually dispatched; otherwise recompute top-k from the logits.
            router_logits = (output[0] if isinstance(output, tuple) else output).detach()
            if isinstance(output, RoutingOutput) or (isinstance(output, tuple) and len(output) == 3):
                selected = output[2]
            else:
                selected = torch.topk(router_logits, k=self.top_k, dim=-1).indices
            selected = filter_selected_by_attention_mask(selected.detach().to("cpu"), self._attention_mask)

            if selected.numel() == 0:
                return

            bincount = torch.bincount(selected.reshape(-1), minlength=self.num_experts)
            self.counts[layer_idx] += bincount[: self.num_experts].to(self.counts.dtype)
            self.valid_tokens_per_layer[layer_idx] += selected.reshape(-1, self.top_k).shape[0]

        return hook


def build_dataloader(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerBase,
    *,
    shard_index: int,
    num_shards: int,
) -> tuple[DataLoader, int]:
    dataset = load_corpus(args.corpus)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    column_names = list(dataset.column_names or [])
    if args.text_column not in column_names:
        raise ValueError(f"Text column '{args.text_column}' not found. Available columns: {column_names}")
    # Rows that would produce a zero-length model sequence.
    dataset = dataset.filter(
        lambda row: row.get(args.text_column) is not None and bool(str(row[args.text_column]).strip())
    )
    dataset = dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    collate_fn = make_collate_fn(tokenizer, args.text_column, args.max_length, args.long_document_policy)
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn), len(dataset)


def load_model(model_dir: Path) -> torch.nn.Module:
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map=None,
        dtype=TORCH_DTYPE,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        # Fused kernels used by the linear-attention layers require CUDA tensors,
        # so a CPU-resident model fails at the first forward instead of running slowly.
        model = model.to("cuda")
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    cpu_parameters = sorted(
        {name for name, parameter in model.named_parameters() if parameter.device.type == "cpu"}
    )
    if cpu_parameters and torch.cuda.is_available():
        raise RuntimeError(
            f"{len(cpu_parameters)} parameters stayed on CPU (e.g. {cpu_parameters[:3]}). "
            "Collection would either fail in fused CUDA kernels or run at CPU speed; "
            "fix the device placement instead of collecting partial statistics."
        )
    return model


def model_routing_shape(model: torch.nn.Module) -> tuple[int, int]:
    top_k = config_int(model.config, model_adapter(model).top_k_field)
    num_experts = model_adapter(model).config_num_experts(model.config)
    if top_k <= 0 or num_experts <= 0:
        raise ValueError("Model config must define positive num_experts_per_tok and num_experts.")
    return top_k, num_experts


def routing_forward_kwargs(model: torch.nn.Module) -> dict[str, Any]:
    """Extra forward kwargs that avoid materializing the vocabulary projection.

    The vocabulary logits dominate activation memory (batch x seq x vocab) and are
    irrelevant for router statistics, which come from gate hooks inside the MoE
    blocks. Only pass kwargs the model's signature actually declares: several
    multimodal wrappers swallow unknown kwargs via **kwargs, so a try/except on
    TypeError would not reliably detect support.
    """
    import inspect

    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    return {"logits_to_keep": 1} if "logits_to_keep" in parameters else {}


def collect_activation_counts(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    counter: ExpertActivationCounter,
) -> None:
    batches_seen = 0
    forward_kwargs = routing_forward_kwargs(model)
    print(f"Routing forward kwargs: {forward_kwargs or '{}'}", flush=True)
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for batch in dataloader:
                if not batch:
                    continue
                batch = {key: value for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
                batches_seen += 1
                counter.set_attention_mask(batch["attention_mask"])
                model(**{key: value.to(device) for key, value in batch.items()}, **forward_kwargs)
                if batches_seen % 10 == 0:
                    tokens = int(counter.valid_tokens_per_layer[0].item())
                    print(f"Processed batches={batches_seen} valid_tokens={tokens}", flush=True)
    finally:
        counter.close()


def shard_paths(counts_dir: Path, shard_index: int) -> tuple[Path, Path]:
    return counts_dir / f"shard-{shard_index}.counts.npz", counts_dir / f"shard-{shard_index}.log"


def collect_shard(args: argparse.Namespace, shard_index: int, num_shards: int, counts_dir: Path) -> None:
    """Worker body: count one contiguous corpus shard on the single visible GPU."""
    import numpy as np

    counts_path, log_path = shard_paths(counts_dir, shard_index)
    with log_path.open("w", encoding="utf-8") as log:
        os.dup2(log.fileno(), sys.stdout.fileno())
        os.dup2(log.fileno(), sys.stderr.fileno())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ensure_pad_token(tokenizer, add_if_missing=True)
    dataloader, documents = build_dataloader(args, tokenizer, shard_index=shard_index, num_shards=num_shards)
    model = load_model(args.model)
    top_k, num_experts = model_routing_shape(model)
    moe_gates = [(layer.name, layer.router) for layer in model_adapter(model).moe_layers(model)]
    if not moe_gates:
        raise RuntimeError("No supported text decoder MoE routers found.")
    counter = ExpertActivationCounter(moe_gates, top_k=top_k, num_experts=num_experts)
    collect_activation_counts(model=model, dataloader=dataloader, counter=counter)

    np.savez(
        counts_path,
        counts=counter.counts.numpy(),
        valid_tokens=counter.valid_tokens_per_layer.numpy(),
        top_k=np.array(counter.top_k, dtype=np.int64),
        layer_names=np.array(counter.layer_names, dtype=object),
        documents=np.array(documents, dtype=np.int64),
    )
    print(f"Wrote counts shard {counts_path}", flush=True)


def visible_devices() -> list[str | None]:
    count = torch.cuda.device_count()
    if count == 0:
        return [None]
    listed = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    return listed[:count] if listed else [str(index) for index in range(count)]


def run_workers(args: argparse.Namespace, counts_dir: Path) -> int:
    devices = visible_devices()
    context = multiprocessing.get_context("spawn")
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    processes = []
    try:
        for shard_index, device in enumerate(devices):
            # A spawned worker inherits the environment at start, so each one
            # sees exactly one GPU before CUDA is initialized.
            if device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = device
            process = context.Process(target=collect_shard, args=(args, shard_index, len(devices), counts_dir))
            process.start()
            processes.append(process)
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved

    failed = []
    for shard_index, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append(shard_index)
    for shard_index in failed:
        _, log_path = shard_paths(counts_dir, shard_index)
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        print(f"Collection shard {shard_index} failed; tail of {log_path}:", file=sys.stderr)
        print("\n".join(tail), file=sys.stderr)
    if failed:
        raise SystemExit("Aborting: at least one collection shard failed, so counts are incomplete.")
    return len(devices)


def merge_counts_shards(paths: list[Path]) -> dict[str, Any]:
    import numpy as np

    counts: Any = None
    valid_tokens: Any = None
    top_k: int | None = None
    layer_names: list[str] | None = None
    documents = 0

    for shard_path in paths:
        with np.load(shard_path, allow_pickle=True) as payload:
            shard_counts = payload["counts"]
            shard_valid = payload["valid_tokens"]
            shard_top_k = int(payload["top_k"])
            shard_layers = [str(name) for name in payload["layer_names"]]
            documents += int(payload["documents"])
        if counts is None:
            counts, valid_tokens, top_k, layer_names = shard_counts.copy(), shard_valid.copy(), shard_top_k, shard_layers
            continue
        if shard_counts.shape != counts.shape:
            raise ValueError(f"Counts shard {shard_path} has shape {shard_counts.shape}, expected {counts.shape}.")
        if shard_top_k != top_k:
            raise ValueError(f"Counts shard {shard_path} has top_k={shard_top_k}, expected {top_k}.")
        if shard_layers != layer_names:
            raise ValueError(f"Counts shard {shard_path} has a different layer list than the first shard.")
        counts += shard_counts
        valid_tokens += shard_valid

    layers: list[dict[str, Any]] = []
    for layer_idx, layer_name in enumerate(layer_names or []):
        token_count = int(valid_tokens[layer_idx])
        denominator = token_count * int(top_k or 0)
        experts = [
            {
                "expert": expert_idx,
                "count": int(counts[layer_idx, expert_idx]),
                "dispatch_freq": (int(counts[layer_idx, expert_idx]) / denominator) if denominator else 0.0,
            }
            for expert_idx in range(counts.shape[1])
        ]
        total = sum(expert["count"] for expert in experts)
        if total != denominator:
            raise ValueError(
                f"Merged layer {layer_idx} violates count conservation: "
                f"{total} dispatches counted, expected {denominator}."
            )
        layers.append(
            {
                "layer": layer_idx,
                "module": layer_name,
                "valid_tokens": token_count,
                "top_k": int(top_k or 0),
                "experts": experts,
            }
        )
    return {
        "num_layers": len(layers),
        "num_experts": int(counts.shape[1]),
        "top_k": int(top_k or 0),
        "documents": documents,
        "layers": layers,
    }


def iter_dispatch_freq_rows(results: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for layer in results["layers"]:
        dispatches_seen = int(layer["valid_tokens"]) * int(layer["top_k"])
        for expert in layer["experts"]:
            yield {
                "layer": layer["layer"],
                "module": layer["module"],
                "expert": expert["expert"],
                "dispatch_count": expert["count"],
                "valid_tokens": layer["valid_tokens"],
                "top_k": layer["top_k"],
                "dispatches_seen": dispatches_seen,
                "dispatch_freq": expert["dispatch_freq"],
            }


def write_dispatch_freq_csv(output_path: Path, results: dict[str, Any]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DISPATCH_FREQ_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(iter_dispatch_freq_rows(results))
    print(f"Wrote {output_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    for name in ("batch_size", "max_length", "max_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    args.model, args.corpus, args.output = (path.resolve() for path in (args.model, args.corpus, args.output))

    counts_dir = args.output / COUNTS_DIR
    counts_dir.mkdir(parents=True)
    num_shards = run_workers(args, counts_dir)
    results = merge_counts_shards([shard_paths(counts_dir, index)[0] for index in range(num_shards)])
    write_dispatch_freq_csv(args.output / DISPATCH_FREQ_CSV, results)
    write_run_record(
        args.output,
        step="collect_routing_stats",
        inputs={"model": args.model, "corpus": args.corpus},
        args={
            "text_column": args.text_column,
            "long_document_policy": args.long_document_policy,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "torch_dtype": str(TORCH_DTYPE).removeprefix("torch."),
            "num_shards": num_shards,
        },
        results={
            "documents": results["documents"],
            "num_layers": results["num_layers"],
            "num_experts": results["num_experts"],
            "top_k": results["top_k"],
            "valid_tokens_per_layer": sorted({layer["valid_tokens"] for layer in results["layers"]}),
        },
    )


if __name__ == "__main__":
    main()
