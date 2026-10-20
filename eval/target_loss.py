"""Mean token log-probability of a checkpoint on the full target (forget) corpus.

The target corpus is every ``role: harmful`` dataset of a training recipe,
tokenized and chunked exactly as in training. Writes ``target_loss.json``
(corpus totals), ``documents.csv`` (one row per document) and ``run.json``.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer

from common.io_utils import write_json_atomic
from common.run_record import write_run_record
from common.tokenizers import ensure_pad_token
from dataset.xtuner.hf_dataset import HfPretrainDataset
from dataset.xtuner.recipe import load_dataset_sources
from dataset.xtuner_vendor.datasets.pt_tokenize_fn import LongTextPretrainTokenizeFunction
from moe.model_adapters import adapter_from_config

BATCH_SIZE = 1


def causal_token_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1].to(torch.float32)
    shifted_labels = labels[:, 1:].to(device=logits.device)
    valid = shifted_labels != ignore_index
    targets = shifted_labels.masked_fill(~valid, 0)
    token_nll = -torch.gather(
        torch.log_softmax(shifted_logits, dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return (token_nll * valid).sum(dim=1), valid.sum(dim=1)


@dataclass
class _DocumentNll:
    nll_sum: float = 0.0
    tokens: int = 0
    chunks: int = 0


@dataclass
class TargetLossAccumulator:
    documents: dict[str, _DocumentNll] = field(default_factory=dict)
    skipped_zero_token_chunks: int = 0
    skipped_zero_token_sample_ids: set[str] = field(default_factory=set)

    def add(
        self,
        sample_ids: Sequence[str],
        nll_sums: Sequence[float],
        token_counts: Sequence[int],
    ) -> None:
        for sample_id, nll_sum, token_count in zip(
            sample_ids, nll_sums, token_counts, strict=True
        ):
            sample_id = str(sample_id)
            token_count = int(token_count)
            if token_count == 0:
                self.skipped_zero_token_chunks += 1
                self.skipped_zero_token_sample_ids.add(sample_id)
                continue
            document = self.documents.setdefault(sample_id, _DocumentNll())
            document.nll_sum += float(nll_sum)
            document.tokens += token_count
            document.chunks += 1

    def document_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": sample_id,
                "mean_token_nll": document.nll_sum / document.tokens,
                "mean_token_logprob": -document.nll_sum / document.tokens,
                "tokens": document.tokens,
                "chunks": document.chunks,
            }
            for sample_id, document in sorted(self.documents.items())
        ]

    def totals(self) -> dict[str, Any]:
        nll_sum = sum(document.nll_sum for document in self.documents.values())
        tokens = sum(document.tokens for document in self.documents.values())
        chunks = sum(document.chunks for document in self.documents.values())
        if tokens == 0:
            raise ValueError("Target corpus has no tokens with a causal prediction target.")
        mean_nll = nll_sum / tokens
        return {
            "metric": "mean_token_logprob",
            "direction": "minimize",
            "mean_token_logprob": -mean_nll,
            "mean_token_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "documents": len(self.documents),
            "chunks": chunks,
            "tokens": tokens,
            "skipped_zero_token_chunks": self.skipped_zero_token_chunks,
            "skipped_zero_token_documents": len(self.skipped_zero_token_sample_ids),
            "skipped_zero_token_sample_ids": sorted(self.skipped_zero_token_sample_ids),
        }


def _target_datasets(config: dict[str, Any], *, tokenizer: Any) -> list[HfPretrainDataset]:
    xtuner = config.get("xtuner") or {}
    cache_dir = Path(xtuner["cache_dir"]) if xtuner.get("cache_dir") else None
    chunk_size = int(xtuner.get("pack_max_length", config["max_seq_length"]))
    tokenize_fn = LongTextPretrainTokenizeFunction(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        # Keep the training tokenizer's default character window. A token pack
        # size can be <= the 512-character overlap and is not a valid window.
        min_chunk_tokens=0,
        max_length=None,
        add_eos_token=bool(xtuner.get("add_eos_token", True)),
        add_bos_token=bool(xtuner.get("add_bos_token", False)),
    )
    return [
        HfPretrainDataset(
            source=source,
            tokenize_fn=tokenize_fn,
            sample_ratio=1.0,
            cache_dir=cache_dir,
        )
        for source in load_dataset_sources(config)
        if source.role == "harmful"
    ]


def _items(datasets: Sequence[HfPretrainDataset]) -> Iterable[dict[str, Any]]:
    for dataset in datasets:
        for index in range(len(dataset)):
            item = dict(dataset[index])
            # Plain user JSONL need not carry an explicit sample ID. The source
            # row index stays constant across chunks of the same document.
            sample_id = item.get("_rex_sample_id", item["_rex_row_index"])
            item["_rex_sample_id"] = f"{dataset.source.name}/{sample_id}"
            yield item


def _batches(items: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _collate(
    items: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(item["input_ids"]) for item in items)
    input_ids = torch.full((len(items), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), length), dtype=torch.long)
    labels = torch.full((len(items), length), -100, dtype=torch.long)
    sample_ids: list[str] = []
    for row, item in enumerate(items):
        item_length = len(item["input_ids"])
        input_ids[row, :item_length] = torch.tensor(item["input_ids"])
        attention_mask[row, :item_length] = 1
        labels[row, :item_length] = torch.tensor(item["labels"])
        sample_ids.append(str(item["_rex_sample_id"]))
    return sample_ids, input_ids, attention_mask, labels


def evaluate_target_loss(
    *,
    model: torch.nn.Module,
    datasets: Sequence[HfPretrainDataset],
    pad_token_id: int,
    batch_size: int,
) -> TargetLossAccumulator:
    accumulator = TargetLossAccumulator()
    input_device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for batch_index, items in enumerate(_batches(_items(datasets), batch_size), start=1):
            empty = [item for item in items if len(item["input_ids"]) < 2]
            if empty:
                accumulator.add([item["_rex_sample_id"] for item in empty],
                                [0.0] * len(empty), [0] * len(empty))
                items = [item for item in items if len(item["input_ids"]) >= 2]
            if not items:
                continue
            sample_ids, input_ids, attention_mask, labels = _collate(items, pad_token_id=pad_token_id)
            outputs = model(
                input_ids=input_ids.to(input_device),
                attention_mask=attention_mask.to(input_device),
                use_cache=False,
            )
            nll_sums, token_counts = causal_token_nll(outputs.logits, labels)
            accumulator.add(
                sample_ids,
                nll_sums.detach().cpu().tolist(),
                token_counts.detach().cpu().tolist(),
            )
            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"[target-loss] batches={batch_index} documents={len(accumulator.documents)}",
                    flush=True,
                )
    return accumulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    ensure_pad_token(tokenizer)
    datasets = _target_datasets(config, tokenizer=tokenizer)
    checkpoint_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = adapter_from_config(checkpoint_config).auto_model_class(checkpoint_config).from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    accumulator = evaluate_target_loss(
        model=model,
        datasets=datasets,
        pad_token_id=int(tokenizer.pad_token_id),
        batch_size=BATCH_SIZE,
    )

    rows = accumulator.document_rows()
    with (args.output / "documents.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    totals = accumulator.totals()
    target_sources = {
        name: source.get("path")
        for name, source in config["datasets"].items()
        if source.get("role") == "harmful"
    }
    write_json_atomic(
        args.output / "target_loss.json",
        {"target_domain": config.get("target_domain"), "target_sources": target_sources, **totals},
        sort_keys=True,
    )
    write_run_record(
        args.output,
        step="eval_target_loss",
        inputs={"model": args.model, "config": args.config},
        config=config,
        args={"batch_size": BATCH_SIZE, "dtype": "bfloat16", "corpus": "all role: harmful datasets, full"},
        results={k: v for k, v in totals.items() if k != "skipped_zero_token_sample_ids"},
    )


if __name__ == "__main__":
    main()
