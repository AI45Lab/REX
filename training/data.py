from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from accelerate.utils import send_to_device


def batch_token_count(batch: dict[str, torch.Tensor]) -> int:
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask.detach().sum().item())
    input_ids = batch.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.numel())
    return 0


@dataclass
class PairedBatchStream:
    retain_loader: Any
    forget_loader: Any
    device: torch.device
    non_blocking: bool

    def __post_init__(self) -> None:
        self._retain_iterator = iter(self.retain_loader)
        self._forget_iterator = iter(self.forget_loader)

    def next(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, int]:
        retain_batch, retain_tokens = self._next("retain")
        forget_batch, forget_tokens = self._next("forget")
        return retain_batch, forget_batch, retain_tokens, forget_tokens

    def _next(self, kind: str) -> tuple[dict[str, torch.Tensor], int]:
        if kind == "retain":
            iterator_name = "_retain_iterator"
            loader = self.retain_loader
        else:
            iterator_name = "_forget_iterator"
            loader = self.forget_loader
        iterator = getattr(self, iterator_name)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            setattr(self, iterator_name, iterator)
            batch = next(iterator)
        tokens = batch_token_count(batch)
        return (
            send_to_device(batch, self.device, non_blocking=self.non_blocking),
            tokens,
        )


@dataclass
class TrainBatchStream:
    loaders: dict[str, Any]
    device: torch.device
    non_blocking: bool

    def __post_init__(self) -> None:
        self._iters = {name: iter(loader) for name, loader in self.loaders.items()}

    def next(self, split_name: str) -> tuple[dict[str, torch.Tensor], int]:
        try:
            batch = next(self._iters[split_name])
        except StopIteration:
            self._iters[split_name] = iter(self.loaders[split_name])
            batch = next(self._iters[split_name])
        local_tokens = batch_token_count(batch)
        return send_to_device(
            batch, self.device, non_blocking=self.non_blocking
        ), local_tokens
