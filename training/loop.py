"""Shared optimizer, checkpoint timing and cleanup primitives."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingFinalizer:
    accelerator: Any
    resources: tuple = ()
    _closed: bool = field(default=False, init=False)

    def __post_init__(self):
        atexit.register(self.close)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for resource in self.resources:
                if resource is not None:
                    resource.close()
        finally:
            atexit.unregister(self.close)
            try:
                self.accelerator.end_training()
            except RuntimeError:
                pass


def optimizer_step(
    ctx,
    clip: float = 0.0,
    *,
    before_step=None,
    after_step=None,
    after_clip=None,
) -> None:
    """Apply the update inside the caller's accumulate block, after backward."""
    if clip > 0 and ctx.accelerator.sync_gradients:
        norm = ctx.accelerator.clip_grad_norm_(ctx.model.parameters(), clip)
        if after_clip is not None:
            after_clip(norm)
    token = before_step() if before_step is not None else None
    try:
        ctx.optimizer.step()
    finally:
        if after_step is not None:
            after_step(token)
    if ctx.scheduler is not None and ctx.accelerator.sync_gradients:
        ctx.scheduler.step()
    ctx.optimizer.zero_grad(set_to_none=True)

