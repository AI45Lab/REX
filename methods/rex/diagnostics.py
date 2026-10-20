"""Opt-in REX probes at optimizer boundaries, using fixed training microbatches."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
from accelerate.utils import send_to_device

from moe.model_adapters import parameter_role
from training.lm_eval_runner import preserve_rng_state


class RexDiagnostics:
    def __init__(
        self,
        *,
        model,
        accelerator,
        config,
        output_dir,
        activation_masker,
        routing_collector,
    ):
        self.model = model
        self.accelerator = accelerator
        self.unwrapped = accelerator.unwrap_model(model)
        self.masker = activation_masker
        self.collector = routing_collector
        self.steps = set(config.get("diagnostic_probe_steps", []))
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in self.steps
        ):
            raise ValueError("diagnostic_probe_steps must contain positive integers")
        self.samples = int(config.get("diagnostic_parameter_samples", 256))
        if self.samples <= 0:
            raise ValueError("diagnostic_parameter_samples must be positive")
        self.entries = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                role = parameter_role(self.unwrapped, name)
                if role.role not in {
                    "std_router",
                    "harmful_router",
                    "std_expert",
                    "harmful_expert",
                }:
                    raise ValueError(
                        f"Unexpected trainable diagnostic parameter: {name}"
                    )
                self.entries.append((name, parameter, role))
        expected_dtype = config.get("diagnostic_expected_parameter_dtype")
        if expected_dtype is not None and any(
            str(p.dtype) != expected_dtype for _, p, _ in self.entries
        ):
            raise ValueError(
                f"Diagnostic parameter dtype differs from {expected_dtype}"
            )
        groups = {f"layer{r.layer}/{r.role}" for _, _, r in self.entries}
        if getattr(self.unwrapped, "_rex_harmful_expert_scale_lr", None) is not None:
            groups.update(
                f"layer{r.layer}/{r.role}/per_expert_scale"
                for _, _, r in self.entries
                if r.projection == "per_expert_scale"
            )
        self.groups = sorted(groups)
        self.probe_batch_count = config.get("diagnostic_probe_batches", 1)
        if (
            isinstance(self.probe_batch_count, bool)
            or not isinstance(self.probe_batch_count, int)
            or self.probe_batch_count < 1
        ):
            raise ValueError("diagnostic_probe_batches must be a positive integer")
        self.batch_banks: dict[str, list[dict]] = {"std": [], "harmful": []}
        self.batches: dict[str, dict] = {}
        self.completed: set[int] = set()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_expert_scales(step=0)

    def _write(self, filename: str, row: dict) -> None:
        if self.accelerator.is_main_process:
            with (self.output_dir / filename).open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")

    def remember(self, split: str, batch: dict) -> None:
        if split not in self.batch_banks:
            raise ValueError(f"Unsupported diagnostic split: {split}")
        if len(self.batch_banks[split]) < self.probe_batch_count:
            stored = {
                k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            self.batch_banks[split].append(stored)
            self.batches.setdefault(split, stored)

    def record_clipping(
        self, total_norm, *, max_norm: float, step: int, split: str
    ) -> None:
        norm = (
            float(total_norm.detach().float().item())
            if torch.is_tensor(total_norm)
            else float(total_norm)
        )
        self._write(
            "gradient_clipping.jsonl",
            {
                "step": step,
                "split": split,
                "pre_clipping_total_norm": norm,
                "max_norm": max_norm,
                "clip_coefficient": min(1.0, max_norm / (norm + 1e-6)),
                "measurement": "actual accumulated training gradient; norm returned by accelerator clipping",
            },
        )

    def capture_parameters(self) -> list[tuple[torch.Tensor | None, torch.Tensor]]:
        captured = []
        for _, p, role in self.entries:
            count = p.numel()
            indices = None
            if role.role.endswith("expert") and count > self.samples:
                indices = (
                    torch.arange(self.samples, device=p.device) * count // self.samples
                )
            flat = p.detach().reshape(-1)
            values = flat if indices is None else flat[indices]
            captured.append((indices, values.float().cpu().clone()))
        return captured

    def record_update(self, before, *, step: int, split: str) -> None:
        values = []
        for (_, p, _), (indices, old) in zip(self.entries, before):
            flat = p.detach().reshape(-1)
            new = (flat if indices is None else flat[indices]).float().cpu()
            delta = new.double() - old.double()
            values.append(
                [
                    old.double().square().sum().item(),
                    delta.square().sum().item(),
                    int((old != new).sum()),
                    old.numel(),
                    p.numel(),
                ]
            )
        total = (
            self.accelerator.reduce(
                torch.tensor(
                    values, dtype=torch.float64, device=self.accelerator.device
                ),
                reduction="sum",
            )
            .cpu()
            .tolist()
        )
        records = []
        for (name, p, role), (old_sq, delta_sq, changed, count, size) in zip(
            self.entries, total
        ):
            records.append(
                {
                    "name": name,
                    "role": role.role,
                    "layer": role.layer,
                    "dtype": str(p.dtype),
                    "sampled_count": int(count),
                    "parameter_count": int(size),
                    "measurement": "exact"
                    if role.role.endswith("router")
                    else "shard_stratified_sample",
                    "changed_count": int(changed),
                    "delta_norm": math.sqrt(delta_sq),
                    "relative_delta_norm": math.sqrt(delta_sq / old_sq)
                    if old_sq
                    else None,
                }
            )
        self._write(
            "parameter_updates.jsonl",
            {"step": step, "split": split, "parameters": records},
        )
        self.record_expert_scales(step=step)

    def record_expert_scales(self, *, step: int) -> None:
        if getattr(self.unwrapped, "_rex_harmful_expert_scale_lr", None) is None:
            return
        entries = [
            (n, p, parameter_role(self.unwrapped, n))
            for n, p in self.model.named_parameters()
            if parameter_role(self.unwrapped, n).projection == "per_expert_scale"
        ]
        values = []
        for _, p, _ in entries:
            x = p.detach().double().reshape(-1)
            values.append(
                [
                    x.sum().item(),
                    x.square().sum().item(),
                    x.numel(),
                    (x <= 0).sum().item(),
                ]
            )
        totals = (
            self.accelerator.reduce(
                torch.tensor(
                    values, dtype=torch.float64, device=self.accelerator.device
                ),
                reduction="sum",
            )
            .cpu()
            .tolist()
        )
        self._write(
            "expert_scales.jsonl",
            {
                "step": step,
                "parameters": [
                    {
                        "name": n,
                        "layer": r.layer,
                        "bank": r.role,
                        "dtype": str(p.dtype),
                        "trainable": p.requires_grad,
                        "count": int(count),
                        "mean": total / count,
                        "rms": math.sqrt(squares / count),
                        "nonpositive_count": int(nonpositive),
                    }
                    for (n, p, r), (total, squares, count, nonpositive) in zip(
                        entries, totals
                    )
                ],
            },
        )

    def _gradient_stats(self):
        squares = {group: 0.0 for group in self.groups}
        routers = {}
        for name, p, role in self.entries:
            group = f"layer{role.layer}/{role.role}"
            if p.grad is not None:
                norm = torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64)
                squares[group] += norm.item() ** 2
                scale_group = group + "/per_expert_scale"
                if role.projection == "per_expert_scale" and scale_group in squares:
                    squares[scale_group] += norm.item() ** 2
            if role.role.endswith("router"):
                routers[name] = (
                    p.grad.detach().reshape(-1).float().cpu().clone()
                    if p.grad is not None
                    else torch.zeros(p.numel())
                )
        totals = (
            self.accelerator.reduce(
                torch.tensor(
                    [squares[g] for g in self.groups],
                    dtype=torch.float64,
                    device=self.accelerator.device,
                ),
                reduction="sum",
            )
            .cpu()
            .tolist()
        )
        return {g: math.sqrt(v) for g, v in zip(self.groups, totals)}, routers

    def _cosines(self, left, right):
        stats = {g: [0.0, 0.0, 0.0] for g in self.groups if g.endswith("router")}
        for name, _, role in self.entries:
            if not role.role.endswith("router"):
                continue
            a, b = left[name].double(), right[name].double()
            row = stats[f"layer{role.layer}/{role.role}"]
            row[0] += torch.dot(a, b).item()
            row[1] += torch.dot(a, a).item()
            row[2] += torch.dot(b, b).item()
        total = (
            self.accelerator.reduce(
                torch.tensor(
                    list(stats.values()),
                    dtype=torch.float64,
                    device=self.accelerator.device,
                ),
                reduction="sum",
            )
            .cpu()
            .tolist()
        )
        return {
            g: dot / math.sqrt(a * b) if a and b else None
            for g, (dot, a, b) in zip(stats, total)
        }

    def probe(
        self,
        *,
        step: int,
        current_split: str,
        forward_loss: Callable,
        final: bool = False,
    ) -> None:
        if step in self.completed or (step not in self.steps and not final):
            return
        if set(self.batches) != {"std", "harmful"}:
            raise ValueError(
                "Probe requires a fixed training microbatch from each split"
            )
        if not self.accelerator.sync_gradients or any(
            p.grad is not None for _, p, _ in self.entries
        ):
            raise RuntimeError(
                "Probe must run after an optimizer boundary with cleared gradients"
            )
        if any(
            len(bank) != self.probe_batch_count for bank in self.batch_banks.values()
        ):
            raise ValueError("Probe requires all configured fixed training batches")
        original = self.batches
        try:
            for batch_index in range(self.probe_batch_count):
                self.batches = {
                    split: bank[batch_index] for split, bank in self.batch_banks.items()
                }
                self._probe_one(
                    step=step,
                    current_split=current_split,
                    forward_loss=forward_loss,
                    batch_index=batch_index,
                )
            self.completed.add(step)
        finally:
            self.batches = original

    def _probe_one(
        self, *, step: int, current_split: str, forward_loss: Callable, batch_index: int
    ) -> None:
        records, vectors, totals, routing = {}, {}, {}, {}
        batch_hashes = {}
        local_token_forwards = 0
        try:
            with preserve_rng_state():
                for split in ("std", "harmful"):
                    batch = send_to_device(self.batches[split], self.accelerator.device)
                    digest = hashlib.sha256()
                    for key, value in sorted(self.batches[split].items()):
                        if torch.is_tensor(value):
                            digest.update(key.encode())
                            digest.update(str((value.shape, value.dtype)).encode())
                            digest.update(
                                value.contiguous().view(torch.uint8).numpy().tobytes()
                            )
                    encoded = torch.tensor(
                        list(digest.digest()),
                        dtype=torch.uint8,
                        device=self.accelerator.device,
                    )
                    hashes = (
                        self.accelerator.gather(encoded).cpu().reshape(-1, 32).tolist()
                    )
                    batch_hashes[split] = [bytes(row).hex() for row in hashes]
                    terms = ["lm_loss", "weighted_routing_loss"]
                    if split == "std":
                        terms.append("weighted_rank_loss")
                    split_vectors = []
                    for term in terms:
                        self.model.zero_grad(set_to_none=True)
                        self.masker.enable() if split == "std" else self.masker.disable()
                        # Each component uses the same RNG state and fixed batch.
                        with preserve_rng_state():
                            result = forward_loss(batch, split, True)[1]
                            value = getattr(result, term)
                            # Keep the root FSDP output in auxiliary-only backward graphs.
                            loss = value + result.lm_loss * 0.0
                            scale = int(
                                getattr(
                                    self.accelerator, "gradient_accumulation_steps", 1
                                )
                            )
                            self.accelerator.backward(loss * scale)
                        from methods.rex.masking import GradientMasker

                        GradientMasker(self.unwrapped).mask_non_target_parameters(split)
                        norms, grads = self._gradient_stats()
                        label = f"{split}/{term}"
                        mean = self.accelerator.reduce(
                            value.detach().float().reshape(1), reduction="sum"
                        )
                        records[label] = {
                            "loss": mean.item() / self.accelerator.num_processes,
                            "gradient_norms": norms,
                        }
                        vectors[label] = grads
                        split_vectors.append(grads)
                        local_token_forwards += int((batch["labels"] != -100).sum())
                        if term == "lm_loss":
                            # These are per-rank microbatch statistics, reduced with the same training helper.
                            from methods.rex.metrics import (
                                finalize_layer_routing_stats,
                                reduce_layer_routing_stats,
                            )

                            stats = reduce_layer_routing_stats(
                                self.accelerator, self.collector.layer_stats(value)
                            )
                            routing[split] = finalize_layer_routing_stats(stats)
                        del result, value, loss
                    totals[split] = {
                        name: sum(v[name] for v in split_vectors)
                        for name in split_vectors[0]
                    }
                cosines = {
                    "std/lm_vs_routing": self._cosines(
                        vectors["std/lm_loss"], vectors["std/weighted_routing_loss"]
                    ),
                    "std/lm_vs_rank": self._cosines(
                        vectors["std/lm_loss"], vectors["std/weighted_rank_loss"]
                    ),
                    "harmful/lm_vs_routing": self._cosines(
                        vectors["harmful/lm_loss"],
                        vectors["harmful/weighted_routing_loss"],
                    ),
                    "std_vs_harmful_total": self._cosines(
                        totals["std"], totals["harmful"]
                    ),
                }
                tokens = self.accelerator.reduce(
                    torch.tensor(
                        [local_token_forwards], device=self.accelerator.device
                    ),
                    reduction="sum",
                ).item()
                self._write(
                    "gradient_probes.jsonl",
                    {
                        "step": step,
                        "probe_batch_index": batch_index,
                        "measurement": "fixed microbatch per rank; gradients averaged across ranks; before clipping; no optimizer step",
                        "batch_sha256_by_rank": batch_hashes,
                        "extra_forward_label_tokens": tokens,
                        "components": records,
                        "router_cosines": cosines,
                        "routing": routing,
                    },
                )
        finally:
            self.model.zero_grad(set_to_none=True)
            self.collector.reset()
            self.masker.enable() if current_split == "std" else self.masker.disable()
