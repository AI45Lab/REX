"""REX training of an expanded checkpoint: F0 -> F1.

The final F1 is saved at the output directory itself; optional token
milestones are saved under ``<output>/milestones/tokens-<N>/``. Every saved
checkpoint carries a ``run.json`` with the actual per-split token counts.
In-training lm-eval results go to ``<output>/train_eval/`` and diagnostic
probes to ``<output>/diagnostics/``, so they are not inherited as runtime
files by checkpoints derived from F1.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

from common.checkpoint_io import (
    cuda_empty_cache_if_available as _cuda_empty_cache_if_available,
)
from common.checkpoint_io import (
    cuda_synchronize_if_available as _cuda_synchronize_if_available,
)
from common.checkpoint_io import (
    record_token_milestone_metrics,
    reduce_tokens,
    write_final_validation_metrics,
)
from common.checkpoint_io import (
    runtime_source_dirs as _runtime_source_dirs,
)
from common.checkpoint_io import (
    save_checkpoint as _save_shared_checkpoint,
)
from methods.rex.config import (
    estimate_training_steps,
    parse_harmful_expert_lr_multiplier,
    parse_harmful_router_lr_multiplier,
    parse_harmful_router_std_step_lr_multiplier,
    parse_split_token_budgets,
    parse_rex_variant,
    parse_std_expert_lr_multiplier,
    parse_std_router_lr_multiplier,
    prepare_training_config,
)
from common.run_record import write_run_record
from moe.runtime import convert_packed_moe_to_expanded_runtime
from training.runtime import (
    apply_selective_activation_checkpointing,
    build_accelerator,
    build_scheduler,
    build_xtuner_train_dataloaders,
    load_model_and_tokenizer,
    model_decoder_classes,
    verify_finite_training_step,
)
from methods.rex.optim import build_rex_optimizer, validate_gradient_masker_matches
from methods.rex.masking import (
    ActivationMasker,
    GradientMasker,
    RoutingStatsCollector,
    set_rex_trainable_parameters,
)
from methods.rex.metrics import (
    LOSS_METRIC_FIELDS,
    build_train_metrics,
    finalize_layer_routing_stats,
    reduce_layer_routing_stats,
    reduce_loss_accumulator,
    rex_split_label,
    sum_layer_routing_stats,
)
from methods.rex.step import (
    compute_step_loss,
    quota_step_kind,
    run_step_forward,
)
from training.data import TrainBatchStream
from training.loop import TrainingFinalizer, optimizer_step
from training.setup import TrainingContext, build_training_context
from training.lm_eval_runner import LMEvalRunner


STEP_NAME = "train_rex"
MILESTONE_SUBDIR = "milestones"
TRAIN_EVAL_SUBDIR = "train_eval"
DIAGNOSTICS_SUBDIR = "diagnostics"
# Recipe keys filled from the command line; the run record keeps them in args.
INJECTED_PATH_KEYS = ("initial_checkpoint_dir", "tokenizer_dir", "output_dir")


@dataclass
class ThroughputWindow:
    start_time: float | None = None
    elapsed_time: float = 0.0
    steps: int = 0
    tokens: int = 0

    def needs_start(self) -> bool:
        return self.start_time is None

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def add_step(self, tokens: int) -> None:
        self.steps += 1
        self.tokens += int(tokens)

    def pause(self) -> None:
        if self.start_time is None:
            return
        self.elapsed_time += time.perf_counter() - self.start_time
        self.start_time = None

    def finish(self) -> tuple[float, float]:
        self.pause()
        elapsed = self.elapsed_time
        step_time = elapsed / self.steps if self.steps > 0 else 0.0
        tokens_per_sec = self.tokens / elapsed if elapsed > 0 else 0.0
        self.elapsed_time = 0.0
        self.steps = 0
        self.tokens = 0
        return step_time, tokens_per_sec


@dataclass(frozen=True)
class RoutingLossSettings:
    weight_std: float
    weight_harmful: float
    harmful_mass_floor: float
    std_mass_ceiling: float
    loss_type: str
    std_rank_weight: float
    std_rank_margin: float


@dataclass(frozen=True)
class TrainRunSchedule:
    save_token_milestones: tuple[int, ...]
    lm_eval_interval: int
    lm_eval_token_milestones: tuple[int, ...]
    final_validation: bool
    lm_eval_token_basis: str = "total"

    def eval_token_progress(
        self,
        tokens_seen: int,
        final_quota_target: int | None,
        split_tokens_seen: dict[str, int] | None,
        split_token_budgets: dict[str, int] | None,
    ) -> tuple[int, int | None]:
        if self.lm_eval_token_basis == "total":
            return tokens_seen, final_quota_target
        if (
            split_tokens_seen is None
            or split_token_budgets is None
            or self.lm_eval_token_basis not in split_tokens_seen
            or self.lm_eval_token_basis not in split_token_budgets
        ):
            raise ValueError("Split-based evaluation requires split token counters and budgets.")
        return (
            split_tokens_seen[self.lm_eval_token_basis],
            split_token_budgets[self.lm_eval_token_basis],
        )

    def ready_eval_token_milestones(
        self,
        tokens_seen: int,
        evaluated_milestones: set[int],
        *,
        final_quota_target: int,
        quota_complete: bool,
        split_tokens_seen: dict[str, int] | None,
        split_token_budgets: dict[str, int] | None,
    ) -> tuple[int, ...]:
        eval_tokens, eval_target = self.eval_token_progress(
            tokens_seen, final_quota_target, split_tokens_seen, split_token_budgets
        )
        return _ready_token_milestones(
            self.due_eval_token_milestones(eval_tokens, evaluated_milestones),
            final_quota_target=eval_target,
            quota_complete=quota_complete,
        )

    @property
    def needs_lm_eval_runner(self) -> bool:
        return (
            self.lm_eval_interval > 0
            or bool(self.lm_eval_token_milestones)
            or self.final_validation
        )

    def due_token_milestones(
        self,
        tokens_seen: int,
        saved_milestones: set[int],
    ) -> tuple[int, ...]:
        return tuple(
            milestone
            for milestone in self.save_token_milestones
            if milestone <= tokens_seen and milestone not in saved_milestones
        )

    def should_eval_step(self, step: int) -> bool:
        return self.lm_eval_interval > 0 and step % self.lm_eval_interval == 0

    def due_eval_token_milestones(
        self,
        tokens_seen: int,
        evaluated_milestones: set[int],
    ) -> tuple[int, ...]:
        return tuple(
            milestone
            for milestone in self.lm_eval_token_milestones
            if milestone <= tokens_seen and milestone not in evaluated_milestones
        )


@dataclass(frozen=True)
class CheckpointWriter:
    accelerator: Any
    tokenizer: Any
    checkpoint_dir: Path
    model: torch.nn.Module
    runtime_source_dirs: list[Path]
    init_dir: str
    recipe: dict[str, Any]
    args: dict[str, Any]

    @property
    def train_eval_dir(self) -> Path:
        return self.checkpoint_dir / TRAIN_EVAL_SUBDIR

    def save(self, output_dir: Path, results: dict[str, Any]) -> None:
        self.accelerator.print(f"[checkpoint] saving {output_dir}")
        _save_shared_checkpoint(
            accelerator=self.accelerator,
            tokenizer=self.tokenizer,
            model=self.model,
            output_dir=output_dir,
            source_dirs=self.runtime_source_dirs,
            merge_expanded_moe=True,
            write_record=lambda directory: write_run_record(
                directory,
                step=STEP_NAME,
                inputs={"f0": self.init_dir},
                config=self.recipe,
                args=self.args,
                results=results,
            ),
        )


def _routing_loss_settings(config: dict[str, Any]) -> RoutingLossSettings:
    variant = parse_rex_variant(config)
    without_loss = variant in ("gradient_mask_only", "without_routing_loss")
    return RoutingLossSettings(
        weight_std=0.0 if without_loss else float(config.get("routing_loss_weight_std", 0.0)),
        weight_harmful=0.0 if without_loss else float(config.get("routing_loss_weight_harmful", 0.1)),
        harmful_mass_floor=float(config.get("routing_harmful_mass_floor", 0.6)),
        std_mass_ceiling=float(config.get("routing_std_mass_ceiling", 0.4)),
        loss_type=str(config.get("routing_loss_type", "softplus")),
        std_rank_weight=0.0 if without_loss else float(config.get("routing_std_rank_loss_weight", 0.0)),
        std_rank_margin=float(config.get("routing_std_rank_margin", 0.0)),
    )


def parse_token_milestones(
    config: dict[str, Any],
    key: str,
    *,
    max_tokens: int | None = None,
) -> tuple[int, ...]:
    raw_token_milestones = config[key] if key in config else []
    if not isinstance(raw_token_milestones, list):
        raise ValueError(f"{key} must be a list of positive integers.")
    parsed_milestones: list[int] = []
    max_tokens = int(config["max_tokens"]) if max_tokens is None else max_tokens
    for milestone in raw_token_milestones:
        if (
            isinstance(milestone, bool)
            or not isinstance(milestone, int)
            or milestone <= 0
            or milestone > max_tokens
        ):
            raise ValueError(
                f"{key} must contain positive integers no greater than max_tokens."
            )
        if milestone in parsed_milestones:
            raise ValueError(f"{key} must not contain duplicates.")
        parsed_milestones.append(milestone)
    return tuple(sorted(parsed_milestones))


def _train_run_schedule(config: dict[str, Any]) -> TrainRunSchedule:
    lm_eval_interval = int(config.get("lm_eval_interval", 0))
    token_basis = config.get("lm_eval_token_basis", "total")
    if token_basis not in ("total", "std", "harmful"):
        raise ValueError("lm_eval_token_basis must be total, std or harmful.")
    eval_max_tokens = int(config["max_tokens"])
    if token_basis != "total":
        budgets = parse_split_token_budgets(config, max_tokens=eval_max_tokens)
        eval_max_tokens = budgets[token_basis]
    return TrainRunSchedule(
        save_token_milestones=parse_token_milestones(
            config,
            "save_token_milestones",
        ),
        lm_eval_interval=lm_eval_interval,
        lm_eval_token_milestones=parse_token_milestones(
            config,
            "lm_eval_token_milestones",
            max_tokens=eval_max_tokens,
        ),
        final_validation=bool(config.get("final_validation", True)),
        lm_eval_token_basis=token_basis,
    )


def _log_scheduler_setup(
    accelerator: Any, config: dict[str, Any], estimated_steps: int
) -> None:
    std_router_lr_multiplier = parse_std_router_lr_multiplier(config)
    std_expert_lr_multiplier = parse_std_expert_lr_multiplier(config)
    harmful_router_lr_multiplier = parse_harmful_router_lr_multiplier(config)
    harmful_router_std_step_lr_multiplier = parse_harmful_router_std_step_lr_multiplier(
        config
    )
    harmful_expert_lr_multiplier = parse_harmful_expert_lr_multiplier(config)
    accelerator.print(
        "[scheduler] "
        f"estimated_steps={estimated_steps} "
        f"warmup_steps={int(config.get('lr_warmup_steps', 0))} "
        f"base_lr={float(config['lr']):.3e} "
        f"std_router_lr={float(config['lr']) * std_router_lr_multiplier:.3e} "
        f"std_router_lr_multiplier={std_router_lr_multiplier:g} "
        f"std_expert_lr={float(config['lr']) * std_expert_lr_multiplier:.3e} "
        f"std_expert_lr_multiplier={std_expert_lr_multiplier:g} "
        f"harmful_router_lr={float(config['lr']) * harmful_router_lr_multiplier:.3e} "
        f"harmful_router_lr_multiplier={harmful_router_lr_multiplier:g} "
        f"harmful_router_std_step_lr={float(config['lr']) * harmful_router_lr_multiplier * harmful_router_std_step_lr_multiplier:.3e} "
        f"harmful_router_std_step_lr_multiplier={harmful_router_std_step_lr_multiplier:g} "
        f"harmful_expert_lr={float(config['lr']) * harmful_expert_lr_multiplier:.3e} "
        f"harmful_expert_lr_multiplier={harmful_expert_lr_multiplier:g} "
        f"min_lr={float(config.get('min_lr', 0.0)):.3e} "
        "step_scheduler_with_optimizer=False"
    )


def _prepare_model_runtime(
    *,
    accelerator: Any,
    model: torch.nn.Module,
    config: dict[str, Any],
    effective_global_batch_size: int,
) -> tuple[torch.nn.Module, Any, Any]:
    optimizer = build_rex_optimizer(model, config)
    estimated_steps = estimate_training_steps(
        config,
        effective_global_batch_size=effective_global_batch_size,
    )
    scheduler = build_scheduler(optimizer, config, estimated_steps)
    _log_scheduler_setup(accelerator, config, estimated_steps)
    return accelerator.prepare(model, optimizer, scheduler)


def apply_split_lr_multiplier(
    optimizer: Any,
    *,
    config: dict[str, Any],
    step_kind: str,
) -> float:
    """Temporarily scale the harmful-router bank on a ``std`` step.

    ``LambdaLR`` owns each group's scheduled base rate and rewrites it on the
    following scheduler step.  The split-only adjustment therefore surrounds
    only ``optimizer.step`` and is restored before ``scheduler.step``.
    """
    if step_kind != "std":
        return 1.0
    multiplier = parse_harmful_router_std_step_lr_multiplier(config)
    if multiplier == 1.0:
        return 1.0
    for group in optimizer.param_groups:
        if group.get("rex_bank") == "harmful_router":
            group["lr"] *= multiplier
    return multiplier


def restore_split_lr_multiplier(optimizer: Any, *, multiplier: float) -> None:
    if multiplier == 1.0:
        return
    for group in optimizer.param_groups:
        if group.get("rex_bank") == "harmful_router":
            group["lr"] /= multiplier


def _apply_selective_checkpointing(
    *,
    accelerator: Any,
    model: torch.nn.Module,
) -> None:
    checkpointed_layers, skipped_layers = apply_selective_activation_checkpointing(
        model,
        transformer_cls_names=model_decoder_classes(model),
        checkpoint_stride=1,
    )
    accelerator.print(
        "[fsdp] selective activation checkpointing "
        f"checkpointed_layers={checkpointed_layers} skipped_layers={skipped_layers}"
    )


def _set_activation_mask_for_step(
    activation_masker: ActivationMasker, step_kind: str, variant: str = "standard"
) -> None:
    if step_kind == "std" and variant not in ("gradient_mask_only", "without_std_activation_mask"):
        activation_masker.enable()
    else:
        activation_masker.disable()


def _should_collect_routing_stats(
    *,
    step_kind: str,
    logging_enabled: bool,
    routing_loss_weight_std: float,
    routing_loss_weight_harmful: float,
) -> bool:
    return (
        logging_enabled
        or (step_kind == "std" and routing_loss_weight_std != 0.0)
        or (step_kind == "harmful" and routing_loss_weight_harmful != 0.0)
    )


def _ready_token_milestones(
    due_milestones: tuple[int, ...],
    *,
    final_quota_target: int,
    quota_complete: bool,
) -> tuple[int, ...]:
    return tuple(
        milestone
        for milestone in due_milestones
        if milestone != final_quota_target or quota_complete
    )


def _maybe_save_token_checkpoints(
    *,
    schedule: TrainRunSchedule,
    checkpoint_writer: CheckpointWriter,
    activation_masker: ActivationMasker,
    tokens_seen: int,
    saved_milestones: set[int],
    final_quota_target: int,
    quota_complete: bool,
    results: Any,
) -> None:
    due_milestones = schedule.due_token_milestones(
        tokens_seen,
        saved_milestones,
    )
    for milestone in _ready_token_milestones(
        due_milestones,
        final_quota_target=final_quota_target,
        quota_complete=quota_complete,
    ):
        activation_masker.disable()
        checkpoint_writer.save(
            checkpoint_writer.checkpoint_dir / MILESTONE_SUBDIR / f"tokens-{milestone:012d}",
            {"milestone_tokens": milestone, **results()},
        )
        saved_milestones.add(milestone)


def training_is_complete(
    *,
    split_token_budgets: dict[str, int],
    split_tokens_seen: dict[str, int],
) -> bool:
    return all(
        split_tokens_seen[split_name] >= budget
        for split_name, budget in split_token_budgets.items()
    )


def run_results(run: "RexRun") -> dict[str, Any]:
    """Actual counts of the model state being saved, for its ``run.json``."""
    elapsed = time.perf_counter() - run.training_start_time
    budgets = run.ctx.setup.split_token_budgets
    return {
        "tokens_seen": run.tokens_seen,
        "split_token_budgets": dict(budgets),
        "split_tokens_seen": dict(run.split_tokens_seen),
        "split_token_overshoot": {
            name: max(run.split_tokens_seen[name] - budget, 0)
            for name, budget in budgets.items()
        },
        "split_optimizer_updates": dict(run.split_updates),
        "optimizer_steps": run.current_step,
        "train_elapsed_seconds": elapsed,
        "gpu_hours": elapsed * run.ctx.accelerator.num_processes / 3600.0,
        "tokens_per_second": run.tokens_seen / elapsed if elapsed > 0 else 0.0,
    }


def maybe_run_step_validation(
    *,
    schedule: TrainRunSchedule,
    lm_eval_runner: LMEvalRunner | None,
    activation_masker: ActivationMasker,
    current_step: int,
    log_step: int,
) -> dict[str, float] | None:
    if lm_eval_runner is None or not schedule.should_eval_step(current_step):
        return None
    activation_masker.disable()
    metrics = lm_eval_runner.run_validation(step=log_step)
    _cuda_empty_cache_if_available()
    return metrics


def maybe_run_token_validation(
    *,
    schedule: TrainRunSchedule,
    lm_eval_runner: LMEvalRunner | None,
    activation_masker: ActivationMasker,
    checkpoint_dir: Path,
    tokens_seen: int,
    evaluated_milestones: set[int],
    final_quota_target: int,
    quota_complete: bool,
    log_step: int,
    validation_metrics: dict[str, float] | None = None,
    split_tokens_seen: dict[str, int] | None = None,
    split_token_budgets: dict[str, int] | None = None,
) -> None:
    if lm_eval_runner is None:
        return
    due_milestones = schedule.ready_eval_token_milestones(
        tokens_seen,
        evaluated_milestones,
        final_quota_target=final_quota_target,
        quota_complete=quota_complete,
        split_tokens_seen=split_tokens_seen,
        split_token_budgets=split_token_budgets,
    )
    if not due_milestones:
        return

    metrics = validation_metrics
    if metrics is None:
        activation_masker.disable()
        metrics = lm_eval_runner.run_validation(step=log_step)
        _cuda_empty_cache_if_available()

    for milestone in due_milestones:
        record_token_milestone_metrics(
            checkpoint_dir=checkpoint_dir,
            milestone=milestone,
            tokens_seen=tokens_seen,
            log_step=log_step,
            metrics=metrics,
            token_basis=schedule.lm_eval_token_basis,
            split_tokens_seen=split_tokens_seen,
        )
        evaluated_milestones.add(milestone)


def has_due_checkpoint_or_validation(
    *,
    schedule: TrainRunSchedule,
    has_lm_eval_runner: bool,
    current_step: int,
    tokens_seen: int,
    saved_milestones: set[int],
    evaluated_milestones: set[int],
    final_quota_target: int,
    quota_complete: bool,
    split_tokens_seen: dict[str, int] | None = None,
    split_token_budgets: dict[str, int] | None = None,
) -> bool:
    due_save_milestones = _ready_token_milestones(
        schedule.due_token_milestones(tokens_seen, saved_milestones),
        final_quota_target=final_quota_target,
        quota_complete=quota_complete,
    )
    due_eval_milestones = schedule.ready_eval_token_milestones(
        tokens_seen,
        evaluated_milestones,
        final_quota_target=final_quota_target,
        quota_complete=quota_complete,
        split_tokens_seen=split_tokens_seen,
        split_token_budgets=split_token_budgets,
    )
    return (
        bool(due_save_milestones)
        or (
            has_lm_eval_runner
            and (schedule.should_eval_step(current_step) or bool(due_eval_milestones))
        )
    )


def run_final_outputs(
    *,
    schedule: TrainRunSchedule,
    checkpoint_writer: CheckpointWriter,
    lm_eval_runner: LMEvalRunner | None,
    log_step: int,
    results: Any,
) -> None:
    if lm_eval_runner is not None and schedule.final_validation:
        metrics = lm_eval_runner.run_validation(step=log_step, lm_eval_limit=None)
        result_path = write_final_validation_metrics(
            checkpoint_dir=checkpoint_writer.train_eval_dir,
            log_step=log_step,
            metrics=metrics,
        )
        if result_path is not None:
            checkpoint_writer.accelerator.print(
                f"final lm-eval metrics written to {result_path}"
            )
    # The final checkpoint is the run output itself.
    checkpoint_writer.save(checkpoint_writer.checkpoint_dir, results())


def log_train_step(
    *,
    accelerator: Any,
    scheduler: Any,
    throughput_window: ThroughputWindow,
    loss_accumulators: dict[str, dict[str, list[torch.Tensor]]],
    per_split_layer_routing_stats: dict[
        str,
        list[dict[str, torch.Tensor]] | None,
    ],
    current_step_kind: str,
    current_step: int,
    current_iter: int,
    tokens_seen: int,
    split_token_budgets: dict[str, int],
    split_tokens_seen: dict[str, int],
    split_updates: dict[str, int],
    training_start_time: float,
    log_step: int,
) -> None:
    _cuda_synchronize_if_available()
    step_time, tokens_per_sec = throughput_window.finish()
    train_elapsed_seconds = time.perf_counter() - training_start_time
    gpu_hours = train_elapsed_seconds * accelerator.num_processes / 3600.0

    active_splits = [
        split_kind
        for split_kind, accumulator in loss_accumulators.items()
        if any(accumulator.values())
    ]
    ordered_splits = [
        split_kind for split_kind in active_splits if split_kind != current_step_kind
    ]
    if current_step_kind in active_splits:
        ordered_splits.append(current_step_kind)

    train_metrics: dict[str, float] = {}
    loss_values_by_split: dict[str, dict[str, float]] = {}
    for split_kind in ordered_splits:
        reduced_layer_stats = reduce_layer_routing_stats(
            accelerator,
            per_split_layer_routing_stats[split_kind] or [],
        )
        routing_summary = finalize_layer_routing_stats(reduced_layer_stats)
        loss_values = reduce_loss_accumulator(
            accelerator,
            loss_accumulators[split_kind],
        )
        loss_values_by_split[split_kind] = loss_values
        train_metrics.update(
            build_train_metrics(
                step_kind=split_kind,
                current_step=current_step,
                current_iter=current_iter,
                tokens_seen=tokens_seen,
                split_token_budgets=split_token_budgets,
                split_tokens_seen=split_tokens_seen,
                split_updates=split_updates,
                learning_rate=scheduler.get_last_lr()[0],
                loss_values=loss_values,
                routing_summary=routing_summary,
                step_time=step_time,
                tokens_per_sec=tokens_per_sec,
                train_elapsed_seconds=train_elapsed_seconds,
                gpu_hours=gpu_hours,
            )
        )
    accelerator.log(train_metrics, step=log_step)
    split_loss_summary = " ".join(
        f"{rex_split_label(split_kind)}:"
        f"loss={loss_values['total_loss']:.4f},"
        f"lm={loss_values['lm_loss']:.4f}"
        for split_kind, loss_values in loss_values_by_split.items()
    )
    accelerator.print(
        "train "
        f"step={current_step} splits=[{split_loss_summary}] "
        f"tokens={tokens_seen} "
        f"split_tokens=std:{split_tokens_seen['std']},"
        f"harmful:{split_tokens_seen['harmful']} "
        f"tokens_per_sec={tokens_per_sec:.1f} gpu_hours={gpu_hours:.3f}"
    )


@dataclass
class RexRun:
    ctx: TrainingContext | None = None
    routing_settings: RoutingLossSettings | None = None
    expanded_moe_summary: Any = None
    harmful_experts: list[int] | None = None
    routing_collector: RoutingStatsCollector | None = None
    activation_masker: ActivationMasker | None = None
    gradient_masker: GradientMasker | None = None
    checkpoint_writer: CheckpointWriter | None = None
    recipe: dict[str, Any] | None = None
    args: dict[str, Any] | None = None
    finalizer: TrainingFinalizer | None = None
    train_batch_stream: TrainBatchStream | None = None
    schedule: TrainRunSchedule | None = None
    lm_eval_runner: LMEvalRunner | None = None
    log_interval: int = 0
    max_norm: float = 0
    tokens_seen: int = 0
    split_tokens_seen: dict[str, int] | None = None
    split_updates: dict[str, int] | None = None
    current_step: int = 0
    log_step: int = 0
    saved_token_milestones: set[int] = None
    evaluated_token_milestones: set[int] = None
    throughput_window: ThroughputWindow | None = None
    training_start_time: float = 0
    logging_enabled: bool = False
    loss_accumulators: Any = None
    per_split_layer_routing_stats: dict[str, list[dict[str, torch.Tensor]] | None] = (
        None
    )
    final_quota_target: int = 0
    diagnostics: Any = None
    current_iter: int = 0
    should_log_step: bool = False
    step_kind: str | None = None
    step_local_tokens: int = 0
    diagnostic_before: Any = None
    quota_complete: bool = False


def configure_rex_model(run, ctx):
    run.ctx = ctx
    run.recipe = copy.deepcopy(
        {key: value for key, value in ctx.config.items() if key not in INJECTED_PATH_KEYS}
    )
    run.routing_settings = _routing_loss_settings(run.ctx.config)

    run.expanded_moe_summary = convert_packed_moe_to_expanded_runtime(
        run.ctx.model,
        initial_checkpoint_dir=run.ctx.config["initial_checkpoint_dir"],
    )
    set_rex_trainable_parameters(run.ctx.model, run.ctx.config)
    run.harmful_experts = list(run.expanded_moe_summary.harmful_experts)
    run.ctx.accelerator.print(
        "[expanded-moe] converted packed checkpoint to expanded MoE runtime "
        f"converted_layers={run.expanded_moe_summary.converted_layers} "
        f"old_num_experts={run.expanded_moe_summary.old_num_experts} "
        f"num_harmful_experts={run.expanded_moe_summary.num_harmful_experts} "
        f"new_num_experts={run.expanded_moe_summary.new_num_experts}"
    )
    run.routing_collector = RoutingStatsCollector(run.ctx.model, run.harmful_experts)
    run.activation_masker = ActivationMasker(run.ctx.model, run.harmful_experts)


def prepare_rex_optimizer(ctx):
    return _prepare_model_runtime(
        accelerator=ctx.accelerator,
        model=ctx.model,
        config=ctx.config,
        effective_global_batch_size=ctx.setup.effective_global_batch_size,
    )


def finish_rex_setup(run):
    _apply_selective_checkpointing(
        accelerator=run.ctx.accelerator,
        model=run.ctx.model,
    )

    run.gradient_masker = GradientMasker(
        run.ctx.accelerator.unwrap_model(run.ctx.model),
    )
    validate_gradient_masker_matches(
        run.ctx.accelerator,
        run.gradient_masker,
        require_expert_params=True,
    )
    run.ctx.accelerator.print(
        "[gradient-mask] shared parameters are frozen; both routers and the split-selected expert bank are trainable."
    )

    runtime_source_dirs = _runtime_source_dirs(run.ctx.config)
    run.checkpoint_writer = CheckpointWriter(
        accelerator=run.ctx.accelerator,
        tokenizer=run.ctx.tokenizer,
        checkpoint_dir=run.ctx.setup.checkpoint_dir,
        model=run.ctx.model,
        runtime_source_dirs=runtime_source_dirs,
        init_dir=run.ctx.config["initial_checkpoint_dir"],
        recipe=run.recipe,
        args=run.args,
    )

    run.ctx.init_trackers(
        {
            "gradient_accumulation_steps": run.ctx.setup.gradient_accumulation_steps,
            "effective_global_batch_size": run.ctx.setup.effective_global_batch_size,
            "checkpoint_dir": str(run.ctx.setup.checkpoint_dir),
        }
    )
    run.finalizer = TrainingFinalizer(
        run.ctx.accelerator,
        (
            run.activation_masker,
            run.routing_collector,
        ),
    )


def initialize_rex_run(run):
    non_blocking_batch_transfer = bool(run.ctx.config.get("pin_memory", True))
    train_loaders = {"std": run.ctx.loaders[0], "harmful": run.ctx.loaders[1]}
    run.train_batch_stream = TrainBatchStream(
        loaders=train_loaders,
        device=run.ctx.accelerator.device,
        non_blocking=non_blocking_batch_transfer,
    )

    run.schedule = _train_run_schedule(run.ctx.config)
    run.lm_eval_runner = (
        LMEvalRunner(
            accelerator=run.ctx.accelerator,
            model=run.ctx.model,
            tokenizer=run.ctx.tokenizer,
            activation_masker=run.activation_masker,
            config=run.ctx.config,
            routing_collector=run.routing_collector,
        )
        if run.schedule.needs_lm_eval_runner
        or run.ctx.config.get("diagnostic_initial_eval", False)
        else None
    )

    run.log_interval = int(run.ctx.config.get("log_interval", 1))
    run.max_norm = float(run.ctx.config.get("max_norm", 0.0))
    run.ctx.accelerator.print(
        f"[split-schedule] budgets={run.ctx.setup.split_token_budgets}"
    )

    run.ctx.model.train()
    run.tokens_seen = 0
    run.split_tokens_seen = {"std": 0, "harmful": 0}
    run.split_updates = {"std": 0, "harmful": 0}
    run.current_step = 0
    run.log_step = 0
    run.saved_token_milestones: set[int] = set()
    run.evaluated_token_milestones: set[int] = set()
    run.throughput_window = ThroughputWindow()
    run.training_start_time = time.perf_counter()
    run.logging_enabled = run.log_interval > 0
    run.loss_accumulators = {
        split_kind: {key: [] for key in LOSS_METRIC_FIELDS}
        for split_kind in run.split_tokens_seen
    }
    run.per_split_layer_routing_stats: dict[
        str,
        list[dict[str, torch.Tensor]] | None,
    ] = {split_kind: None for split_kind in run.split_tokens_seen}
    run.final_quota_target = sum(run.ctx.setup.split_token_budgets.values())

    run.diagnostics = None
    if run.ctx.config.get("diagnostic_probe_steps"):
        from methods.rex.diagnostics import RexDiagnostics

        run.diagnostics = RexDiagnostics(
            model=run.ctx.model,
            accelerator=run.ctx.accelerator,
            config=run.ctx.config,
            output_dir=run.ctx.setup.checkpoint_dir / DIAGNOSTICS_SUBDIR,
            activation_masker=run.activation_masker,
            routing_collector=run.routing_collector,
        )

    if run.ctx.config.get("diagnostic_initial_eval", False):
        run.activation_masker.disable()
        initial_metrics = run.lm_eval_runner.run_validation(step=0)
        record_token_milestone_metrics(
            checkpoint_dir=run.checkpoint_writer.train_eval_dir,
            milestone=0,
            tokens_seen=0,
            log_step=0,
            metrics=initial_metrics,
            token_basis=run.schedule.lm_eval_token_basis,
            split_tokens_seen=run.split_tokens_seen,
        )
        _cuda_empty_cache_if_available()


def forward_loss(run, batch, step_kind, collect_routing_stats):
    run.routing_collector.reset()
    run.routing_collector.set_collect_stats(collect_routing_stats)
    run.routing_collector.set_attention_mask(batch.get("attention_mask"))

    outputs = run_step_forward(model=run.ctx.model, batch=batch)
    loss_result = compute_step_loss(
        step_kind=step_kind,
        batch=batch,
        outputs=outputs,
        routing_collector=run.routing_collector,
        loss_chunk_size=run.ctx.setup.loss_chunk_size,
        routing_harmful_mass_floor=run.routing_settings.harmful_mass_floor,
        routing_std_mass_ceiling=run.routing_settings.std_mass_ceiling,
        routing_loss_type=run.routing_settings.loss_type,
        routing_loss_weight_std=run.routing_settings.weight_std,
        routing_loss_weight_harmful=run.routing_settings.weight_harmful,
        routing_std_rank_loss_weight=run.routing_settings.std_rank_weight,
        routing_std_rank_margin=run.routing_settings.std_rank_margin,
    )
    return outputs, loss_result


def train_microbatches(run):
    for _ in range(run.ctx.setup.gradient_accumulation_steps):
        batch, local_batch_tokens = run.train_batch_stream.next(
            run.step_kind
        )
        run.step_local_tokens += int(local_batch_tokens)
        if run.diagnostics is not None:
            run.diagnostics.remember(run.step_kind, batch)
        with run.ctx.accelerator.accumulate(run.ctx.model):
            collect_routing_stats = _should_collect_routing_stats(
                step_kind=run.step_kind,
                logging_enabled=run.logging_enabled,
                routing_loss_weight_std=run.routing_settings.weight_std,
                routing_loss_weight_harmful=run.routing_settings.weight_harmful,
            )
            outputs, loss_result = forward_loss(
                run, batch, run.step_kind, collect_routing_stats
            )
            run.ctx.accelerator.backward(loss_result.loss)
            logged_layer_routing_stats = (
                run.routing_collector.layer_stats(loss_result.lm_loss)
                if run.logging_enabled
                else None
            )
            if run.logging_enabled:
                for metric_name, attr_name in LOSS_METRIC_FIELDS.items():
                    run.loss_accumulators[run.step_kind][metric_name].append(
                        getattr(loss_result, attr_name).detach()
                    )
                run.per_split_layer_routing_stats[run.step_kind] = (
                    sum_layer_routing_stats(
                        run.per_split_layer_routing_stats[run.step_kind],
                        logged_layer_routing_stats or [],
                    )
                )

            run.gradient_masker.mask_non_target_parameters(run.step_kind)
            if run.ctx.config.get("verify_finite_training_step", False):
                verify_finite_training_step(
                    run.ctx.model, loss_result.loss, run.ctx.accelerator
                )
                run.ctx.accelerator.print(
                    f"[acceptance] split={run.step_kind} finite_loss_and_gradients=True"
                )
            optimizer_step(
                run.ctx,
                clip=run.max_norm,
                before_step=lambda: apply_split_lr_multiplier(
                    run.ctx.optimizer, config=run.ctx.config, step_kind=run.step_kind
                ),
                after_step=lambda multiplier: restore_split_lr_multiplier(
                    run.ctx.optimizer, multiplier=multiplier
                ),
                after_clip=(
                    lambda norm: run.diagnostics.record_clipping(
                        norm,
                        max_norm=run.max_norm,
                        step=run.current_step,
                        split=run.step_kind,
                    )
                )
                if run.diagnostics is not None
                else None,
            )


def log_rex_step(run):
    if run.should_log_step:
        log_train_step(
            accelerator=run.ctx.accelerator,
            scheduler=run.ctx.scheduler,
            throughput_window=run.throughput_window,
            loss_accumulators=run.loss_accumulators,
            per_split_layer_routing_stats=run.per_split_layer_routing_stats,
            current_step_kind=run.step_kind,
            current_step=run.current_step,
            current_iter=run.current_iter,
            tokens_seen=run.tokens_seen,
            split_token_budgets=run.ctx.setup.split_token_budgets,
            split_tokens_seen=run.split_tokens_seen,
            split_updates=run.split_updates,
            training_start_time=run.training_start_time,
            log_step=run.log_step,
        )
        run.loss_accumulators = {
            split_kind: {key: [] for key in LOSS_METRIC_FIELDS}
            for split_kind in run.split_tokens_seen
        }
        run.per_split_layer_routing_stats = {
            split_kind: None for split_kind in run.split_tokens_seen
        }


def checkpoint_and_evaluate(run):
    run.quota_complete = training_is_complete(
        split_token_budgets=run.ctx.setup.split_token_budgets,
        split_tokens_seen=run.split_tokens_seen,
    )
    if run.diagnostics is not None:
        run.diagnostics.probe(
            step=run.current_step,
            current_split=run.step_kind,
            forward_loss=partial(forward_loss, run),
            final=run.quota_complete,
        )
    if has_due_checkpoint_or_validation(
        schedule=run.schedule,
        has_lm_eval_runner=run.lm_eval_runner is not None,
        current_step=run.current_step,
        tokens_seen=run.tokens_seen,
        saved_milestones=run.saved_token_milestones,
        evaluated_milestones=run.evaluated_token_milestones,
        final_quota_target=run.final_quota_target,
        quota_complete=run.quota_complete,
        split_tokens_seen=run.split_tokens_seen,
        split_token_budgets=run.ctx.setup.split_token_budgets,
    ):
        _cuda_synchronize_if_available()
        run.throughput_window.pause()

    _maybe_save_token_checkpoints(
        schedule=run.schedule,
        checkpoint_writer=run.checkpoint_writer,
        activation_masker=run.activation_masker,
        tokens_seen=run.tokens_seen,
        saved_milestones=run.saved_token_milestones,
        final_quota_target=run.final_quota_target,
        quota_complete=run.quota_complete,
        results=partial(run_results, run),
    )
    step_validation_metrics = maybe_run_step_validation(
        schedule=run.schedule,
        lm_eval_runner=run.lm_eval_runner,
        activation_masker=run.activation_masker,
        current_step=run.current_step,
        log_step=run.log_step,
    )
    maybe_run_token_validation(
        schedule=run.schedule,
        lm_eval_runner=run.lm_eval_runner,
        activation_masker=run.activation_masker,
        checkpoint_dir=run.checkpoint_writer.train_eval_dir,
        tokens_seen=run.tokens_seen,
        evaluated_milestones=run.evaluated_token_milestones,
        final_quota_target=run.final_quota_target,
        quota_complete=run.quota_complete,
        log_step=run.log_step,
        validation_metrics=step_validation_metrics,
        split_tokens_seen=run.split_tokens_seen,
        split_token_budgets=run.ctx.setup.split_token_budgets,
    )


def train_step(run):
    should_start_throughput_window = (
        run.log_interval > 0 and run.throughput_window.needs_start()
    )
    if should_start_throughput_window:
        _cuda_synchronize_if_available()
        run.throughput_window.start()

    run.current_step += 1
    run.current_iter = run.current_step * run.ctx.setup.gradient_accumulation_steps
    run.log_step = run.current_step
    run.should_log_step = (
        run.log_interval > 0 and run.current_step % run.log_interval == 0
    )
    run.step_kind = quota_step_kind(
        run.ctx.setup.split_token_budgets, run.split_tokens_seen
    )
    _set_activation_mask_for_step(run.activation_masker, run.step_kind, parse_rex_variant(run.ctx.config))

    run.diagnostic_before = (
        run.diagnostics.capture_parameters() if run.diagnostics is not None else None
    )
    run.step_local_tokens = 0

    train_microbatches(run)

    if run.diagnostics is not None:
        run.diagnostics.record_update(
            run.diagnostic_before, step=run.current_step, split=run.step_kind
        )
        del run.diagnostic_before

    step_tokens = reduce_tokens(run.ctx.accelerator, run.step_local_tokens)
    run.throughput_window.add_step(step_tokens)
    run.tokens_seen += step_tokens
    run.split_tokens_seen[run.step_kind] += step_tokens
    run.split_updates[run.step_kind] += 1

    log_rex_step(run)
    checkpoint_and_evaluate(run)


def finish_training(run):
    # A final std step leaves harmful experts masked. Final outputs and the
    # pre-save reload probe must describe the full expanded checkpoint.
    run.activation_masker.disable()
    run_final_outputs(
        schedule=run.schedule,
        checkpoint_writer=run.checkpoint_writer,
        lm_eval_runner=run.lm_eval_runner,
        log_step=run.log_step,
        results=partial(run_results, run),
    )


def run_training(config_path: Path, *, init_dir: Path, output_dir: Path) -> None:
    run = RexRun()
    run.args = {
        "config": str(config_path.resolve()),
        "init": str(init_dir.resolve()),
        "output": str(output_dir.resolve()),
    }
    ctx = build_training_context(
        config_path,
        paths={
            "initial_checkpoint_dir": str(init_dir),
            "tokenizer_dir": str(init_dir),
            "output_dir": str(output_dir),
        },
        prepare_fn=prepare_training_config,
        build_dataloaders=build_xtuner_train_dataloaders,
        accelerator_factory=build_accelerator,
        model_loader=load_model_and_tokenizer,
        configure_model=partial(configure_rex_model, run),
        prepare_model=prepare_rex_optimizer,
        needs_reference=False,
        finish_setup=lambda ctx: finish_rex_setup(run),
    )
    try:
        initialize_rex_run(run)
        while not training_is_complete(
            split_token_budgets=ctx.setup.split_token_budgets,
            split_tokens_seen=run.split_tokens_seen,
        ):
            train_step(run)
        finish_training(run)
    finally:
        run.finalizer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an expanded checkpoint with REX (F0 -> F1).")
    parser.add_argument("--config", required=True, type=Path, help="REX recipe YAML.")
    parser.add_argument("--init", required=True, type=Path, help="Expanded F0 checkpoint (also the tokenizer).")
    parser.add_argument("--output", required=True, type=Path, help="New directory for F1.")
    args = parser.parse_args()
    run_training(args.config, init_dir=args.init, output_dir=args.output)


if __name__ == "__main__":
    main()
