"""RMU on a standard checkpoint: A1 -> A2 and, with the same recipe, B -> R.

The retain reference is always the parent itself. The final checkpoint is saved
at the output directory; optional forget-token milestones are saved under
``<output>/milestones/forget-tokens-<N>/``. Each saved checkpoint carries a
``run.json`` with the actual forget and retain token counts.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from common.checkpoint_io import (
    reduce_tokens as _reduce_tokens,
)
from common.checkpoint_io import (
    runtime_source_dirs as _runtime_source_dirs,
)
from common.checkpoint_io import (
    save_checkpoint as _save_shared_checkpoint,
)
from common.run_record import write_run_record
from training.config import effective_global_batch_size_for, load_config
from moe.model_adapters import parameter_role
from methods.rmu.loss import (
    RmuActivationRecorder,
    resolve_layer_module,
    RmuControlVectorCache,
    compute_rmu_loss,
    compute_rmu_retain_loss,
    normalize_rmu_loss_compute_dtype,
    normalize_rmu_loss_reduction,
    rmu_token_mask_from_batch,
)
from training.runtime import (
    apply_selective_activation_checkpointing,
    build_accelerator,
    build_adamw,
    build_scheduler,
    build_xtuner_train_dataloaders,
    load_model_and_tokenizer,
    SKIP_ACTIVATION_CHECKPOINT_ATTR,
    default_parameter_groups,
    model_decoder_classes,
    verify_finite_training_step,
)
from training.data import PairedBatchStream
from training.loop import (
    TrainingFinalizer,
    optimizer_step,
)
from training.setup import build_training_context, resolve_output_dir
from training.config import positive_int as _positive_int

TRAINABLE_SCOPE_STD_DOWN_PROJ = "selected_std_expert_down_proj_only"
TRAINABLE_SCOPE_MOE_MLP = "selected_routed_and_shared_expert_mlp"
BUDGET_FORGET_TOKENS = "forget_tokens"
BUDGET_FORGET_EPOCHS = "forget_epochs"
MILESTONES_KEY = "save_forget_token_milestones"
MILESTONE_SUBDIR = "milestones"
_TRAINABLE_SCOPES = {
    TRAINABLE_SCOPE_STD_DOWN_PROJ,
    TRAINABLE_SCOPE_MOE_MLP,
}


@dataclass(frozen=True)
class RmuSetup:
    run_name: str
    checkpoint_dir: Path
    budget_unit: str
    max_forget_tokens: int | None
    max_forget_epochs: int | None
    micro_batch_size: int
    effective_global_batch_size: int
    gradient_accumulation_steps: int
    rmu_monitor_layer_indices: tuple[int, ...]
    rmu_trainable_layer_indices: tuple[int, ...]
    rmu_trainable_scope: str
    rmu_module_template: str | None
    selective_activation_checkpointing: bool


@dataclass(frozen=True)
class RmuSettings:
    forget_loss_weight: float
    retain_loss_weight: float
    token_mask_mode: str
    max_tokens_per_batch: int | None
    loss_reduction: str
    loss_compute_dtype: str
    target_seed: int
    steering_coeff: float


@dataclass(frozen=True)
class RmuSchedule:
    save_forget_token_milestones: tuple[int, ...]

    def due(self, forget_tokens_seen: int, saved: set[int]) -> tuple[int, ...]:
        return tuple(
            milestone
            for milestone in self.save_forget_token_milestones
            if milestone <= forget_tokens_seen and milestone not in saved
        )


@dataclass
class RmuStepLoss:
    loss: torch.Tensor
    forget_loss: torch.Tensor
    retain_loss: torch.Tensor
    weighted_forget_loss: torch.Tensor
    weighted_retain_loss: torch.Tensor
    forget_activation_norm: torch.Tensor
    forget_target_norm: torch.Tensor
    forget_cosine: torch.Tensor
    forget_tokens: torch.Tensor
    retain_activation_norm: torch.Tensor
    retain_frozen_activation_norm: torch.Tensor
    retain_cosine: torch.Tensor
    retain_tokens: torch.Tensor



def mark_rmu_layers_to_skip_activation_checkpointing(
    model: torch.nn.Module,
    layer_indices: list[int],
    *,
    module_template: str | None,
) -> None:
    for layer_idx in layer_indices:
        target_module = resolve_layer_module(model, layer_idx, module_template=module_template)
        setattr(target_module, SKIP_ACTIVATION_CHECKPOINT_ATTR, True)

        # Selective activation checkpointing wraps decoder layers rather than
        # arbitrary nested RMU targets such as ``layer.mlp``. Mark the owning
        # decoder layer as well so a nested hook keeps the same no-checkpoint
        # semantics as a decoder-layer hook.
        decoder_layer = resolve_layer_module(model, layer_idx, module_template=None)
        setattr(decoder_layer, SKIP_ACTIVATION_CHECKPOINT_ATTR, True)

def prepare_rmu_config(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    world_size: int,
) -> tuple[RmuSetup, RmuSettings, RmuSchedule]:
    required = {
        "initial_checkpoint_dir",
        "tokenizer_dir",
        "rmu_retain_reference_checkpoint_dir",
        "target_domain",
        "datasets",
        "max_seq_length",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "lr",
        "rmu_monitor_layer_indices",
        "rmu_trainable_layer_indices",
        "rmu_trainable_scope",
        "selective_activation_checkpointing",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"RMU config is missing required keys: {missing}")
    budget_unit, max_forget_tokens, max_forget_epochs = rmu_budget(config)
    micro_batch_size = _positive_int(
        config["micro_batch_size"], name="micro_batch_size"
    )
    gradient_accumulation_steps = _positive_int(
        config["gradient_accumulation_steps"],
        name="gradient_accumulation_steps",
    )
    _positive_int(world_size, name="world_size")
    (
        selective_activation_checkpointing,
        monitor_layer_indices,
        trainable_layer_indices,
        trainable_scope,
        module_template,
    ) = rmu_scope(config)
    schedule = rmu_schedule(config, max_forget_tokens)
    run_name, checkpoint_dir = resolve_output_dir(config)
    setup = RmuSetup(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        budget_unit=budget_unit,
        max_forget_tokens=max_forget_tokens,
        max_forget_epochs=max_forget_epochs,
        micro_batch_size=micro_batch_size,
        effective_global_batch_size=effective_global_batch_size_for(
            micro_batch_size=micro_batch_size,
            world_size=world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        gradient_accumulation_steps=gradient_accumulation_steps,
        rmu_monitor_layer_indices=monitor_layer_indices,
        rmu_trainable_layer_indices=trainable_layer_indices,
        rmu_trainable_scope=trainable_scope,
        rmu_module_template=module_template,
        selective_activation_checkpointing=selective_activation_checkpointing,
    )
    settings = rmu_settings(config)
    return setup, settings, schedule


def _layer_indices(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = config[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"RMU {key} must be a non-empty list.")
    parsed: list[int] = []
    for layer in value:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"RMU {key} must contain only non-negative integers.")
        parsed.append(int(layer))
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"RMU {key} must not contain duplicates.")
    return tuple(parsed)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return parsed


def select_rmu_trainable_parameters(
    model: torch.nn.Module,
    layer_indices: Sequence[int],
    trainable_scope: str,
) -> list[str]:
    if trainable_scope not in _TRAINABLE_SCOPES:
        raise ValueError(f"Unknown RMU trainable scope: {trainable_scope!r}.")
    target_layers = set(int(layer) for layer in layer_indices)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[str] = []
    for name, parameter in model.named_parameters():
        role = parameter_role(model, name)
        if role.layer not in target_layers:
            continue
        if trainable_scope == TRAINABLE_SCOPE_STD_DOWN_PROJ:
            eligible = role.role == "std_expert" and role.projection == "down_proj"
        else:
            eligible = role.role in {"std_expert", "shared_mlp"}
        if eligible:
            parameter.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise ValueError(
            f"RMU trainable selection found no parameters for scope "
            f"{trainable_scope!r} in layers {sorted(target_layers)}."
        )
    validate_rmu_trainable_parameters(
        model, layer_indices, trainable_scope, selected
    )
    return selected


def validate_rmu_trainable_parameters(
    model: torch.nn.Module,
    layer_indices: Sequence[int],
    trainable_scope: str,
    selected_names: Sequence[str] | None = None,
) -> None:
    if trainable_scope not in _TRAINABLE_SCOPES:
        raise ValueError(f"Unknown RMU trainable scope: {trainable_scope!r}.")
    target_layers = set(int(layer) for layer in layer_indices)
    actual = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if selected_names is not None and sorted(actual) != sorted(selected_names):
        raise RuntimeError("RMU trainable parameter set changed after selection.")
    if not actual:
        raise RuntimeError("RMU requires at least one trainable parameter.")
    for name in actual:
        role = parameter_role(model, name)
        valid = role.layer in target_layers
        if trainable_scope == TRAINABLE_SCOPE_STD_DOWN_PROJ:
            valid = (
                valid and role.role == "std_expert" and role.projection == "down_proj"
            )
        else:
            valid = valid and role.role in {"std_expert", "shared_mlp"}
        if not valid:
            raise RuntimeError(
                f"RMU scope {trainable_scope!r} rejected trainable parameter "
                f"{name!r}."
            )


def _budget_limit(setup: RmuSetup) -> int:
    if setup.budget_unit == BUDGET_FORGET_TOKENS:
        if setup.max_forget_tokens is None:
            raise RuntimeError("Forget-token budget is missing max_forget_tokens.")
        return setup.max_forget_tokens
    if setup.max_forget_epochs is None:
        raise RuntimeError("Forget-epoch budget is missing max_forget_epochs.")
    return setup.max_forget_epochs


def forget_epoch_optimizer_updates(
    setup: RmuSetup,
    forget_loader: Any,
) -> int | None:
    if setup.budget_unit != BUDGET_FORGET_EPOCHS:
        return None
    micro_batches = len(forget_loader)
    if micro_batches <= 0:
        raise ValueError("Forget dataloader must contain at least one batch.")
    if micro_batches % setup.gradient_accumulation_steps != 0:
        raise ValueError(
            "One forget epoch must end on an optimizer boundary: "
            f"micro_batches={micro_batches} "
            f"gradient_accumulation_steps={setup.gradient_accumulation_steps}."
        )
    return micro_batches // setup.gradient_accumulation_steps


def estimate_steps(
    setup: RmuSetup,
    config: Mapping[str, Any],
    *,
    epoch_optimizer_updates: int | None = None,
) -> int:
    configured = config.get("scheduler_total_steps")
    if setup.budget_unit == BUDGET_FORGET_EPOCHS:
        if epoch_optimizer_updates is None:
            raise ValueError(
                "Forget-epoch scheduling requires the dataloader-derived update count."
            )
        derived = epoch_optimizer_updates
    else:
        forget_tokens_per_step = setup.effective_global_batch_size * int(
            config["max_seq_length"]
        )
        derived = max(
            1,
            math.ceil(_budget_limit(setup) / forget_tokens_per_step),
        )
    if configured is None:
        return derived
    configured_steps = _positive_int(configured, name="scheduler_total_steps")
    if configured_steps < derived:
        raise ValueError(
            f"scheduler_total_steps={configured_steps} is below derived steps {derived}."
        )
    return configured_steps


def _train_rmu(ctx) -> None:
    config, accelerator, updated_model = ctx.config, ctx.accelerator, ctx.model
    setup, settings, schedule = ctx.prepared
    state = ctx.method_state
    reference_model, scheduler = ctx.reference_model, ctx.scheduler
    updated_recorder, reference_recorder = state.updated_recorder, state.reference_recorder
    epoch_optimizer_updates = state.epoch_optimizer_updates
    control_cache = RmuControlVectorCache(
        seed=settings.target_seed, steering_coeff=settings.steering_coeff
    )
    stream = PairedBatchStream(
        retain_loader=ctx.loaders[0],
        forget_loader=ctx.loaders[1],
        device=accelerator.device,
        non_blocking=bool(config.get("pin_memory", True)),
    )
    tokens_seen = 0
    forget_tokens_seen = 0
    retain_tokens_seen = 0
    optimizer_updates = 0
    updated_forwards = 0
    reference_forwards = 0
    backward_calls = 0
    saved_milestones: set[int] = set()
    started_at = time.perf_counter()
    checkpoint_seconds = 0.0
    log_interval = int(config.get("log_interval", 1))
    max_norm = float(config.get("max_norm", 0.0))

    def save(output_dir: Path, forget_epochs_completed: int) -> None:
        nonlocal checkpoint_seconds
        train_elapsed = max(0.0, time.perf_counter() - started_at - checkpoint_seconds)
        results = rmu_run_results(
            ctx,
            tokens_seen=tokens_seen,
            forget_tokens_seen=forget_tokens_seen,
            retain_tokens_seen=retain_tokens_seen,
            optimizer_updates=optimizer_updates,
            updated_forwards=updated_forwards,
            reference_forwards=reference_forwards,
            backward_calls=backward_calls,
            forget_epochs_completed=forget_epochs_completed,
            train_elapsed_seconds=train_elapsed,
            prior_checkpoint_elapsed_seconds=checkpoint_seconds,
        )
        save_started = time.perf_counter()
        _save_rmu_checkpoint(ctx, output_dir=output_dir, results=results)
        checkpoint_seconds += time.perf_counter() - save_started

    updated_model.train()
    reference_model.eval()

    while (
        optimizer_updates < epoch_optimizer_updates
        if epoch_optimizer_updates is not None
        else forget_tokens_seen < _budget_limit(setup)
    ):
        optimizer_updates += 1
        metric_values: dict[str, list[torch.Tensor]] = {}
        local_forget_tokens = 0
        local_retain_tokens = 0
        for _ in range(setup.gradient_accumulation_steps):
            retain_batch, forget_batch, retain_tokens, forget_tokens = stream.next()
            local_retain_tokens += retain_tokens
            local_forget_tokens += forget_tokens
            with accelerator.accumulate(updated_model):
                result = compute_rmu_step_loss(
                    updated_model=updated_model,
                    reference_model=reference_model,
                    retain_batch=retain_batch,
                    forget_batch=forget_batch,
                    updated_recorder=updated_recorder,
                    reference_recorder=reference_recorder,
                    control_cache=control_cache,
                    settings=settings,
                    layer_indices=setup.rmu_monitor_layer_indices,
                )
                updated_forwards += 1
                reference_forwards += 1
                accelerator.backward(result.loss)
                if config.get("verify_finite_training_step", False):
                    verify_finite_training_step(updated_model, result.loss, accelerator)
                    accelerator.print("[acceptance] RMU finite_loss_and_gradients=True")
                backward_calls += 1
                optimizer_step(ctx, clip=max_norm)
                for key, value in _loss_metrics(result).items():
                    metric_values.setdefault(key, []).append(value.detach())

        step_forget_tokens = _reduce_tokens(accelerator, local_forget_tokens)
        step_retain_tokens = _reduce_tokens(accelerator, local_retain_tokens)
        forget_tokens_seen += step_forget_tokens
        retain_tokens_seen += step_retain_tokens
        tokens_seen += step_forget_tokens + step_retain_tokens
        if log_interval > 0 and optimizer_updates % log_interval == 0:
            metrics = {
                "rmu/tokens_seen": float(tokens_seen),
                "rmu/forget_tokens_seen": float(forget_tokens_seen),
                "rmu/retain_tokens_seen": float(retain_tokens_seen),
                "rmu/lr": float(scheduler.get_last_lr()[0]),
            }
            for key, values in metric_values.items():
                metrics[f"rmu/{key}"] = float(
                    accelerator.gather(torch.stack(values).mean().reshape(1).float())
                    .mean()
                    .item()
                )
            if config.get("rmu_log_loss_norm_ratio", False):
                # A loss-derived activation-residual proxy, not a parameter gradient ratio.
                forget = metrics["rmu/forget_loss"]
                retain = metrics["rmu/retain_loss"]
                ratio = (
                    settings.retain_loss_weight
                    / settings.forget_loss_weight
                    * math.sqrt(retain / forget)
                    if forget > 0 and retain >= 0
                    else None
                )
                if ratio is not None:
                    metrics["rmu/retain_forget_loss_norm_ratio"] = ratio
                if accelerator.is_main_process:
                    setup.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    with (setup.checkpoint_dir / "rmu_metrics.jsonl").open(
                        "a"
                    ) as handle:
                        handle.write(
                            json.dumps(
                                {"step": optimizer_updates, **metrics}, allow_nan=False
                            )
                            + "\n"
                        )
            accelerator.log(metrics, step=optimizer_updates)
            accelerator.print(
                "[rmu] "
                f"step={optimizer_updates} tokens={tokens_seen} "
                f"loss={metrics.get('rmu/loss', 0.0):.4f} "
                f"forget={metrics.get('rmu/forget_loss', 0.0):.4f} "
                f"retain={metrics.get('rmu/retain_loss', 0.0):.4f}"
            )

        completed_epochs = (
            (setup.max_forget_epochs or 0)
            if epoch_optimizer_updates is not None
            and optimizer_updates >= epoch_optimizer_updates
            else 0
        )
        for milestone in schedule.due(forget_tokens_seen, saved_milestones):
            save(
                setup.checkpoint_dir / MILESTONE_SUBDIR / f"forget-tokens-{milestone:012d}",
                completed_epochs,
            )
            saved_milestones.add(milestone)

    # The final checkpoint is the run output itself.
    save(
        setup.checkpoint_dir,
        setup.max_forget_epochs if setup.budget_unit == BUDGET_FORGET_EPOCHS else 0,
    )


def rmu_run_results(
    ctx,
    *,
    tokens_seen: int,
    forget_tokens_seen: int,
    retain_tokens_seen: int,
    optimizer_updates: int,
    updated_forwards: int,
    reference_forwards: int,
    backward_calls: int,
    forget_epochs_completed: int,
    train_elapsed_seconds: float,
    prior_checkpoint_elapsed_seconds: float,
) -> dict[str, Any]:
    """Actual counts at a save; forget and retain tokens are kept separate."""
    setup, state = ctx.setup, ctx.method_state
    budget: dict[str, Any] = {"budget_unit": setup.budget_unit}
    if setup.budget_unit == BUDGET_FORGET_TOKENS:
        budget["max_forget_tokens"] = setup.max_forget_tokens
    else:
        budget["max_forget_epochs"] = setup.max_forget_epochs
        budget["forget_epochs_completed"] = int(forget_epochs_completed)
    world_size = int(ctx.accelerator.num_processes)
    return {
        **budget,
        "total_tokens_seen": int(tokens_seen),
        "forget_tokens_seen": int(forget_tokens_seen),
        "retain_tokens_seen": int(retain_tokens_seen),
        "optimizer_updates": int(optimizer_updates),
        "updated_model_forwards": int(updated_forwards),
        "reference_model_forwards": int(reference_forwards),
        "backward_calls": int(backward_calls),
        "world_size": world_size,
        "effective_global_batch_size": setup.effective_global_batch_size,
        "trainable_scope": setup.rmu_trainable_scope,
        "trainable_parameter_count": int(state.trainable_parameter_count),
        "trainable_parameter_names": sorted(state.trainable_names),
        "train_elapsed_seconds": train_elapsed_seconds,
        "prior_checkpoint_elapsed_seconds": prior_checkpoint_elapsed_seconds,
        "gpu_hours": train_elapsed_seconds * world_size / 3600.0,
        "tokens_per_second": (
            tokens_seen / train_elapsed_seconds if train_elapsed_seconds > 0 else 0.0
        ),
        "forget_tokens_per_second": (
            forget_tokens_seen / train_elapsed_seconds if train_elapsed_seconds > 0 else 0.0
        ),
    }


def compute_rmu_step_loss(
    *,
    updated_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    retain_batch: dict[str, torch.Tensor],
    forget_batch: dict[str, torch.Tensor],
    updated_recorder: RmuActivationRecorder,
    reference_recorder: RmuActivationRecorder,
    control_cache: RmuControlVectorCache,
    settings: RmuSettings,
    layer_indices: tuple[int, ...],
) -> RmuStepLoss:
    updated_batch, forget_batch_size = _concat_batches(forget_batch, retain_batch)
    updated_outputs, updated_activations = _forward_with_recorder(
        updated_model, updated_batch, updated_recorder, no_grad=False
    )
    forget_activations, retain_activations = _split_activations(
        updated_activations, forget_batch_size
    )
    _, reference_activations = _forward_with_recorder(
        reference_model, retain_batch, reference_recorder, no_grad=True
    )
    forget_result = compute_rmu_loss(
        forget_activations,
        layer_indices,
        rmu_token_mask_from_batch(forget_batch, settings.token_mask_mode),
        control_cache,
        max_tokens_per_batch=settings.max_tokens_per_batch,
        loss_reduction=settings.loss_reduction,
        loss_compute_dtype=settings.loss_compute_dtype,
    )
    retain_result = compute_rmu_retain_loss(
        retain_activations,
        reference_activations,
        layer_indices,
        rmu_token_mask_from_batch(retain_batch, settings.token_mask_mode),
        max_tokens_per_batch=settings.max_tokens_per_batch,
        loss_reduction=settings.loss_reduction,
        loss_compute_dtype=settings.loss_compute_dtype,
    )
    weighted_forget = forget_result.forget_loss * settings.forget_loss_weight
    weighted_retain = retain_result.retain_loss * settings.retain_loss_weight
    loss = weighted_forget + weighted_retain + _zero_output_anchor(updated_outputs)
    return RmuStepLoss(
        loss=loss,
        forget_loss=forget_result.forget_loss,
        retain_loss=retain_result.retain_loss,
        weighted_forget_loss=weighted_forget,
        weighted_retain_loss=weighted_retain,
        forget_activation_norm=forget_result.forget_activation_norm,
        forget_target_norm=forget_result.target_norm,
        forget_cosine=forget_result.forget_cosine,
        forget_tokens=forget_result.forget_tokens,
        retain_activation_norm=retain_result.retain_activation_norm,
        retain_frozen_activation_norm=retain_result.frozen_activation_norm,
        retain_cosine=retain_result.retain_cosine,
        retain_tokens=retain_result.retain_tokens,
    )


def _concat_batches(
    forget: dict[str, torch.Tensor], retain: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int]:
    if set(forget) != set(retain):
        raise ValueError("RMU forget and retain batches must have identical keys.")
    forget_size = _batch_size(forget)
    combined: dict[str, torch.Tensor] = {}
    for key in forget:
        forget_value = forget[key]
        retain_value = retain[key]
        if not torch.is_tensor(forget_value) or not torch.is_tensor(retain_value):
            raise ValueError(f"RMU batch key {key!r} must be tensor-valued.")
        if forget_value.ndim == 0 or forget_value.shape[1:] != retain_value.shape[1:]:
            raise ValueError(
                f"RMU batch shapes must align for {key}: "
                f"forget={tuple(forget_value.shape)} retain={tuple(retain_value.shape)}"
            )
        combined[key] = torch.cat((forget_value, retain_value), dim=0)
    return combined, forget_size


def _batch_size(batch: Mapping[str, torch.Tensor]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("RMU batch has no tensor with a batch dimension.")


def _split_activations(
    activations: Mapping[int, torch.Tensor], forget_size: int
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    forget: dict[int, torch.Tensor] = {}
    retain: dict[int, torch.Tensor] = {}
    for layer, activation in activations.items():
        if activation.ndim == 0 or activation.shape[0] <= forget_size:
            raise ValueError(
                f"RMU activation cannot be split at {forget_size}: "
                f"layer={layer} shape={tuple(activation.shape)}"
            )
        forget[layer] = activation[:forget_size]
        retain[layer] = activation[forget_size:]
    return forget, retain


def _forward_with_recorder(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    recorder: RmuActivationRecorder,
    *,
    no_grad: bool,
) -> tuple[Any, dict[int, torch.Tensor]]:
    model_batch = {key: value for key, value in batch.items() if key != "labels"}
    context = torch.no_grad() if no_grad else nullcontext()
    with context:
        with recorder.recording():
            outputs = model(
                **model_batch,
                use_cache=False,
                output_hidden_states=False,
                logits_to_keep=1,
            )
    return outputs, recorder.activations()


def _zero_output_anchor(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if torch.is_tensor(logits):
        return logits.sum() * 0.0
    if torch.is_tensor(outputs):
        return outputs.sum() * 0.0
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item):
                return item.sum() * 0.0
    raise RuntimeError("RMU updated forward returned no tensor for the FSDP anchor.")


def _loss_metrics(result: RmuStepLoss) -> dict[str, torch.Tensor]:
    return {
        "loss": result.loss,
        "forget_loss": result.forget_loss,
        "retain_loss": result.retain_loss,
        "weighted_forget_loss": result.weighted_forget_loss,
        "weighted_retain_loss": result.weighted_retain_loss,
        "forget_activation_norm": result.forget_activation_norm,
        "forget_target_norm": result.forget_target_norm,
        "forget_cosine": result.forget_cosine,
        "forget_tokens": result.forget_tokens,
        "retain_activation_norm": result.retain_activation_norm,
        "retain_frozen_activation_norm": result.retain_frozen_activation_norm,
        "retain_cosine": result.retain_cosine,
        "retain_tokens": result.retain_tokens,
    }


def _save_rmu_checkpoint(ctx, *, output_dir: Path, results: dict[str, Any]) -> None:
    ctx.accelerator.print(f"[rmu][checkpoint] saving {output_dir}")
    _save_shared_checkpoint(
        accelerator=ctx.accelerator,
        tokenizer=ctx.tokenizer,
        model=ctx.model,
        output_dir=output_dir,
        source_dirs=_runtime_source_dirs(ctx.config),
        # Parents are standard checkpoints; there is no harmful bank to fold back.
        merge_expanded_moe=False,
        write_record=lambda directory: write_run_record(
            directory,
            step="train_rmu",
            inputs={"parent": ctx.config["initial_checkpoint_dir"]},
            config=ctx.method_state.recipe,
            args=ctx.method_state.args,
            results=results,
        ),
    )


@dataclass
class RmuRuntime:
    trainable_names: list[str]
    trainable_parameter_count: int
    recipe: dict[str, Any]
    args: dict[str, Any]
    epoch_optimizer_updates: int | None = None
    updated_recorder: Any = None
    reference_recorder: Any = None
    finalizer: Any = None


def configure_rmu_model(ctx, *, recipe, args):
    setup, model = ctx.setup, ctx.model
    names = select_rmu_trainable_parameters(
        model, setup.rmu_trainable_layer_indices, setup.rmu_trainable_scope
    )
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ctx.method_state = RmuRuntime(names, count, recipe, args)
    mark_rmu_layers_to_skip_activation_checkpointing(
        model,
        list(setup.rmu_monitor_layer_indices),
        module_template=setup.rmu_module_template,
    )


def prepare_rmu_optimizer(ctx):
    optimizer = build_adamw(
        default_parameter_groups(ctx.model, float(ctx.config.get("weight_decay", 0.0))), ctx.config
    )
    state = ctx.method_state
    state.epoch_optimizer_updates = forget_epoch_optimizer_updates(
        ctx.setup, ctx.loaders[1]
    )
    steps = estimate_steps(
        ctx.setup, ctx.config, epoch_optimizer_updates=state.epoch_optimizer_updates
    )
    scheduler = build_scheduler(optimizer, ctx.config, steps)
    return ctx.accelerator.prepare(ctx.model, optimizer, scheduler)


def finish_rmu_setup(ctx):
    setup, accelerator, state = ctx.setup, ctx.accelerator, ctx.method_state
    if setup.selective_activation_checkpointing:
        checkpointed, skipped = apply_selective_activation_checkpointing(
            ctx.model,
            transformer_cls_names=model_decoder_classes(ctx.model),
            checkpoint_stride=1,
        )
    else:
        checkpointed, skipped = 0, 0
    accelerator.print(
        f"[rmu][fsdp] checkpointed_layers={checkpointed} skipped_layers={skipped}"
    )
    validate_rmu_trainable_parameters(
        accelerator.unwrap_model(ctx.model),
        setup.rmu_trainable_layer_indices,
        setup.rmu_trainable_scope,
        None,
    )
    state.updated_recorder = RmuActivationRecorder(
        accelerator.unwrap_model(ctx.model),
        list(setup.rmu_monitor_layer_indices),
        module_template=setup.rmu_module_template,
    )
    state.reference_recorder = RmuActivationRecorder(
        accelerator.unwrap_model(ctx.reference_model),
        list(setup.rmu_monitor_layer_indices),
        module_template=setup.rmu_module_template,
    )
    state.finalizer = TrainingFinalizer(
        accelerator, (state.updated_recorder, state.reference_recorder)
    )
    ctx.init_trackers(
        {
            "method": "rmu",
            "effective_global_batch_size": setup.effective_global_batch_size,
            "trainable_parameter_names": state.trainable_names,
            "trainable_parameter_count": state.trainable_parameter_count,
        }
    )


def run_rmu(
    config_path: Path,
    parent: Path,
    output: Path,
    *,
    milestones: Sequence[int] = (),
) -> None:
    config_path = config_path.resolve()
    recipe = load_config(config_path)
    args = {
        "config": str(config_path),
        "parent": str(parent.resolve()),
        "output": str(output.resolve()),
        MILESTONES_KEY: list(milestones),
    }

    def prepare(config, path, *, world_size):
        if MILESTONES_KEY in config:
            raise ValueError(
                f"{MILESTONES_KEY} is a command-line option, not a recipe key."
            )
        config[MILESTONES_KEY] = list(milestones)
        return prepare_rmu_config(config, path, world_size=world_size)

    ctx = build_training_context(
        config_path,
        # The retain reference is the pre-RMU parent itself.
        paths={
            "initial_checkpoint_dir": str(parent),
            "tokenizer_dir": str(parent),
            "rmu_retain_reference_checkpoint_dir": str(parent),
            "output_dir": str(output),
        },
        prepare_fn=prepare,
        build_dataloaders=build_xtuner_train_dataloaders,
        accelerator_factory=build_accelerator,
        model_loader=load_model_and_tokenizer,
        configure_model=lambda ctx: configure_rmu_model(ctx, recipe=recipe, args=args),
        prepare_model=prepare_rmu_optimizer,
        finish_setup=finish_rmu_setup,
        needs_reference=lambda ctx: ctx.config["rmu_retain_reference_checkpoint_dir"],
    )
    state = ctx.method_state
    try:
        _train_rmu(ctx)
    finally:
        state.finalizer.close()


def rmu_budget(config):
    has_forget_token_budget = "max_forget_tokens" in config
    has_forget_epoch_budget = "max_forget_epochs" in config
    if has_forget_token_budget == has_forget_epoch_budget:
        raise ValueError(
            "RMU config must set exactly one of max_forget_tokens or "
            "max_forget_epochs."
        )
    if has_forget_token_budget:
        budget_unit = BUDGET_FORGET_TOKENS
        max_forget_tokens = _positive_int(
            config["max_forget_tokens"], name="max_forget_tokens"
        )
        max_forget_epochs = None
    else:
        budget_unit = BUDGET_FORGET_EPOCHS
        max_forget_tokens = None
        max_forget_epochs = _positive_int(
            config["max_forget_epochs"], name="max_forget_epochs"
        )
        if max_forget_epochs != 1:
            raise ValueError(
                "RMU max_forget_epochs currently supports exactly one epoch."
            )
    return budget_unit, max_forget_tokens, max_forget_epochs


def rmu_scope(config):
    if bool(config.get("gradient_checkpointing", False)):
        raise ValueError(
            "RMU manages checkpointing with selective_activation_checkpointing; "
            "Transformers gradient_checkpointing must be false."
        )
    selective_activation_checkpointing = config["selective_activation_checkpointing"]
    if not isinstance(selective_activation_checkpointing, bool):
        raise ValueError("selective_activation_checkpointing must be a boolean.")
    monitor_layer_indices = _layer_indices(config, "rmu_monitor_layer_indices")
    trainable_layer_indices = _layer_indices(
        config, "rmu_trainable_layer_indices"
    )
    trainable_scope = str(config["rmu_trainable_scope"])
    if trainable_scope not in _TRAINABLE_SCOPES:
        raise ValueError(
            "rmu_trainable_scope must be one of "
            f"{sorted(_TRAINABLE_SCOPES)}, got {trainable_scope!r}."
        )
    module_template = config.get("rmu_module_template")
    if module_template is not None and not isinstance(module_template, str):
        raise ValueError("rmu_module_template must be a string when set.")

    return (
        selective_activation_checkpointing,
        monitor_layer_indices,
        trainable_layer_indices,
        trainable_scope,
        module_template,
    )


def rmu_settings(config):
    forget_weight = _finite_float(
        config.get("rmu_forget_loss_weight", 1.0), name="rmu_forget_loss_weight"
    )
    retain_weight = _finite_float(
        config.get("rmu_retain_loss_weight", 0.0), name="rmu_retain_loss_weight"
    )
    if forget_weight == 0.0:
        raise ValueError("rmu_forget_loss_weight must be non-zero.")
    if retain_weight < 0.0:
        raise ValueError("rmu_retain_loss_weight must be non-negative.")
    steering_coeff = _finite_float(
        config.get("rmu_steering_coeff", 1.0), name="rmu_steering_coeff"
    )
    if steering_coeff <= 0.0:
        raise ValueError("rmu_steering_coeff must be positive.")
    loss_reduction = normalize_rmu_loss_reduction(
        str(config.get("rmu_loss_reduction", "token_mean"))
    )
    loss_compute_dtype = normalize_rmu_loss_compute_dtype(
        str(config.get("rmu_loss_compute_dtype", "float32"))
    )
    max_tokens_per_batch_raw = config.get("rmu_max_tokens_per_batch", 0)
    max_tokens_per_batch = int(max_tokens_per_batch_raw)
    if max_tokens_per_batch < 0:
        raise ValueError("rmu_max_tokens_per_batch must be non-negative.")
    if loss_reduction == "sequence_mean" and max_tokens_per_batch > 0:
        raise ValueError(
            "rmu_loss_reduction=sequence_mean cannot use rmu_max_tokens_per_batch."
        )

    settings = RmuSettings(
        forget_loss_weight=forget_weight,
        retain_loss_weight=retain_weight,
        token_mask_mode=str(config.get("rmu_token_mask", "labels")),
        max_tokens_per_batch=max_tokens_per_batch or None,
        loss_reduction=loss_reduction,
        loss_compute_dtype=loss_compute_dtype,
        target_seed=int(config.get("rmu_target_seed", config.get("seed", 42))),
        steering_coeff=steering_coeff,
    )
    return settings


def rmu_schedule(config, max_forget_tokens):
    milestones_raw = config.get(MILESTONES_KEY, [])
    if not isinstance(milestones_raw, list):
        raise ValueError(
            "save_forget_token_milestones must be a list of positive integers."
        )
    milestones = tuple(
        sorted(
            {
                _positive_int(item, name="save_forget_token_milestones item")
                for item in milestones_raw
            }
        )
    )
    # Checkpoint triggers count observed forget tokens independently of the
    # stopping rule. In epoch mode, the loader boundary still ends training;
    # milestones beyond that boundary are simply not reached.
    if max_forget_tokens is not None and any(
        milestone > max_forget_tokens for milestone in milestones
    ):
        raise ValueError(
            "save_forget_token_milestones cannot exceed max_forget_tokens."
        )
    return RmuSchedule(save_forget_token_milestones=milestones)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RMU on a standard checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path, help="A1 for A2, B for R")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--save-forget-token-milestones", nargs="+", type=int, default=[], metavar="N",
        help="also save checkpoints once N forget tokens have been seen",
    )
    args = parser.parse_args()
    run_rmu(
        args.config, args.parent, args.output,
        milestones=args.save_forget_token_milestones,
    )


if __name__ == "__main__":
    main()
