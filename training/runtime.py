from __future__ import annotations

from common.tokenizers import ensure_pad_token

import functools
import math
import time
from datetime import timedelta
from typing import Any

import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, enable_fsdp_ram_efficient_loading
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from dataset.xtuner.loader import create_xtuner_dataloaders, warmup_xtuner_offset_cache
from moe.model_adapters import adapter_from_checkpoint, model_adapter

SKIP_ACTIVATION_CHECKPOINT_ATTR = "_rex_skip_activation_checkpointing"


def build_fsdp_plugin(adapter) -> FullyShardedDataParallelPlugin:
    enable_fsdp_ram_efficient_loading()
    return FullyShardedDataParallelPlugin(
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=[adapter.decoder_class],
        activation_checkpointing=False,
        backward_prefetch="BACKWARD_POST",
        cpu_ram_efficient_loading=True,
        use_orig_params=True,
        sync_module_states=True,
        mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            # Preserve native buffer precision (e.g. RoPE frequencies and Gemma
            # embedding scale). Nonpersistent buffers are rebuilt by HF on
            # reload, so rounding them during training changes model outputs.
            buffer_dtype=None,
        ),
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def build_accelerator(config: dict[str, Any], *, gradient_accumulation_steps: int) -> Accelerator:
    fp32_master_weights = config.get("fsdp_fp32_master_weights", False)
    if not isinstance(fp32_master_weights, bool):
        raise ValueError("fsdp_fp32_master_weights must be a boolean.")
    distributed_timeout = timedelta(seconds=int(config.get("distributed_timeout_seconds", 21_600)))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # The FSDP policy alone controls forward/backward precision; with
        # Accelerator mixed_precision="no", BF16-loaded parameters also remain
        # BF16 in AdamW. Enable Accelerate's FSDP FP32 master-parameter upcast so
        # small updates can accumulate (notably Gemma's near-unit router scales).
        # Keep this opt-in: existing recipes retain their numerical behavior.
        mixed_precision="bf16" if fp32_master_weights else "no",
        fsdp_plugin=build_fsdp_plugin(
            adapter_from_checkpoint(config["initial_checkpoint_dir"])
        ),
        kwargs_handlers=[InitProcessGroupKwargs(timeout=distributed_timeout)],
        log_with="wandb",
        step_scheduler_with_optimizer=False,
    )
    if fp32_master_weights:
        from accelerate.utils import DistributedType

        if accelerator.distributed_type != DistributedType.FSDP:
            raise ValueError("fsdp_fp32_master_weights requires FSDP execution.")
        accelerator.print(
            "[precision] FSDP FP32 master weights enabled; forward/backward remain BF16."
        )
    return accelerator






def _transformer_checkpoint_candidates(
    model: torch.nn.Module,
    transformer_cls_names: list[str],
) -> list[torch.nn.Module]:
    transformer_cls_name_set = set(transformer_cls_names)
    return [
        module
        for module in model.modules()
        if module.__class__.__name__ in transformer_cls_name_set
    ]


def _selected_checkpoint_module_ids(
    candidates: list[torch.nn.Module],
    *,
    checkpoint_stride: int,
) -> set[int]:
    return {
        id(module)
        for idx, module in enumerate(candidates)
        if idx % checkpoint_stride == 0
        and not getattr(module, SKIP_ACTIVATION_CHECKPOINT_ATTR, False)
    }


def apply_selective_activation_checkpointing(
    model: torch.nn.Module,
    *,
    transformer_cls_names: list[str],
    checkpoint_stride: int = 1,
) -> tuple[int, int]:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    candidates = _transformer_checkpoint_candidates(model, transformer_cls_names)
    if checkpoint_stride <= 0:
        return 0, len(candidates)
    checkpoint_ids = _selected_checkpoint_module_ids(
        candidates,
        checkpoint_stride=checkpoint_stride,
    )
    transformer_cls_name_set = set(transformer_cls_names)

    def check_fn(module: torch.nn.Module) -> bool:
        if module.__class__.__name__ not in transformer_cls_name_set:
            return False
        return id(module) in checkpoint_ids

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )
    return len(checkpoint_ids), len(candidates) - len(checkpoint_ids)


def uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    if parameter.ndim < 2:
        return False
    if lowered.endswith(".bias") or "norm" in lowered or "ln_" in lowered:
        return False
    return True


def default_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = decay_params if uses_weight_decay(name, parameter) else no_decay_params
        target.append(parameter)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_adamw(parameter_groups: list[dict[str, Any]], config: dict[str, Any]) -> torch.optim.Optimizer:
    return AdamW(
        parameter_groups,
        lr=float(config["lr"]),
        betas=tuple(float(v) for v in config.get("betas", (0.9, 0.95))),
    )






def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any], estimated_steps: int) -> LambdaLR:
    warmup_steps = int(config.get("lr_warmup_steps", 0))
    base_lr = float(config["lr"])
    min_lr = float(config.get("min_lr", 0.0))
    min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        if estimated_steps <= warmup_steps:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / (estimated_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def disable_model_cache(model: torch.nn.Module) -> None:
    configs = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
        getattr(model, "generation_config", None),
    ]
    for model_config in configs:
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = False


def _xtuner_config(config: dict[str, Any]) -> dict[str, Any]:
    xtuner_cfg = config.get("xtuner") or {}
    if not isinstance(xtuner_cfg, dict):
        raise ValueError("xtuner config must be a mapping.")
    return xtuner_cfg


def _prewarm_xtuner_metadata_if_requested(
    config: dict[str, Any],
    accelerator: Accelerator,
    xtuner_cfg: dict[str, Any],
) -> None:
    if not bool(xtuner_cfg.get("prewarm_metadata_cache", False)):
        return
    warmup_workers = int(xtuner_cfg.get("prewarm_workers", 16))
    warmup_log_interval = int(xtuner_cfg.get("prewarm_log_interval", 100))
    if accelerator.is_main_process:
        warmed_sources, elapsed = warmup_xtuner_offset_cache(
            config,
            max_workers=warmup_workers,
            log_interval=warmup_log_interval,
        )
        accelerator.print(f"[xtuner-data] prewarm_metadata_cache sources={warmed_sources} elapsed={elapsed:.1f}s")
    accelerator.wait_for_everyone()


def _xtuner_dataloader_kwargs(
    config: dict[str, Any],
    *,
    micro_batch_size: int,
    global_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "train_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "seed": seed,
        "num_workers": int(config.get("num_workers", 0)),
        "pin_memory": bool(config.get("pin_memory", True)),
        "prefetch_factor": config.get("prefetch_factor"),
    }


def _create_xtuner_dataloaders(
    config: dict[str, Any],
    *,
    accelerator: Accelerator,
    pack_level: str,
    dataloader_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    if pack_level == "hard":
        return create_xtuner_dataloaders(
            config,
            **dataloader_kwargs,
            source_build_context=accelerator.main_process_first,
        )
    with accelerator.main_process_first():
        return create_xtuner_dataloaders(
            config,
            **dataloader_kwargs,
        )


def build_xtuner_train_dataloaders(
    config: dict[str, Any],
    *,
    accelerator: Accelerator,
    micro_batch_size: int,
    global_batch_size: int,
    seed: int,
) -> tuple[Any, Any]:
    xtuner_cfg = _xtuner_config(config)
    _prewarm_xtuner_metadata_if_requested(config, accelerator, xtuner_cfg)

    data_start = time.perf_counter()
    accelerator.print("[xtuner-data] building dataloaders")
    pack_level = str(xtuner_cfg.get("pack_level", "soft")).lower()
    loaders = _create_xtuner_dataloaders(
        config,
        accelerator=accelerator,
        pack_level=pack_level,
        dataloader_kwargs=_xtuner_dataloader_kwargs(
            config,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            seed=seed,
        ),
    )
    accelerator.print(f"[xtuner-data] dataloaders ready elapsed={time.perf_counter() - data_start:.1f}s")
    return loaders


def canonicalize_nonpersistent_buffer_order(model: torch.nn.Module) -> None:
    """Make positional FSDP broadcasts agree across ranks without changing state dicts.

    Some HF modules register nonpersistent buffers by iterating a set (Gemma's
    full/sliding-attention RoPE buffers). Python hash randomization can change
    their order between workers; FSDP synchronizes them by position, not name.
    Keep persistent-buffer slots and tensor objects unchanged.
    """
    for module in model.modules():
        transient = module._non_persistent_buffers_set
        ordered = iter(sorted(name for name in module._buffers if name in transient))
        names = [next(ordered) if name in transient else name for name in module._buffers]
        module._buffers = {name: module._buffers[name] for name in names}


def load_model_and_tokenizer(
    config: dict[str, Any],
    accelerator: Accelerator,
) -> tuple[Any, torch.nn.Module]:
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            str(config["tokenizer_dir"]),
            use_fast=True,
            trust_remote_code=True,
        )
    ensure_pad_token(tokenizer)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if config.get("attn_implementation"):
        attn_implementation = config["attn_implementation"]
        model_kwargs["attn_implementation"] = (
            attn_implementation if isinstance(attn_implementation, dict) else str(attn_implementation)
        )

    from transformers import AutoConfig
    checkpoint_config = AutoConfig.from_pretrained(str(config["initial_checkpoint_dir"]), trust_remote_code=True)
    adapter = adapter_from_checkpoint(config["initial_checkpoint_dir"])
    model, loading_info = adapter.auto_model_class(checkpoint_config).from_pretrained(
        str(config["initial_checkpoint_dir"]),
        output_loading_info=True,
        **model_kwargs,
    )
    unexpected = [k for k in loading_info.get("unexpected_keys", []) if not k.startswith("mtp.")]
    if unexpected or loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise ValueError(f"Checkpoint does not satisfy {adapter.name} loading contract: {loading_info}")
    canonicalize_nonpersistent_buffer_order(model)
    # FP32 FSDP masters do not recover precision lost by the BF16 loader.
    # Keep that conversion explicit for frozen-parent preservation on save.
    if config.get("fsdp_fp32_master_weights", False):
        model._rex_parameter_load_dtype = torch.bfloat16
    disable_model_cache(model)
    return tokenizer, model


def model_decoder_classes(model):
    return [model_adapter(model).decoder_class]


def verify_finite_training_step(model, loss, accelerator):
    """Opt-in acceptance check, reduced before raising on every rank."""
    checks = [torch.isfinite(loss.detach()).all()]
    checks.extend(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    failed = (~torch.stack(checks).all()).to(dtype=torch.int32)
    if int(accelerator.reduce(failed, reduction="sum").item()):
        raise RuntimeError("Acceptance check found non-finite loss or gradients.")
