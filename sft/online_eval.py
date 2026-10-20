"""Resident-weight milestone evaluation for Swift SFT; never exports a model.

The repository's MCoreEvalLM supplies EP-synchronous forward/scoring. This
module owns scheduling, full/masked phases, reproducibility and evidence IO.
Imports of Swift/Megatron/lm-eval are deferred so the contracts are CPU-testable.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Self

import torch
import torch.distributed as dist

from moe.runtime import MegatronExpertMasker
from common.io_utils import read_json, write_json_atomic
from common.manifest_names import EXPANSION_MANIFEST
from moe.model_adapters import adapter_from_checkpoint, megatron_moe_layers
from eval.lm_eval_io import numeric_lm_eval_metrics


def eval_milestone_steps(
    milestones: tuple[int, ...], tokens_per_step: int, max_tokens: int
) -> dict[int, int]:
    if not milestones or tuple(sorted(set(milestones))) != milestones:
        raise ValueError(
            "Evaluation milestones must be nonempty and strictly increasing."
        )
    if tokens_per_step <= 0 or max_tokens <= 0:
        raise ValueError("Token budget and tokens per step must be positive.")
    if any(
        value < 0 or value > max_tokens or value % tokens_per_step
        for value in milestones
    ):
        raise ValueError(
            "Evaluation milestones must be in [0, max_tokens] and divisible by tokens per step."
        )
    # Unlike saving, evaluation need not include the endpoint or any save step.
    return {value // tokens_per_step: value for value in milestones}


@dataclass(frozen=True)
class OnlineEvalConfig:
    milestones: tuple[int, ...]
    tasks: tuple[str, ...]
    output_root: Path
    batch_size: int = 1  # questions per forward, independent of training batch
    max_tokens_per_forward: int = 4096
    limit: int | None = None
    seed: int = 42
    num_fewshot: int | None = None  # preserve the task's native default

    @classmethod
    def from_recipe(
        cls, recipe: Mapping[str, Any] | None, *, output_root: Path
    ) -> Self | None:
        """``recipe`` is the ``eval`` section of the SFT recipe; absent or
        without milestones means no resident evaluation."""
        if not recipe or not recipe.get("token_milestones"):
            return None
        tasks = tuple(recipe["tasks"])
        if not all(tasks) or len(set(tasks)) != len(tasks):
            raise ValueError("Evaluation tasks must be nonempty and unique.")
        config = cls(
            milestones=tuple(int(value) for value in recipe["token_milestones"]),
            tasks=tasks,
            output_root=Path(output_root).resolve(),
            batch_size=int(recipe.get("batch_size", 1)),
            max_tokens_per_forward=int(recipe.get("max_tokens_per_forward", 4096)),
            limit=recipe.get("limit"),
            seed=int(recipe["seed"]),
            num_fewshot=recipe.get("num_fewshot"),
        )
        if (
            config.batch_size <= 0
            or config.max_tokens_per_forward <= 0
            or (config.limit is not None and config.limit <= 0)
        ):
            raise ValueError(
                "Evaluation batch size, forward token cap and optional limit must be positive."
            )
        if config.seed < 0 or (
            config.num_fewshot is not None and config.num_fewshot < 0
        ):
            raise ValueError(
                "Evaluation seed and optional fewshot count must be nonnegative."
            )
        return config


@contextlib.contextmanager
def preserve_eval_state(model, routers, rng_tracker):
    """Restore mixed train/eval flags, router statistics and all training RNGs."""
    from training.lm_eval_runner import preserve_rng_state

    flags = [(module, module.training) for module in model.modules()]
    buffers = [
        (buffer, buffer.clone()) for router in routers for buffer in router.buffers()
    ]
    tracker_state = {
        name: value.clone() for name, value in rng_tracker.get_states().items()
    }
    with preserve_rng_state():
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            for module, training in flags:
                module.training = training
            rng_tracker.set_states(tracker_state)


@contextlib.contextmanager
def suspend_training_pre_hooks(trainer):
    from swift.megatron.trainers.base import (
        disable_forward_pre_hook,
        enable_forward_pre_hook,
    )

    had_pre_hook = getattr(trainer, "_pre_hook_enabled", False)
    try:
        if had_pre_hook:
            disable_forward_pre_hook(trainer.wrapped_models)
            trainer._pre_hook_enabled = False
        yield
    finally:
        if had_pre_hook:
            enable_forward_pre_hook(trainer.wrapped_models)
            trainer._pre_hook_enabled = True


def rank0_call(action: Callable[[], Any]) -> Any:
    """Propagate writer failures before any rank enters the next EP forward."""
    distributed = dist.is_initialized()
    payload = [None, None]
    if not distributed or dist.get_rank() == 0:
        try:
            payload[0] = action()
        except Exception as error:
            payload[1] = f"{type(error).__name__}: {error}"
    if distributed:
        dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError(f"SFT online evaluation failed: {payload[1]}")
    return payload[0]


class SftOnlineEvaluator:
    def __init__(self, trainer, config: OnlineEvalConfig, initial_model: Path):
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import handle_non_serializable

        from sft.megatron_eval import MCoreEvalLM, SUPPORTED_ONLINE_EVAL_TASKS

        for field in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ):
            if getattr(trainer.args, field, 1) != 1:
                raise ValueError(
                    "SFT online evaluation requires TP=PP=CP=1 (EP is supported)."
                )
        if len(trainer.unwrapped_models) != 1 or getattr(
            trainer.args, "sequence_parallel", False
        ):
            raise ValueError(
                "SFT online evaluation requires one model chunk and no sequence parallelism."
            )
        if getattr(trainer.args, "online_eval_steps", None) or getattr(
            trainer.args, "online_eval_interval", None
        ):
            raise ValueError(
                "Use only the REX milestone evaluator; disable Swift's separate online-eval schedule."
            )
        if set(config.tasks) - set(SUPPORTED_ONLINE_EVAL_TASKS):
            raise ValueError(
                f"Supported SFT online-eval tasks: {sorted(SUPPORTED_ONLINE_EVAL_TASKS)}"
            )
        self.trainer, self.config = trainer, config
        self.model = trainer.unwrapped_models[0]
        self.rng_tracker = get_cuda_rng_tracker()
        adapter = adapter_from_checkpoint(initial_model)
        layers = megatron_moe_layers(self.model, adapter)
        self.routers = [layer.router for layer in layers]
        input_config = read_json(initial_model / "config.json")
        count = adapter.config_num_experts(input_config)
        if any(int(router.config.num_moe_experts) != count for router in self.routers):
            raise ValueError(
                "Resident Megatron expert count differs from the input config."
            )
        expansion = initial_model / EXPANSION_MANIFEST
        self.masker = (
            MegatronExpertMasker(layers, initial_model, adapter_name=adapter.name)
            if expansion.is_file()
            else None
        )
        self.initial_model = initial_model
        self.simple_evaluate, self.task_manager = simple_evaluate, TaskManager
        self.json_default = handle_non_serializable
        self.lm_class = MCoreEvalLM
        self.dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        self.dp_rank = parallel_state.get_data_parallel_rank(with_context_parallel=True)
        self.dp_size = parallel_state.get_data_parallel_world_size(
            with_context_parallel=True
        )
        backend_path = Path(inspect.getfile(MCoreEvalLM))
        self.backend = {
            "path": str(backend_path),
            "sha256": hashlib.sha256(backend_path.read_bytes()).hexdigest(),
        }

    def _write_task_result(
        self, result: dict | None, *, phase: str, task: str, directory: Path
    ) -> dict:
        if result is None:
            raise RuntimeError("lm-eval returned no result on global rank 0.")
        raw = json.loads(json.dumps(result, default=self.json_default))
        relative = f"{phase}/{task}.json"
        write_json_atomic(directory / relative, raw)
        return {
            "results": raw.get("results", {}),
            "groups": raw.get("groups", {}),
            "path": relative,
        }

    def _phase(self, phase: str, directory: Path) -> dict:
        config = self.config
        tokenizer = self.trainer.template.tokenizer
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        lm = self.lm_class(
            self.model,
            tokenizer,
            sequence_length=self.trainer.args.max_length,
            pad_token_id=pad,
            dp_group=self.dp_group,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            device=next(self.model.parameters()).device,
            questions_per_forward=config.batch_size,
            max_tokens_per_forward=config.max_tokens_per_forward,
        )
        summary = {"results": {}, "groups": {}, "raw_results": []}
        for task in config.tasks:
            lm.task_name = task
            result = self.simple_evaluate(
                model=lm,
                tasks=[task],
                task_manager=self.task_manager(),
                limit=config.limit,
                num_fewshot=config.num_fewshot,
                log_samples=True,
                bootstrap_iters=1000,
                random_seed=config.seed,
                numpy_random_seed=config.seed,
                torch_random_seed=config.seed,
                fewshot_random_seed=config.seed,
            )

            written = rank0_call(
                partial(
                    self._write_task_result,
                    result,
                    phase=phase,
                    task=task,
                    directory=directory,
                )
            )
            for section in ("results", "groups"):
                if set(summary[section]).intersection(written[section]):
                    raise ValueError(
                        f"Overlapping evaluation task results in {section}."
                    )
                summary[section].update(written[section])
            summary["raw_results"].append(written["path"])
            del result
            gc.collect()
        return summary

    def run(
        self, *, step: int, nominal_tokens: int, consumed_tokens: int
    ) -> dict[str, float]:
        # Callback runs at the optimizer boundary, before the next train batch.
        trainer, config = self.trainer, self.config
        directory = config.output_root / f"tokens-{nominal_tokens:012d}"
        record = {
            "schema_version": 1,
            "status": "incomplete",
            "evaluation": "resident_megatron",
            "step": step,
            "nominal_tokens": nominal_tokens,
            "consumed_tokens": consumed_tokens,
            "initial_model": str(self.initial_model),
            "trajectory": str(trainer.args.output_dir),
            "tasks": list(config.tasks),
            "seed": config.seed,
            "limit": config.limit,
            "num_fewshot": config.num_fewshot,
            "batch_size_questions": config.batch_size,
            "max_tokens_per_forward": config.max_tokens_per_forward,
            "backend": self.backend,
            "masked_experts": list(self.masker.harmful_experts) if self.masker else [],
            "phase_semantics": {
                "full": "resident_model",
                **({"masked": "pre_topk_expert_mask"} if self.masker else {}),
            },
        }

        def begin():
            if (directory / "result.json").exists():
                raise FileExistsError(
                    f"Evaluation milestone already exists: {directory}"
                )
            write_json_atomic(directory / "result.json", record)

        rank0_call(begin)
        started = time.monotonic()
        try:
            with suspend_training_pre_hooks(trainer):
                # The optimizer has consumed these gradients. Its momentum/state,
                # scheduler and the data iterator remain resident and untouched.
                for model in trainer.wrapped_models:
                    if hasattr(model, "zero_grad_buffer"):
                        model.zero_grad_buffer()
                trainer.optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                phases = ("full", "masked") if self.masker else ("full",)
                for phase in phases:
                    mask = (
                        self.masker.masked()
                        if phase == "masked"
                        else contextlib.nullcontext()
                    )
                    with (
                        preserve_eval_state(self.model, self.routers, self.rng_tracker),
                        mask,
                    ):
                        record[phase] = self._phase(phase, directory)
                record.update(
                    status="complete", elapsed_seconds=time.monotonic() - started
                )
                rank0_call(lambda: write_json_atomic(directory / "result.json", record))
                metrics = {}
                for phase in phases:
                    metrics.update(numeric_lm_eval_metrics(record[phase], phase))
                metrics["sft_eval/nominal_tokens"] = nominal_tokens
                metrics["sft_eval/consumed_tokens"] = consumed_tokens
                # Avoid trainer.on_log(): it also consumes/reset training MoE loss
                # trackers. Dispatch only to logging callbacks (TensorBoard/print).
                trainer.call_event("on_log", logs=dict(metrics))
                return metrics
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
