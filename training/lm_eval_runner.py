from __future__ import annotations

import contextlib
import os
import random
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.lm_eval_wrapper import RexFSDPLMEvalWrapper
from eval.lm_eval_io import SUMMARY_METRICS, numeric_lm_eval_metrics, summary_drop

DEFAULT_LM_EVAL_TASKS = ["mmlu", "wmdp"]
LM_EVAL_LIMIT_UNSET = object()


def _set_activation_mask_enabled(activation_masker: Any | None, enabled: bool) -> None:
    if activation_masker is None:
        return
    if enabled:
        activation_masker.enable()
    else:
        activation_masker.disable()


def _load_lm_eval_modules() -> tuple[Any, Any]:
    try:
        import lm_eval
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise RuntimeError(
            "lm-eval validation requested but lm_eval is not installed. Run `uv sync` first."
        ) from exc
    return lm_eval, TaskManager


def parse_lm_eval_tasks(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_LM_EVAL_TASKS)
    if isinstance(value, str):
        return [item for item in value.replace(",", " ").split() if item]
    if isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    ):
        return list(value)
    raise ValueError("lm_eval_tasks must be a non-empty string/list of task names.")


# Output types the REX wrapper can actually serve. `multiple_choice` is built on
# loglikelihood requests; `generate_until` / `loglikelihood_rolling` are not implemented.
_SUPPORTED_LM_EVAL_OUTPUT_TYPES = {"loglikelihood", "multiple_choice"}


def _assert_loglikelihood_only(tasks: list[str], lm_eval_module: Any) -> None:
    """Fail fast if any configured task requires generation, before training-time eval runs."""
    import lm_eval.tasks  # noqa: F401 -- ensure the submodule is loaded before attribute access

    task_dict = lm_eval_module.tasks.get_task_dict(tasks)

    def leaf_tasks(node: Any) -> Any:
        if isinstance(node, dict):
            for value in node.values():
                yield from leaf_tasks(value)
        else:
            yield node

    offenders: list[str] = []
    for task in leaf_tasks(task_dict):
        output_type = getattr(task, "OUTPUT_TYPE", None)
        if output_type not in _SUPPORTED_LM_EVAL_OUTPUT_TYPES:
            name = getattr(getattr(task, "config", None), "task", None) or getattr(
                task, "name", "?"
            )
            offenders.append(f"{name} (output_type={output_type})")
    if offenders:
        raise ValueError(
            "REX lm-eval validation only supports loglikelihood tasks. "
            f"Unsupported tasks: {', '.join(offenders)}."
        )


def parse_lm_eval_limit(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("lm_eval_limit must be an int, float, or null.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return float(value) if "." in value else int(value)
    raise ValueError("lm_eval_limit must be an int, float, or null.")


@dataclass(frozen=True)
class LMEvalTaskBatch:
    tasks: tuple[str, ...]
    limit: int | float | None


def build_lm_eval_task_batches(
    tasks: list[str],
    *,
    default_limit: int | float | None,
    task_limits: Any = None,
) -> list[LMEvalTaskBatch]:
    if task_limits is None:
        return [LMEvalTaskBatch(tasks=tuple(tasks), limit=default_limit)]
    if not isinstance(task_limits, Mapping):
        raise ValueError(
            "lm_eval_task_limits must be a mapping from task name to int, float, or null."
        )

    unknown_tasks = sorted(set(task_limits) - set(tasks))
    if unknown_tasks:
        raise ValueError(
            "lm_eval_task_limits contains tasks not present in lm_eval_tasks: "
            f"{', '.join(str(task) for task in unknown_tasks)}."
        )

    batches: list[LMEvalTaskBatch] = []
    for task in tasks:
        limit = (
            parse_lm_eval_limit(task_limits[task])
            if task in task_limits
            else default_limit
        )
        matching_index = next(
            (index for index, batch in enumerate(batches) if batch.limit == limit),
            None,
        )
        if matching_index is None:
            batches.append(LMEvalTaskBatch(tasks=(task,), limit=limit))
            continue
        matching = batches[matching_index]
        batches[matching_index] = LMEvalTaskBatch(
            tasks=(*matching.tasks, task), limit=limit
        )
    return batches


def should_use_distributed_lm_eval(accelerator: Any, limit: int | float | None) -> bool:
    if accelerator.num_processes <= 1:
        return False
    if isinstance(limit, int) and 0 < limit < accelerator.num_processes:
        return False
    if isinstance(limit, float) and 0.0 < limit < 1.0:
        return False
    return True


def add_summary_drop_metrics(
    metrics: dict[str, float],
    *,
    full_results: dict[str, Any],
    ablated_results: dict[str, Any],
) -> None:
    for task_name in SUMMARY_METRICS:
        values = summary_drop(
            full_results=full_results,
            ablated_results=ablated_results,
            task_name=task_name,
        )
        if values is None:
            continue
        metrics[f"lm_eval/summary/{task_name}_drop"] = values[3]


@dataclass(frozen=True)
class LMEvalPhaseResult:
    results: dict[str, Any] | None
    metrics: dict[str, float]
    elapsed_seconds: float


def merge_lm_eval_phase_results(results: list[LMEvalPhaseResult]) -> LMEvalPhaseResult:
    mergeable_sections = {
        "configs",
        "group_subtasks",
        "groups",
        "higher_is_better",
        "n-samples",
        "n-shot",
        "results",
        "samples",
        "task_hashes",
        "versions",
    }
    merged_results: dict[str, Any] | None = None
    merged_metrics: dict[str, float] = {}
    elapsed_seconds = 0.0
    for phase_result in results:
        elapsed_seconds += phase_result.elapsed_seconds
        merged_metrics.update(phase_result.metrics)
        if phase_result.results is None:
            continue
        if merged_results is None:
            merged_results = {}
        for key, value in phase_result.results.items():
            if key in mergeable_sections and isinstance(value, dict):
                existing = merged_results.setdefault(key, {})
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"Cannot merge lm-eval result section {key!r} with incompatible values."
                    )
                overlap = set(existing).intersection(value)
                if overlap:
                    raise ValueError(
                        f"Duplicate lm-eval result entries while merging {key!r}: {', '.join(sorted(overlap))}."
                    )
                existing.update(value)
            else:
                merged_results.setdefault(key, value)
    return LMEvalPhaseResult(
        results=merged_results,
        metrics=merged_metrics,
        elapsed_seconds=elapsed_seconds,
    )


def default_lm_eval_request_cache_dir() -> Path:
    override = os.environ.get("LM_HARNESS_CACHE_PATH")
    if override:
        return Path(override).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = (
        Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    )
    return cache_root / "rex" / "lm_eval_requests"


def _set_lm_eval_cache_path(cache_dir: Path) -> None:
    try:
        import lm_eval.caching.cache as lm_eval_cache
    except (ImportError, AttributeError):
        return
    lm_eval_cache.PATH = str(cache_dir)


def ensure_lm_eval_request_cache_dir() -> Path:
    cache_dir = default_lm_eval_request_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LM_HARNESS_CACHE_PATH", str(cache_dir))
    _set_lm_eval_cache_path(cache_dir)
    return cache_dir


def _wait_for_everyone_if_possible(accelerator: Any) -> None:
    try:
        accelerator.wait_for_everyone()
    except RuntimeError:
        return


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_device = (
        torch.cuda.current_device()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    cuda_state = (
        torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_device is not None and cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def run_single_lm_eval(
    *,
    lm_eval_module: Any,
    wrapper: RexFSDPLMEvalWrapper,
    config: dict[str, Any],
    tasks: list[str],
    limit: int | float | None,
    task_manager: Any | None = None,
) -> dict[str, Any] | None:
    cache_requests = config.get("lm_eval_cache_requests")
    if cache_requests is None:
        cache_requests = limit is None
    with preserve_rng_state():
        return lm_eval_module.simple_evaluate(
            model=wrapper,
            tasks=tasks,
            batch_size=config.get("lm_eval_batch_size", 1),
            limit=limit,
            cache_requests=bool(cache_requests),
            rewrite_requests_cache=False,
            bootstrap_iters=int(config.get("lm_eval_bootstrap_iters", 0)),
            log_samples=bool(config.get("lm_eval_log_samples", False)),
            random_seed=int(config.get("lm_eval_random_seed", config.get("seed", 42))),
            numpy_random_seed=int(
                config.get("lm_eval_numpy_seed", config.get("seed", 42))
            ),
            torch_random_seed=int(
                config.get("lm_eval_torch_seed", config.get("seed", 42))
            ),
            fewshot_random_seed=int(
                config.get("lm_eval_fewshot_seed", config.get("seed", 42))
            ),
            confirm_run_unsafe_code=bool(
                config.get("lm_eval_confirm_run_unsafe_code", False)
            ),
            task_manager=task_manager,
            gen_kwargs=None,
        )


class LMEvalRunner:
    def __init__(
        self,
        *,
        accelerator: Any,
        model: torch.nn.Module,
        tokenizer: Any,
        activation_masker: Any | None,
        config: dict[str, Any],
        routing_collector: Any | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.activation_masker = activation_masker
        self.config = config
        self.routing_collector = routing_collector
        self.cache_dir = ensure_lm_eval_request_cache_dir()
        self.lm_eval, task_manager_cls = _load_lm_eval_modules()
        self.task_manager = task_manager_cls()
        self.tasks = parse_lm_eval_tasks(config.get("lm_eval_tasks"))
        if not self.tasks:
            raise ValueError(
                "lm-eval validation requires at least one task in lm_eval_tasks."
            )
        self.default_limit = parse_lm_eval_limit(config.get("lm_eval_limit"))
        self.task_batches = build_lm_eval_task_batches(
            self.tasks,
            default_limit=self.default_limit,
            task_limits=config.get("lm_eval_task_limits"),
        )
        _assert_loglikelihood_only(self.tasks, self.lm_eval)
        self._warned_replicated_eval = False
        accelerator.print(f"lm_eval request cache: {self.cache_dir}")

    def _resolve_task_batches(
        self,
        lm_eval_limit: int | float | None | object,
    ) -> list[LMEvalTaskBatch]:
        if lm_eval_limit is LM_EVAL_LIMIT_UNSET:
            return self.task_batches
        return [LMEvalTaskBatch(tasks=tuple(self.tasks), limit=lm_eval_limit)]

    def _make_wrapper(self, *, limit: int | float | None) -> RexFSDPLMEvalWrapper:
        distributed = should_use_distributed_lm_eval(self.accelerator, limit)
        if (
            not distributed
            and self.accelerator.num_processes > 1
            and not self._warned_replicated_eval
        ):
            self.accelerator.print(
                "lm_eval_limit is smaller than world size; running replicated lm-eval on every rank for this smoke pass."
            )
            self._warned_replicated_eval = True
        return RexFSDPLMEvalWrapper(
            accelerator=self.accelerator,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.config.get("lm_eval_batch_size", 1),
            distributed=distributed,
        )

    def _routing_suspended(self) -> contextlib.AbstractContextManager[Any]:
        if self.routing_collector is not None:
            return self.routing_collector.suspend()
        return contextlib.nullcontext()

    def run_phase(
        self,
        *,
        phase: str,
        ablated: bool,
        limit: int | float | None,
        tasks: list[str] | tuple[str, ...] | None = None,
    ) -> LMEvalPhaseResult:
        if ablated and self.activation_masker is None:
            raise ValueError("Ablated lm-eval requires an ActivationMasker.")

        was_training = self.model.training
        previous_mask_enabled = getattr(self.activation_masker, "_enabled", False)
        try:
            _set_activation_mask_enabled(self.activation_masker, ablated)

            phase_tasks = list(tasks) if tasks is not None else self.tasks
            wrapper = self._make_wrapper(limit=limit)
            limit_label = "full" if limit is None else str(limit)
            self.accelerator.print(
                f"lm_eval {phase}: tasks={','.join(phase_tasks)} limit={limit_label}"
            )
            started = time.time()
            self.model.eval()
            results = run_single_lm_eval(
                lm_eval_module=self.lm_eval,
                wrapper=wrapper,
                config=self.config,
                tasks=phase_tasks,
                limit=limit,
                task_manager=self.task_manager,
            )
            self.accelerator.wait_for_everyone()
            elapsed = time.time() - started
            metrics = (
                numeric_lm_eval_metrics(results or {}, phase)
                if self.accelerator.is_main_process
                else {}
            )
            return LMEvalPhaseResult(
                results=results, metrics=metrics, elapsed_seconds=elapsed
            )
        finally:
            _set_activation_mask_enabled(self.activation_masker, previous_mask_enabled)
            self.model.train(was_training)

    def _validation_metrics(
        self,
        *,
        full_result: LMEvalPhaseResult,
        ablated_result: LMEvalPhaseResult,
    ) -> dict[str, float]:
        if (
            not self.accelerator.is_main_process
            or full_result.results is None
            or ablated_result.results is None
        ):
            return {}
        metrics: dict[str, float] = {}
        metrics.update(full_result.metrics)
        metrics.update(ablated_result.metrics)
        add_summary_drop_metrics(
            metrics,
            full_results=full_result.results,
            ablated_results=ablated_result.results,
        )
        return metrics

    def _log_validation_metrics(
        self, metrics: dict[str, float], *, step: int, log_metrics: bool
    ) -> None:
        if not metrics:
            return
        if log_metrics:
            self.accelerator.log(metrics, step=step)
        selection_keys = {
            "lm_eval/full/wmdp_bio/acc_none",
            "lm_eval/ablated/wmdp_bio/acc_none",
            "lm_eval/full/mmlu/acc_none",
            "lm_eval/ablated/mmlu/acc_none",
        }
        summary = ", ".join(
            f"{key}={value:.4f}"
            for key, value in sorted(metrics.items())
            if key.startswith("lm_eval/summary/") or key in selection_keys
        )
        self.accelerator.print(f"lm_eval step {step}: {summary or 'logged metrics'}")

    def run_validation(
        self,
        *,
        step: int,
        lm_eval_limit: int | float | None | object = LM_EVAL_LIMIT_UNSET,
        log_metrics: bool = True,
    ) -> dict[str, float]:
        task_batches = self._resolve_task_batches(lm_eval_limit)
        was_training = self.model.training
        try:
            with self._routing_suspended():
                full_result = merge_lm_eval_phase_results(
                    [
                        self.run_phase(
                            phase="full",
                            ablated=False,
                            limit=batch.limit,
                            tasks=batch.tasks,
                        )
                        for batch in task_batches
                    ]
                )
                ablated_result = merge_lm_eval_phase_results(
                    [
                        self.run_phase(
                            phase="ablated",
                            ablated=True,
                            limit=batch.limit,
                            tasks=batch.tasks,
                        )
                        for batch in task_batches
                    ]
                )

            metrics = self._validation_metrics(
                full_result=full_result, ablated_result=ablated_result
            )
            self._log_validation_metrics(metrics, step=step, log_metrics=log_metrics)
            return metrics
        finally:
            _set_activation_mask_enabled(self.activation_masker, False)
            self.model.train(was_training)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _wait_for_everyone_if_possible(self.accelerator)
