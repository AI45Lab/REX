from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

SUMMARY_METRICS = {
    "mmlu": ("acc",),
    "mmlu_pro": ("exact_match",),
    "wmdp": ("acc",),
    "wmdp_bio": ("acc",),
    "wmdp_cyber": ("acc",),
}
DEFAULT_SUMMARY_METRICS = ("acc", "exact_match")


def iter_task_results(document: Mapping, *, sections=("results", "groups")):
    """Yield task, mapping and source section, preserving source order."""
    for section_name in sections:
        section = document.get(section_name)
        if not isinstance(section, dict):
            continue
        for task_name, values in section.items():
            if isinstance(values, dict):
                yield task_name, values, section_name


def result_sections(results: Mapping, *, sections=("results", "groups")) -> dict:
    merged = {}
    for task, values, _ in iter_task_results(results, sections=sections):
        merged.setdefault(task, values)
    return merged


def task_result(document: Mapping, task: str) -> tuple[dict, str]:
    for name, values, section in iter_task_results(document):
        if name == task:
            return values, section
    raise KeyError(task)


@dataclass(frozen=True)
class Metric:
    value: float
    stderr: Any
    sample_count: Any


def select_metric(
    values: Mapping, key: str, *, require_finite: bool = True
) -> Metric | None:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if require_finite and not math.isfinite(float(value)):
        return None
    return Metric(
        float(value),
        values.get(key.replace(",", "_stderr,", 1)),
        values.get("sample_count"),
    )


def read_metric(document: Mapping, task: str, key: str) -> Metric:
    values, _ = task_result(document, task)
    metric = select_metric(values, key)
    if metric is None:
        raise ValueError(f"Missing/non-finite metric {task}/{key}")
    return metric


def numeric_lm_eval_metrics(results: Mapping, prefix: str) -> dict[str, float]:
    # Training logs merge per metric: a groups-only key may supplement results.
    metrics = {}
    for task, values, _ in iter_task_results(results):
        for name in values:
            metric = select_metric(values, name)
            if metric is not None:
                key = f"lm_eval/{prefix}/{task}/{str(name).replace(',', '_')}"
                metrics.setdefault(key, metric.value)
    return metrics


def find_summary_metric(
    results: dict[str, Any],
    task_name: str,
) -> tuple[str, float] | None:
    task_results = result_sections(results).get(task_name)
    if task_results is None:
        return None
    preferred = SUMMARY_METRICS.get(task_name, DEFAULT_SUMMARY_METRICS)
    for metric_base in preferred:
        for metric_name, value in task_results.items():
            current_base = str(metric_name).split(",", 1)[0]
            if current_base.endswith("_stderr") or ",stderr" in str(metric_name):
                continue
            metric = select_metric(task_results, metric_name)
            if current_base == metric_base and metric is not None:
                return metric_base, metric.value
    return None


def summary_drop(
    *,
    full_results: dict[str, Any],
    ablated_results: dict[str, Any],
    task_name: str,
) -> tuple[str, float, float, float] | None:
    full_metric = find_summary_metric(full_results, task_name)
    ablated_metric = find_summary_metric(ablated_results, task_name)
    if full_metric is None or ablated_metric is None or full_metric[1] == 0.0:
        return None
    if full_metric[0] != ablated_metric[0]:
        raise ValueError(
            f"Full and ablated lm-eval metrics differ for {task_name!r}: "
            f"{full_metric[0]!r} != {ablated_metric[0]!r}."
        )
    metric_name, full_value = full_metric
    ablated_value = ablated_metric[1]
    drop = (full_value - ablated_value) / abs(full_value)
    return metric_name, full_value, ablated_value, drop


def evaluation_phase(document: Mapping, phase: str = "full") -> Mapping:
    """Resolve either a direct harness result or a merged full/ablated result."""
    if "results" in document or "groups" in document:
        return document
    return document[phase]
