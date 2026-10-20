"""Evaluate one checkpoint, as it is, under a protocol of eval/protocols.yaml.

Every requested task spec without a result directory under
``<node-dir>/eval/<protocol>/`` is evaluated with lm-eval on the vLLM backend,
in one lm-eval run (specs naming the same lm-eval task run in separate passes).
The run's output is then split per spec into ``<node-dir>/eval/<protocol>/<spec>/``:
``results.json`` (lm-eval's results file restricted to the spec's task and its
subtasks/groups), the matching ``samples_<task>_<date>.jsonl`` files and
``run.json``. Everything is written under a temporary directory first; a spec
directory appears only when complete. Requires the separate evaluation
environment (.venv-eval).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from common.run_record import write_run_record
from eval.chat_tokenizer import prepare_chat_tokenizer

PROTOCOLS_PATH = Path(__file__).with_name("protocols.yaml")
INCLUDE_PATH = Path(__file__).with_name("lm_eval_tasks")
# Sections of an lm-eval results file keyed by task or group name.
TASK_SECTIONS = (
    "results", "groups", "group_subtasks", "configs", "versions",
    "n-shot", "higher_is_better", "n-samples", "task_hashes",
)
SPEC_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def load_protocol(name: str) -> dict[str, Any]:
    protocols = yaml.safe_load(PROTOCOLS_PATH.read_text(encoding="utf-8"))
    if name not in protocols:
        raise ValueError(f"Unknown protocol {name!r}; defined: {', '.join(protocols)}")
    protocol = protocols[name]
    if protocol["backend"] != "vllm":
        raise ValueError(f"Unsupported backend {protocol['backend']!r}")
    for spec in protocol["tasks"]:
        if not SPEC_NAME.fullmatch(spec):
            raise ValueError(f"Invalid task spec name {spec!r}")
    return protocol


def library_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def evaluation_passes(specs: Sequence[str], tasks: Mapping[str, Mapping[str, Any]]) -> list[list[str]]:
    """Group specs so that no lm-eval run gets the same task twice."""
    passes: list[list[str]] = []
    for spec in specs:
        for group in passes:
            if all(tasks[other]["task"] != tasks[spec]["task"] for other in group):
                group.append(spec)
                break
        else:
            passes.append([spec])
    return passes


def load_tasks(task_manager, task_defs: Sequence[Mapping[str, Any]]) -> list:
    """Resolve task definitions to names or pre-built Tasks with few-shot overrides.

    Dict specs handed to lm-eval become inline configs without the registered
    YAML (dataset_path is lost), so overridden tasks are built from their
    registered config and the override is set on the Task object.
    """
    resolved: list = []
    for task_def in task_defs:
        name = task_def["task"]
        if "num_fewshot" not in task_def:
            resolved.append(name)
            continue
        loaded = task_manager.load([name])
        if loaded["groups"] or list(loaded["tasks"]) != [name]:
            raise ValueError(f"Few-shot override needs a leaf task, not a group: {name!r}")
        task = loaded["tasks"][name]
        task.set_config(key="num_fewshot", value=int(task_def["num_fewshot"]))
        resolved.append(task)
    return resolved


def build_model_args(
    protocol: Mapping[str, Any], model: Path, data_parallel_size: int, tokenizer: Path | None
) -> dict[str, Any]:
    model_args = {"pretrained": str(model), **protocol["model_args"], "data_parallel_size": data_parallel_size}
    if tokenizer is not None:
        model_args["tokenizer"] = str(tokenizer)
    return model_args


def run_lm_eval(
    protocol: Mapping[str, Any],
    task_defs: Sequence[Mapping[str, Any]],
    model_args: Mapping[str, Any],
    output: Path,
) -> tuple[Path, str]:
    """Run lm-eval once; return lm-eval's results file and its date id."""
    from lm_eval import evaluator
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(INCLUDE_PATH), metadata=dict(model_args))
    tracker = EvaluationTracker(output_path=str(output))
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=dict(model_args),
        tasks=load_tasks(task_manager, task_defs),
        batch_size=protocol["batch_size"],
        log_samples=True,
        evaluation_tracker=tracker,
        apply_chat_template=protocol["apply_chat_template"],
        fewshot_as_multiturn=protocol["fewshot_as_multiturn"],
        task_manager=task_manager,
        **protocol["seeds"],
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for task_name in results["configs"]:
        tracker.save_results_samples(task_name=task_name, samples=samples[task_name])
    written = sorted(output.rglob("results_*.json"))
    if len(written) != 1:
        raise RuntimeError(f"Expected one results_*.json under {output}, found {len(written)}")
    return written[0], tracker.date_id


def task_closure(document: Mapping[str, Any], task: str) -> set[str]:
    """The task plus, for a group, all its subgroups and leaf tasks."""
    subtasks = document.get("group_subtasks") or {}
    names: set[str] = set()
    stack = [task]
    while stack:
        name = stack.pop()
        if name not in names:
            names.add(name)
            stack.extend(subtasks.get(name, ()))
    return names


def split_results(document: Mapping[str, Any], spec_tasks: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Split one lm-eval results document into one document per spec.

    Task-keyed sections keep only the spec's task closure; every other field is
    copied. Each task of the document must belong to exactly one spec.
    """
    owner: dict[str, str] = {}
    closures: dict[str, set[str]] = {}
    for spec, task in spec_tasks.items():
        if task not in (document.get("results") or {}) and task not in (document.get("groups") or {}):
            raise ValueError(f"lm-eval results lack {task!r} (spec {spec!r})")
        closures[spec] = task_closure(document, task)
        for name in closures[spec]:
            if name in owner:
                raise ValueError(f"Specs {owner[name]!r} and {spec!r} both contain lm-eval task {name!r}")
            owner[name] = spec
    for section in TASK_SECTIONS:
        unclaimed = set(document.get(section) or {}) - set(owner)
        if unclaimed:
            raise ValueError(f"lm-eval {section} entries of no requested spec: {sorted(unclaimed)}")
    return {
        spec: {
            key: {name: v for name, v in value.items() if name in names}
            if key in TASK_SECTIONS and isinstance(value, dict) else value
            for key, value in document.items()
        }
        for spec, names in closures.items()
    }


def write_spec(
    document: Mapping[str, Any], samples_dir: Path, date_id: str, spec_dir: Path
) -> list[str]:
    """Write one spec's results.json and move its leaf tasks' samples files."""
    spec_dir.mkdir()
    (spec_dir / "results.json").write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    files = ["results.json"]
    for task in document["configs"]:
        name = f"samples_{task}_{date_id}.jsonl"
        (samples_dir / name).rename(spec_dir / name)
        files.append(name)
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--node-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tasks", help="comma-separated task spec names of the protocol (default: all)")
    parser.add_argument("--data-parallel-size", type=int, required=True)
    parser.add_argument("--processor-from", type=Path,
                        help="checkpoint (normally B) to take missing multimodal processor configs from")
    return parser


# vLLM builds the multimodal Qwen3.5/Gemma 4 architectures and needs these even for
# text-only evaluation; some external checkpoints were released without them.
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json")


def model_view_with_processor(model: Path, source: Path, view: Path) -> list[dict[str, str]]:
    """Link every entry of ``model`` into ``view`` and add the processor configs it lacks."""
    view.mkdir(parents=True)
    for entry in model.iterdir():
        (view / entry.name).symlink_to(entry)
    added = []
    for name in PROCESSOR_FILES:
        if not (model / name).exists() and (source / name).is_file():
            shutil.copy2(source / name, view / name)
            added.append({"file": name, "source": str((source / name).resolve()),
                          "sha256": hashlib.sha256((view / name).read_bytes()).hexdigest()})
    return added


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    model, node_dir = args.model.resolve(), args.node_dir.resolve()
    protocol = load_protocol(args.protocol)
    tasks = protocol["tasks"]
    requested = list(tasks) if args.tasks is None else [s.strip() for s in args.tasks.split(",")]
    unknown = [s for s in requested if s not in tasks]
    if unknown or len(set(requested)) != len(requested):
        raise ValueError(f"Task specs must be unique names of {args.protocol}: unknown {unknown}, given {requested}")
    eval_dir = node_dir / "eval" / args.protocol
    missing = [s for s in requested if not (eval_dir / s).exists()]
    for spec in requested:
        if spec not in missing:
            print(f"Skipping {spec}: {eval_dir / spec} exists", flush=True)
    if not missing:
        return

    wanted = protocol["libraries"]
    versions = library_versions([*wanted, "transformers", "torch"])
    libraries_match = all(versions[name] == str(version) for name, version in wanted.items())
    if not libraries_match:
        print(f"Warning: library versions {versions} differ from {args.protocol} {wanted}", file=sys.stderr)
    settings = {key: value for key, value in protocol.items() if key != "tasks"}

    eval_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".tmp-", dir=eval_dir))
    try:
        processor_added = []
        source_model = model
        if args.processor_from is not None:
            processor_added = model_view_with_processor(model, args.processor_from.resolve(), scratch / "model")
            if processor_added:
                model = scratch / "model"
        tokenizer, chat = None, None
        if protocol["apply_chat_template"]:
            tokenizer, chat = prepare_chat_tokenizer(
                model, scratch / "tokenizer", protocol.get("chat_template_fallback")
            )
        model_args = build_model_args(protocol, model, args.data_parallel_size, tokenizer)
        recorded_model_args = {k: v for k, v in model_args.items() if k != "tokenizer"}
        recorded_model_args["pretrained"] = str(source_model)
        for index, group in enumerate(evaluation_passes(missing, tasks)):
            pass_dir = scratch / f"pass-{index}"
            results_file, date_id = run_lm_eval(protocol, [tasks[s] for s in group], model_args, pass_dir)
            document = json.loads(results_file.read_text(encoding="utf-8"))
            parts = split_results(document, {spec: tasks[spec]["task"] for spec in group})
            for spec in group:
                part = parts[spec]
                spec_dir = scratch / spec
                files = write_spec(part, results_file.parent, date_id, spec_dir)
                write_run_record(
                    spec_dir,
                    step="eval_lm",
                    inputs={"model": source_model},
                    config={"protocol": args.protocol, "settings": settings, "spec": spec, "task": tasks[spec]},
                    args={
                        "protocol": args.protocol,
                        "tasks": requested,
                        "evaluated_together": group,
                        "data_parallel_size": args.data_parallel_size,
                        "model_args": recorded_model_args,
                        "chat_tokenizer": chat,
                        "processor_files_added": processor_added,
                        "include_path": str(INCLUDE_PATH.relative_to(INCLUDE_PATH.parents[1])),
                    },
                    results={
                        "protocol": args.protocol,
                        "spec": spec,
                        "library_versions": versions,
                        "protocol_libraries": wanted,
                        "libraries_match": libraries_match,
                        "n-shot": part.get("n-shot", {}),
                        "n-samples": part.get("n-samples", {}),
                        "files": files,
                    },
                )
                if (eval_dir / spec).exists():
                    raise FileExistsError(f"{eval_dir / spec} appeared during evaluation")
                spec_dir.rename(eval_dir / spec)
                print(f"Wrote {eval_dir / spec}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
