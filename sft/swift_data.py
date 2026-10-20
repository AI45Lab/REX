"""Convert SFT sources into one shuffled ms-swift ``messages`` JSONL.

Input is either a multi-source manifest (see ``sft/dataset_manifest.py``) or a
single JSONL file/directory. Records are OpenAI-style ``{"messages": [...],
"tools": [...]}`` or plain ``{"text": ...}`` (the latter becomes one assistant
turn, used by ``megatron pt`` for target relearning).

Swift templates do not read ``reasoning_content``, so it is inlined into the
assistant content in the chat format of the target model family. The wrapper is
therefore a conversion parameter, and one converted file serves one family.

Sampling follows XTuner's ``sample_ratio``: ratio <= 1 draws that fraction
without replacement, ratio > 1 repeats the source floor(ratio) times plus a
random remainder. Every source uses its own RNG derived from the seed and its
name, so adding a source does not reshuffle the others. The output order is a
seeded global permutation.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from sft.dataset_manifest import is_dataset_manifest, load_dataset_manifest

# How each model family's chat template spells an assistant thinking block.
THINKING_WRAPPERS = {
    "qwen3_5": ("<think>\n", "\n</think>\n\n"),
    "gemma4": ("<|channel>thought\n", "<channel|>"),
}

ROLE_ALIASES = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool", "function": "tool"}


class SkipRecord(Exception):
    """A record that cannot be represented; the reason is counted, not hidden."""


def annotation_files(annotation: str | Path) -> list[Path]:
    path = Path(annotation)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.jsonl") if p.is_file())
        if files:
            return files
    raise FileNotFoundError(f"No JSONL files under annotation path: {path}")


def text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                raise SkipRecord("non_text_content")
        return "".join(parts)
    raise SkipRecord("unsupported_content_type")


def tool_call_message(call: Any) -> dict[str, str]:
    function = call.get("function", call) if isinstance(call, dict) else None
    if not isinstance(function, dict) or not function.get("name"):
        raise SkipRecord("malformed_tool_call")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass  # keep the raw string; the agent template dumps it verbatim
    return {"role": "tool_call", "content": json.dumps({"name": function["name"], "arguments": arguments},
                                                      ensure_ascii=False)}


def convert_record(record: Any, thinking_format: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise SkipRecord("not_an_object")
    if "messages" not in record and isinstance(record.get("text"), str):
        if not record["text"]:
            raise SkipRecord("empty_text")
        return {"messages": [{"role": "assistant", "content": record["text"]}]}
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise SkipRecord("missing_messages")
    think_open, think_close = THINKING_WRAPPERS[thinking_format]
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise SkipRecord("malformed_message")
        role = ROLE_ALIASES.get(message.get("role"))
        if role is None:
            raise SkipRecord("unknown_role")
        content = text_content(message.get("content"))
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                content = f"{think_open}{reasoning.strip()}{think_close}{content}"
            if content:
                messages.append({"role": "assistant", "content": content})
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise SkipRecord("malformed_tool_call")
            messages.extend(tool_call_message(call) for call in calls)
            if not content and not calls:
                raise SkipRecord("empty_assistant")
        else:
            messages.append({"role": role, "content": content})
    if not any(m["role"] in {"assistant", "tool_call"} for m in messages):
        raise SkipRecord("no_assistant_turn")
    converted: dict[str, Any] = {"messages": messages}
    tools = record.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise SkipRecord("malformed_tools")
        converted["tools"] = tools
    return converted


def source_rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def sample_indices(count: int, ratio: float, rng: random.Random) -> list[int]:
    whole, fraction = divmod(ratio, 1.0)
    indices = list(range(count)) * int(whole)
    extra = round(count * fraction)
    indices += rng.sample(range(count), extra)
    return sorted(indices)


def scan_offsets(path: str) -> tuple[str, list[int], int]:
    """Byte offset of every non-blank line, so pass two can seek directly."""
    offsets, position = [], 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)
    return path, offsets, position


def read_and_convert(task: tuple[str, int, str, str]) -> tuple[str, str | None, str | None]:
    source, offset, path, thinking_format = task
    with open(path, "rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    try:
        converted = convert_record(json.loads(line), thinking_format)
    except json.JSONDecodeError:
        return source, None, "invalid_json"
    except SkipRecord as error:
        return source, None, str(error)
    return source, json.dumps(converted, ensure_ascii=False), None


def build(data_path: Path, output_dir: Path, *, thinking_format: str, seed: int, workers: int) -> dict[str, Any]:
    """Write ``output_dir/train.jsonl`` and return the conversion counts."""
    if is_dataset_manifest(data_path):
        sources = [(s.name, s.annotation, s.sample_ratio) for s in load_dataset_manifest(data_path)
                   if not s.uses_media or _refuse_media(s.name)]
    else:
        sources = [(data_path.name, str(data_path), 1.0)]

    files_by_source = {name: annotation_files(annotation) for name, annotation, _ in sources}
    all_files = [str(p) for files in files_by_source.values() for p in files]
    with Pool(workers) as pool:
        scanned = {path: (offsets, size) for path, offsets, size in pool.imap_unordered(scan_offsets, all_files)}

    tasks: list[tuple[str, int, str, str]] = []
    source_stats: dict[str, dict[str, Any]] = {}
    for name, annotation, ratio in sources:
        lines = [(str(p), off) for p in files_by_source[name] for off in scanned[str(p)][0]]
        chosen = sample_indices(len(lines), ratio, source_rng(seed, name))
        tasks.extend((name, lines[i][1], lines[i][0], thinking_format) for i in chosen)
        source_stats[name] = {
            "sample_ratio": ratio, "input_records": len(lines), "sampled_records": len(chosen),
            "input_files": len(files_by_source[name]),
            "input_bytes": sum(scanned[str(p)][1] for p in files_by_source[name]),
        }
    random.Random(seed).shuffle(tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "train.jsonl"
    temporary = output.with_name(output.name + ".tmp")
    written, skipped = Counter(), Counter()
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle, Pool(workers) as pool:
        for source, line, reason in pool.imap(read_and_convert, tasks, chunksize=256):
            if line is None:
                skipped[(source, reason)] += 1
                continue
            data = line + "\n"
            handle.write(data)
            digest.update(data.encode("utf-8"))
            written[source] += 1
    temporary.replace(output)

    for name, stats in source_stats.items():
        stats["written_records"] = written[name]
        stats["skipped"] = {reason: n for (source, reason), n in sorted(skipped.items()) if source == name}
    return {
        "output": str(output), "sha256": digest.hexdigest(),
        "records": sum(written.values()), "skipped_records": sum(skipped.values()),
        "thinking_format": thinking_format, "seed": seed, "sources": source_stats,
    }


def _refuse_media(name: str) -> bool:
    raise ValueError(f"Manifest source {name!r} has a media_root; the swift SFT path is text-only")
