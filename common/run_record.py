"""Write the per-step ``run.json`` provenance record.

Each step records what it was run with and what it produced. Records of the
inputs are embedded as found, so the full lineage travels with the output;
nothing here validates them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.io_utils import read_json, write_json_atomic
from common.manifest_names import RUN_RECORD

REPO_DIR = Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_DIR), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit + ("-dirty" if dirty else "")


def input_record(path: str | Path) -> Any:
    record = Path(path) / RUN_RECORD
    return read_json(record) if record.is_file() else None


def write_run_record(
    output_dir: str | Path,
    *,
    step: str,
    inputs: Mapping[str, str | Path],
    config: Mapping[str, Any] | None = None,
    args: Mapping[str, Any] | None = None,
    results: Mapping[str, Any] | None = None,
) -> Path:
    resolved = {name: str(Path(value).resolve()) for name, value in inputs.items()}
    record = {
        "step": step,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": resolved,
        "config": dict(config) if config is not None else None,
        "args": dict(args) if args is not None else None,
        "results": dict(results or {}),
        "input_records": {
            name: input_record(path) for name, path in resolved.items() if Path(path).is_dir()
        },
    }
    path = Path(output_dir) / RUN_RECORD
    write_json_atomic(path, record, default=str)
    return path
