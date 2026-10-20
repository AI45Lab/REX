"""Delete the weight files of a recorded node directory, keeping everything else.

Invoked by ``scripts/drop_weights.sh``. A temporary node (A0, A1, A2, an
excised A3, ...) keeps its ``run.json``, evaluation and training records; only
``*.safetensors`` and ``model.safetensors.index.json`` go, at any depth (save
milestones included). Symlinks are removed as links, never followed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from common.manifest_names import RUN_RECORD

REPO_DIR = Path(__file__).resolve().parents[1]
WEIGHT_INDEX = "model.safetensors.index.json"
# Base models and data stay on shared storage; never strip anything there.
PROTECTED_DIRS = ("checkpoints", "data", "internal", "internal/checkpoints", "internal/data")


def is_weight_file(name: str) -> bool:
    return name.endswith(".safetensors") or name == WEIGHT_INDEX


def check_node_dir(node_dir: Path) -> None:
    if not node_dir.is_dir():
        raise SystemExit(f"Node directory does not exist: {node_dir}")
    if not (node_dir / RUN_RECORD).is_file():
        raise SystemExit(f"Refusing to drop weights of an unrecorded directory (no {RUN_RECORD}): {node_dir}")
    if node_dir == REPO_DIR or node_dir in REPO_DIR.parents:
        raise SystemExit(f"Refusing to drop weights in the repository root or above: {node_dir}")
    for name in PROTECTED_DIRS:
        protected = (REPO_DIR / name).resolve()
        if node_dir == protected or protected in node_dir.parents:
            raise SystemExit(f"Refusing to drop weights under {protected}: {node_dir}")


def weight_files(node_dir: Path) -> list[Path]:
    found = []
    for root, _, files in os.walk(node_dir):  # does not descend into symlinked directories
        found.extend(Path(root) / name for name in files if is_weight_file(name))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("node_dir", type=Path)
    args = parser.parse_args()
    node_dir = args.node_dir.resolve()
    check_node_dir(node_dir)
    removed_bytes = 0
    files = weight_files(node_dir)
    for path in files:
        removed_bytes += path.lstat().st_size
        path.unlink()
    print(f"Removed {len(files)} weight files ({removed_bytes / 1e9:.2f} GB) from {node_dir}")


if __name__ == "__main__":
    main()
