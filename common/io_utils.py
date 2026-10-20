"""Local JSON and bounded safetensors IO shared by training and transforms."""
from __future__ import annotations

import json
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any, *, sort_keys: bool = False,
                      ensure_ascii: bool = False, default=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Retain an existing file's access policy. New files follow the caller's
    # umask, unlike NamedTemporaryFile's unconditional 0600 (which makes root
    # container outputs unreadable to their submitting user on shared storage).
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp") as directory:
        temporary = Path(directory) / path.name
        with temporary.open(mode="w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=ensure_ascii, indent=2, sort_keys=sort_keys, default=default)
            handle.write("\n")
        if existing_mode is not None:
            temporary.chmod(existing_mode)
        temporary.replace(path)


def tensor_nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_shard(model_dir: Path, shard_name: str, *, with_metadata: bool = False):
    from safetensors import safe_open
    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = handle.metadata()
    return (tensors, metadata) if with_metadata else tensors


def prepare_output_dir(output_dir: Path, *, overwrite: bool = False) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def sanitize_filename(value: str, *, fallback: str = "dataset") -> str:
    name = re.sub(r"[/:\s]+", "_", value)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("._") or fallback


class ShardedSafetensorsWriter:
    """Append tensors in caller order, preserving each caller's shard naming.

    A tensor larger than the cap occupies one shard. ``weight_map`` can be an
    existing index when appending passthrough weights to a native HF save.
    """
    def __init__(self, output_dir: Path, *, max_shard_bytes: int = 2_000_000_000,
                 prefix: str, start_index: int = 0, weight_map: dict | None = None):
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        self.output_dir = output_dir
        self.max_shard_bytes = max_shard_bytes
        self.prefix = prefix
        self.start_index = start_index
        self.weight_map = weight_map if weight_map is not None else {}
        self.shards: list[str] = []
        self.total_size = 0
        self.pending: dict = {}
        self.pending_size = 0

    def add(self, key, tensor) -> None:
        size = tensor_nbytes(tensor)
        if self.pending_size + size > self.max_shard_bytes:
            self.flush()
        self.pending[key] = tensor.clone().contiguous()
        self.pending_size += size

    def flush(self) -> None:
        from safetensors.torch import save_file
        if not self.pending:
            return
        name = f"{self.prefix}-{self.start_index + len(self.shards):05d}.safetensors"
        save_file(self.pending, str(self.output_dir / name), metadata={"format": "pt"})
        self.weight_map.update(dict.fromkeys(self.pending, name))
        self.shards.append(name)
        self.total_size += self.pending_size
        self.pending, self.pending_size = {}, 0
