from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_DATA_FORMAT_TYPES = {"pretrain", "pretrain_text_column"}
SUPPORTED_PATH_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt", ".parquet"}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    files: tuple[Path, ...]
    data_format_type: str
    role: str
    sample_ratio: float
    label_weight: float
    text_column: str | None = None
    row_split_manifest: Path | None = None
    row_split: str | None = None


def _discover_data_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.exists():
        raise FileNotFoundError(f"Dataset annotation path does not exist: {path}")

    subset_manifest = path / "subset_manifest.json"
    if subset_manifest.is_file():
        payload = json.loads(subset_manifest.read_text(encoding="utf-8"))
        source = Path(payload["source"])
        selection = payload["selection"]
        if not isinstance(selection, list) or not selection:
            raise ValueError(f"Invalid or empty selection in {subset_manifest}.")
        files = tuple(source / str(name) for name in selection)
    else:
        suffixes = SUPPORTED_PATH_EXTENSIONS
        files = tuple(
            sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() in suffixes)
        )
        if not files:
            files = tuple(sorted(child for child in path.rglob("*") if child.is_file() and child.suffix.lower() in suffixes))
    if not files:
        expected = ", ".join(sorted(SUPPORTED_PATH_EXTENSIONS))
        raise FileNotFoundError(f"No supported data files ({expected}) found under dataset annotation path: {path}")
    missing = [file for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset manifest references missing files: {missing[:3]}")
    return files


def _require_role(name: str, entry: dict[str, Any]) -> str:
    role = entry.get("role")
    if role not in {"std", "harmful"}:
        raise ValueError(f"datasets.{name}.role must be 'std' or 'harmful'.")
    return str(role)


def _source_from_path(name: str, entry: dict[str, Any]) -> DatasetSource:
    if "path" not in entry:
        raise ValueError(f"datasets.{name} must define path.")
    data_format_type = str(entry.get("data_format_type") or ("pretrain_text_column" if entry.get("text_column") else "pretrain"))
    if data_format_type not in SUPPORTED_DATA_FORMAT_TYPES:
        raise ValueError(f"Unsupported data_format_type for datasets.{name}: {data_format_type!r}.")
    return DatasetSource(
        name=name,
        files=_discover_data_files(Path(entry["path"])),
        data_format_type=data_format_type,
        role=_require_role(name, entry),
        sample_ratio=float(entry.get("sample_ratio", 1.0)),
        label_weight=float(entry.get("label_weight", 1.0)),
        text_column=str(entry.get("text_column")) if entry.get("text_column") is not None else None,
        row_split_manifest=(
            Path(entry["row_split_manifest"])
            if entry.get("row_split_manifest") is not None
            else None
        ),
        row_split=(
            str(entry["row_split"])
            if entry.get("row_split") is not None
            else None
        ),
    )


def _validate_dataset_sources(sources: list[DatasetSource]) -> None:
    if not any(source.role == "std" for source in sources):
        raise ValueError("At least one std dataset source is required.")
    if not any(source.role == "harmful" for source in sources):
        raise ValueError("At least one harmful dataset source is required.")
    if any(source.label_weight <= 0 or source.sample_ratio <= 0 for source in sources):
        raise ValueError("All dataset label_weight and sample_ratio values must be positive.")


def load_dataset_sources(config: dict[str, Any]) -> list[DatasetSource]:
    datasets_cfg = config.get("datasets")
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        raise ValueError("Config must define a non-empty datasets mapping.")

    sources: list[DatasetSource] = []
    for name, entry in datasets_cfg.items():
        if not isinstance(entry, dict):
            raise ValueError(f"datasets.{name} must be a mapping.")
        sources.append(_source_from_path(str(name), entry))

    _validate_dataset_sources(sources)
    return sources
