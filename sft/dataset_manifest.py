"""Multi-source SFT dataset manifests (XTuner meta-json style).

A manifest is a JSON object mapping a source name to
``{"annotation": PATH, "media_root": PATH|null, "sample_ratio": float, ...}``.
This module only parses and validates; ``sft/swift_data.py`` samples the entries
into one converted training file.  It must stay free of training-framework
imports so it can be imported outside the training image.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestSource:
    name: str
    annotation: str
    media_root: str
    sample_ratio: float

    @property
    def uses_media(self) -> bool:
        return bool(self.media_root)


def is_dataset_manifest(path: Path) -> bool:
    """A manifest is a ``.json`` file; single JSONL files and directories are not."""
    return path.is_file() and path.suffix == ".json"


def load_dataset_manifest(path: Path) -> list[ManifestSource]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Dataset manifest must be a non-empty JSON object: {path}")
    sources = []
    for name, entry in raw.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("annotation"), str) or not entry["annotation"]:
            raise ValueError(f"Manifest source {name!r} must define a non-empty annotation path")
        if entry.get("text_pretrain"):
            raise ValueError(f"Manifest source {name!r} is text_pretrain; benign SFT manifests must be conversational")
        ratio = entry.get("sample_ratio", 1.0)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Manifest source {name!r} has invalid sample_ratio {ratio!r}")
        media_root = entry.get("media_root") or ""
        if not isinstance(media_root, str):
            raise ValueError(f"Manifest source {name!r} has invalid media_root {media_root!r}")
        sources.append(ManifestSource(name=name, annotation=entry["annotation"], media_root=media_root,
                                      sample_ratio=float(ratio)))
    return sources
