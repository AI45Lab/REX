"""Evaluation-only chat tokenizer, without changing checkpoint assets."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

EVAL_DIR = Path(__file__).parent


def prepare_chat_tokenizer(
    model: Path, output: Path, fallback: Mapping[str, Any] | None
) -> tuple[Path | None, dict[str, Any]]:
    """Return (tokenizer override or None, description of what was used).

    A tokenizer that has a chat template is used as is. Otherwise, if the
    protocol declares a ``chat_template_fallback`` for this model type, a copy
    of the tokenizer with that template is saved to ``output``. Any other
    missing template fails.
    """
    from transformers import AutoTokenizer

    # Remote tokenizer code may print diagnostics; keep stdout for lm-eval.
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
        if tokenizer.chat_template:
            return None, {"fallback_applied": False}
        model_type = json.loads((model / "config.json").read_text()).get("model_type")
        if fallback is None or model_type != fallback["model_type"]:
            raise ValueError(f"Tokenizer of {model} has no chat template and the protocol has no fallback for {model_type!r}")
        template = (EVAL_DIR / fallback["template"]).read_text()
        digest = hashlib.sha256(template.encode()).hexdigest()
        if digest != fallback["sha256"]:
            raise ValueError(f"{fallback['template']} differs from the protocol's template (sha256 {digest})")
        tokenizer.chat_template = template
        output.mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(str(output))
    return output, {
        "fallback_applied": True,
        "template": fallback["template"],
        "template_source": fallback["source"],
        "template_sha256": digest,
    }
