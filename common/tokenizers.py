"""Tokenizer preparation shared by model and data loaders."""
from __future__ import annotations


def ensure_pad_token(tokenizer, *, add_if_missing: bool = False) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    fallback = tokenizer.eos_token
    if not add_if_missing:
        fallback = fallback or tokenizer.unk_token
    if fallback is not None:
        tokenizer.pad_token = fallback
    elif add_if_missing:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        tokenizer.pad_token = fallback
