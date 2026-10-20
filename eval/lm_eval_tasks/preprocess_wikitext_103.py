from __future__ import annotations

import re
from typing import Any


def wikitext_detokenizer(doc: dict[str, Any]) -> str:
    text = doc["page"]
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @,@ ", ",")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" : ", ": ")
    text = text.replace(" ; ", "; ")
    text = text.replace(" . ", ". ")
    text = text.replace(" ! ", "! ")
    text = text.replace(" ? ", "? ")
    text = text.replace(" , ", ", ")
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r'"\s*([^\"]*?)\s*"', r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    text = text.replace("= = = =", "====")
    text = text.replace("= = =", "===")
    text = text.replace("= =", "==")
    text = text.replace(" " + chr(176) + " ", chr(176))
    text = text.replace(" \n", "\n")
    text = text.replace("\n ", "\n")
    text = text.replace(" N ", " 1 ")
    return text.replace(" 's", "'s")


def process_results(
    doc: dict[str, Any], results: tuple[float]
) -> dict[str, tuple[float, int]]:
    (loglikelihood,) = results
    words = len(re.split(r"\s+", doc["page"]))
    byte_count = len(doc["page"].encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, words),
        "byte_perplexity": (loglikelihood, byte_count),
        "bits_per_byte": (loglikelihood, byte_count),
    }
