"""Multilingual translation with NLLB-200 (paper section 4B; ref [15]).

``facebook/nllb-200-distilled-600M``, the checkpoint the original code used,
covering every language the interface offers in both directions.

Two things the original did not do:

* It truncated input at 512 tokens, so a long passage lost everything after
  its first few paragraphs without any sign it had. Text is now split at
  sentence boundaries into pieces the model can take whole, translated in
  batches, and reassembled with its line breaks intact.
* The text pipeline translates once for display, and the analytics step asks
  for the English version again. A small cache makes the second request free.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from backend.modules.hub import PAPER_MODELS
from backend.modules.lazy import Lazy
from backend.utils.languages import nllb_code

MAX_INPUT_TOKENS = 512
# Pieces well under the input limit leave the decoder room for scripts that
# take more tokens than the source (English into Telugu roughly doubles).
CHUNK_TOKENS = 200
BATCH_SIZE = 8

_SENTENCE_END = re.compile(r"(?<=[.!?\u0964\u0965\u3002\uff01\uff1f])\s+")


def _build() -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from backend.utils.cpu import quantize_for_cpu

    repo = PAPER_MODELS["translate"].repo_id
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    return tokenizer, quantize_for_cpu(model.eval())


MODEL = Lazy("translate", "NLLB-200 (600M)", _build, resident_mb=1700, peak_mb=2600)


def _normalize(text: str) -> str:
    """Tidy spacing inside lines, but keep the line structure."""
    lines = [re.sub(r"[ \t\u00a0\ufeff]+", " ", line).strip() for line in (text or "").splitlines()]
    return "\n".join(line for line in lines if line)


def _pieces(text: str, tokenizer: Any) -> list[tuple[int, str]]:
    """Split text into (line number, piece) pairs of at most CHUNK_TOKENS."""
    pieces: list[tuple[int, str]] = []
    for line_number, line in enumerate(text.split("\n")):
        current: list[str] = []
        size = 0
        for sentence in _SENTENCE_END.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            length = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
            if current and size + length > CHUNK_TOKENS:
                pieces.append((line_number, " ".join(current)))
                current, size = [], 0
            current.append(sentence)
            size += length
        if current:
            pieces.append((line_number, " ".join(current)))
    return pieces


@lru_cache(maxsize=64)
def _translate_cached(text: str, source: str, target: str) -> str:
    import torch

    tokenizer, model = MODEL.get()
    target_token = nllb_code(target)
    forced_bos = tokenizer.convert_tokens_to_ids(target_token)
    # An unknown language token maps to <unk>, which would produce fluent
    # nonsense rather than an error.
    if forced_bos is None or forced_bos == tokenizer.unk_token_id:
        raise ValueError(f"'{target}' is not a supported translation target.")

    translated: list[str] = []
    # The tokenizer's source language is shared state: two requests in
    # different languages must not interleave between setting it and using it.
    with MODEL.use_lock:
        tokenizer.src_lang = nllb_code(source)
        pieces = _pieces(text, tokenizer)
        for start in range(0, len(pieces), BATCH_SIZE):
            batch = [piece for _, piece in pieces[start:start + BATCH_SIZE]]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS,
            )
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    num_beams=4,
                    max_new_tokens=MAX_INPUT_TOKENS,
                )
            translated.extend(item.strip() for item in tokenizer.batch_decode(tokens, skip_special_tokens=True))

    lines: dict[int, list[str]] = {}
    for (line_number, _), result in zip(pieces, translated, strict=True):
        lines.setdefault(line_number, []).append(result)
    return "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))


def translate_text(text: str, source_lang: str | None = None, target_lang: str = "en") -> dict[str, Any]:
    """Translate between any two NLLB-supported languages."""
    cleaned = _normalize(text)
    source = (source_lang or "en").lower()
    target = (target_lang or "en").lower()

    try:
        translated = _translate_cached(cleaned, source, target) if cleaned else ""
    except ValueError as error:
        return {
            "translated_text": cleaned,
            "source_language": source,
            "target_language": target,
            "translation_applied": False,
            "translation_error": str(error),
        }

    return {
        "translated_text": translated or cleaned,
        "source_language": source,
        "target_language": target,
        "translation_applied": bool(translated),
    }


def translate_to_english(text: str, source_lang: str | None = None) -> dict[str, str]:
    """Convenience wrapper used by the text-analytics step."""
    return {"translated_text": translate_text(text, source_lang=source_lang, target_lang="en")["translated_text"]}
