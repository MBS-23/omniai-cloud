"""Multilingual translation with NLLB-200 (distilled, 600M).

The original implementation passed ``local_files_only=True`` on both branches,
including the one that falls back to the Hugging Face model ID. On any machine
where the weights were not already cached this raised, translation was marked
permanently unavailable for the life of the process, and every request
returned the input text with ``translation_applied: False`` and no visible
error. ``download_models.py`` did not fetch NLLB either, so that was the
default experience.

Now: a local directory is used offline when present, otherwise the model is
fetched from the Hub once and cached. Language codes come from the shared
table in :mod:`backend.utils.languages`.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from backend.config import get_settings
from backend.utils.languages import nllb_code
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.translation")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
LOCAL_DIR_NAME = "nllb-200-distilled-600M"
MAX_INPUT_TOKENS = 512

_lock = threading.Lock()
_tokenizer: Any = None
_model: Any = None
_unavailable_reason: str | None = None


def _local_model_dir() -> Path:
    return get_settings().model_dir / LOCAL_DIR_NAME


def _resolve_device() -> str:
    configured = get_settings().torch_device
    if configured in {"cpu", "cuda"}:
        return configured

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load() -> tuple[Any, Any]:
    """Load tokenizer and model once, thread-safely."""
    global _tokenizer, _model, _unavailable_reason

    if _unavailable_reason:
        raise RuntimeError(_unavailable_reason)

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    with _lock:
        if _tokenizer is not None and _model is not None:
            return _tokenizer, _model
        if _unavailable_reason:
            raise RuntimeError(_unavailable_reason)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        local_dir = _local_model_dir()
        if local_dir.exists():
            target, offline = str(local_dir), True
            logger.info("Loading NLLB-200 from %s", local_dir)
        else:
            target, offline = MODEL_NAME, False
            logger.info(
                "NLLB-200 not found locally; downloading %s from Hugging Face "
                "(about 2.4 GB, once). Run `python -m backend.models.download_models "
                "nllb` to pre-fetch it.",
                MODEL_NAME,
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=offline)
            model = AutoModelForSeq2SeqLM.from_pretrained(target, local_files_only=offline)
        except Exception as error:
            _unavailable_reason = (
                "The translation model could not be loaded. Pre-download it with "
                "`python -m backend.models.download_models nllb`, or set "
                "OMNIAI_PROFILE=lite to translate through the hosted API. "
                f"Underlying error: {error}"
            )
            logger.error(_unavailable_reason)
            raise RuntimeError(_unavailable_reason) from error

        device = _resolve_device()
        model = model.to(device).eval()
        logger.info("NLLB-200 ready on %s", device)

        _tokenizer, _model = tokenizer, model
        return _tokenizer, _model


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("﻿", " ")).strip()


def translate_text(
    text: str,
    source_lang: str | None = None,
    target_lang: str = "en",
) -> dict[str, Any]:
    """Translate between any two NLLB-supported languages."""
    cleaned = _normalize_text(text)
    source = (source_lang or "en").lower()
    target = (target_lang or "en").lower()

    try:
        tokenizer, model = _load()
    except RuntimeError as error:
        return {
            "translated_text": cleaned,
            "source_language": source,
            "target_language": target,
            "translation_applied": False,
            "translation_error": str(error),
        }

    import torch

    tokenizer.src_lang = nllb_code(source)
    target_token = nllb_code(target)

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=True,
    ).to(model.device)

    # convert_tokens_to_ids returns the unk id for an unknown language token,
    # which would silently produce garbage rather than failing.
    forced_bos = tokenizer.convert_tokens_to_ids(target_token)
    if forced_bos == tokenizer.unk_token_id:
        return {
            "translated_text": cleaned,
            "source_language": source,
            "target_language": target,
            "translation_applied": False,
            "translation_error": f"'{target}' is not a supported translation target.",
        }

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=MAX_INPUT_TOKENS,
            num_beams=4,
        )

    translated = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    return {
        "translated_text": translated,
        "source_language": source,
        "target_language": target,
        "translation_applied": True,
    }


def translate_to_english(text: str, source_lang: str | None = None) -> dict[str, str]:
    """Convenience wrapper used by the text-analytics pipeline."""
    result = translate_text(text, source_lang=source_lang, target_lang="en")
    return {"translated_text": result["translated_text"]}
