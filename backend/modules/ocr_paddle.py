"""Text extraction with PaddleOCR (paper section 4A; PP-OCR, ref [10]).

PaddleOCR 3.x running the PP-OCRv5 models, which the thesis cites as its OCR
engine. The model files are fetched from the PaddlePaddle organisation on the
Hugging Face Hub, the same source as every other model in the paper profile.

The original tried seven language packs on every image, one after another.
That is kept as an idea, not as a cost: languages are tried in the configured
order (``OMNIAI_OCR_LANGS``), the search stops as soon as one reads the image
confidently, and it stops immediately when the detector finds no text regions
at all, since a different recogniser cannot read text that is not there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from backend.modules.hub import PADDLE_OCR_OPTIONS
from backend.modules.lazy import Lazy
from backend.utils.cpu import effective_cpus
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.ocr")

# Fetch PP-OCRv5 from the Hugging Face Hub rather than Baidu's servers, and do
# not probe every model source for connectivity on each start.
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Lines the recogniser is less sure of than this are noise, not text.
MIN_LINE_CONFIDENCE = 0.5
# A result this confident ends the language search early.
CONFIDENT = 0.85
# Longest image side the detector sees. PP-OCRv5's default processes a large
# photograph at full resolution, which on two CPU cores takes tens of seconds
# and reads no more text than this does.
DETECTION_SIDE = 1920

# What each language pack reads, for the "Script" row in the interface.
SCRIPT_NAMES = {
    "en": "Latin (English)",
    "ch": "Chinese, English and Japanese",
    "te": "Telugu",
    "hi": "Devanagari (Hindi)",
    "ta": "Tamil",
    "kn": "Kannada",
    "ar": "Arabic",
    "ru": "Cyrillic",
    "japan": "Japanese",
    "korean": "Korean",
    "latin": "Latin",
}


# Scripts written under a continuous headline (the shirorekha of Devanagari).
# PaddleOCR's line-orientation model reads that bar as the bottom of an
# upside-down line and flips every one: measured on a Hindi sample, it turned
# 0.93-0.99 confidence readings into three garbled characters. Orientation
# correction stays on for every other script, as the thesis configured it.
_HEADLINE_SCRIPTS = {"hi", "mr", "ne", "sa", "bh", "mai", "devanagari"}


def engine_options(lang: str) -> dict[str, Any]:
    """PaddleOCR settings for one language pack."""
    options: dict[str, Any] = {
        **PADDLE_OCR_OPTIONS,
        "text_det_limit_type": "max",
        "text_det_limit_side_len": DETECTION_SIDE,
        # PaddleOCR defaults to 8 threads whatever the machine; on a two-CPU
        # container that halves throughput.
        "cpu_threads": effective_cpus(),
    }
    if lang in _HEADLINE_SCRIPTS:
        options["use_textline_orientation"] = False
    return options


def _build_engine(lang: str) -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(lang=lang, **engine_options(lang))


_ENGINES: dict[str, Lazy] = {}


def engine(lang: str) -> Lazy:
    if lang not in _ENGINES:
        _ENGINES[lang] = Lazy(
            f"ocr-{lang}", f"PaddleOCR PP-OCRv5 ({lang})", lambda: _build_engine(lang), resident_mb=450
        )
    return _ENGINES[lang]


@dataclass
class _Reading:
    lang: str
    regions: int = 0
    lines: list[str] = field(default_factory=list)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def score(self) -> float:
        # Characters read, weighted by how sure the recogniser was of each line.
        return sum(len(line) * conf for line, conf in zip(self.lines, self.confidences, strict=True))


def _field(result: Any, key: str) -> Any:
    try:
        return result[key]
    except (KeyError, TypeError, IndexError):
        return None


def _read(lang: str, image_path: str) -> _Reading:
    lazy = engine(lang)
    ocr = lazy.get()
    with lazy.use_lock:
        results = list(ocr.predict(image_path))

    reading = _Reading(lang=lang)
    for result in results:
        polys = _field(result, "dt_polys")
        reading.regions += len(polys) if polys is not None else 0

        texts = _field(result, "rec_texts") or []
        scores = _field(result, "rec_scores")
        scores = list(scores) if scores is not None else [1.0] * len(texts)
        boxes = _field(result, "rec_boxes")
        boxes = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes or [])

        for index, text in enumerate(texts):
            cleaned = str(text or "").strip()
            confidence = float(scores[index]) if index < len(scores) else 0.0
            if not cleaned or confidence < MIN_LINE_CONFIDENCE:
                continue
            reading.lines.append(cleaned)
            reading.confidences.append(confidence)
            if index < len(boxes):
                reading.boxes.append({
                    "text": cleaned,
                    "confidence": round(confidence * 100, 2),
                    "bbox": [round(float(v), 2) for v in list(boxes[index])[:4]],
                })
    return reading


def extract_text_paddle(image_path: str, langs: tuple[str, ...] | list[str] = ("en",)) -> dict[str, Any]:
    """Read the text in an image, trying each language pack in turn.

    Returns the pipeline's OCR shape: ``has_text``, ``text``, ``lines``, plus
    the per-line boxes and the language pack that produced the best reading.
    """
    best: _Reading | None = None
    tried: list[str] = []

    for lang in langs or ("en",):
        reading = _read(lang, image_path)
        tried.append(lang)
        if reading.regions == 0:
            # No text regions at all: another recogniser cannot help.
            break
        if best is None or reading.score > best.score:
            best = reading
        if reading.mean_confidence >= CONFIDENT:
            break

    if best is None or not best.lines:
        return {"has_text": False, "text": "", "lines": [], "language_hint": None, "languages_tried": tried}

    text = "\n".join(best.lines)
    return {
        "has_text": True,
        "text": text,
        "lines": best.lines,
        "word_count": len(text.split()),
        "language_hint": SCRIPT_NAMES.get(best.lang, best.lang),
        "mean_confidence": round(best.mean_confidence * 100, 2),
        "boxes": best.boxes,
        "languages_tried": tried,
    }


def initialize_ocr_engines(langs: tuple[str, ...] | list[str] = ("en",)) -> None:
    """Load the OCR engines up front, as the paper's pre-initialisation does.

    Under a memory budget, stops at the first engine that would not fit
    rather than unloading others to make room at startup.
    """
    from backend.modules.lazy import fits_in_budget

    for lang in langs:
        handle = engine(lang)
        if not handle.loaded and not fits_in_budget(handle):
            break
        handle.get()
