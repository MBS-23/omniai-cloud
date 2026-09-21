"""The auto-selector agent.

Given an input, decide what it is and which models should run. This is the
component the paper argues for in section 3.B: the alternative is executing
every model on every request, which the ablation measures at 12.4-15 s per
request instead of 1.8-2.5 s.

The species taxonomy this used to carry now lives in
:mod:`backend.agent.taxonomy`; it is re-exported here so existing imports keep
working.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from backend.agent.taxonomy import (  # re-exported for backwards compatibility
    ANIMAL_CLASSES,
    SPECIES_COUNT,
    is_animal_image,
)

__all__ = [
    "ANIMAL_CLASSES",
    "SPECIES_COUNT",
    "is_animal_image",
    "analyze_input",
    "auto_model_selector",
    "IMAGE_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
]

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx"})


class InputMetadata(TypedDict, total=False):
    input_type: str
    file_extension: str | None
    language: str | None
    source_language: str
    target_language: str


class RoutingDecision(TypedDict):
    models: list[str]
    reasoning: list[str]


def analyze_input(
    file_path: str | None = None,
    text: str | None = None,
) -> InputMetadata:
    """Classify the input as text, image, document or unknown."""
    if text:
        try:
            from langdetect import detect

            language = detect(text)
        except Exception:
            # langdetect raises for input too short or with no letters.
            language = "unknown"

        return {"input_type": "text", "file_extension": None, "language": language}

    if file_path:
        extension = os.path.splitext(file_path)[1].lower()

        if extension in IMAGE_EXTENSIONS:
            return {"input_type": "image", "file_extension": extension, "language": None}

        if extension in DOCUMENT_EXTENSIONS:
            return {"input_type": "document", "file_extension": extension, "language": None}

    return {"input_type": "unknown", "file_extension": None, "language": None}


def auto_model_selector(metadata: dict[str, Any]) -> RoutingDecision:
    """Choose the model sequence for an input.

    Order matters for images. OCR is listed before object detection so that a
    screenshot or scanned page can short-circuit detection entirely -- the
    pipeline reads the OCR result when it reaches the detection step.
    """
    models: list[str] = []
    reasoning: list[str] = []

    input_type = metadata.get("input_type")
    language = metadata.get("language")

    if input_type == "text":
        if language and language not in {"en", "unknown"}:
            models.append("translation_ai")
            reasoning.append(f"Non-English text ({language}) detected, translation required")

        models.append("text_analytics")
        reasoning.append("Text analytics for sentiment and entity extraction")

    elif input_type == "image":
        models.append("image_classification")
        reasoning.append("Classifying overall image content")

        models.append("ocr_paddle")
        reasoning.append(
            "OCR runs first so that text-bearing images (screenshots, posters, "
            "scanned pages) can skip scene analysis"
        )

        models.append("object_detection")
        reasoning.append(
            "Object detection, routed to the high-capacity model when the "
            "classifier reports a species"
        )

        models.append("blip_vision_language")
        reasoning.append("Vision-language model for a semantic description")

    elif input_type == "document":
        models.append("document_ai")
        reasoning.append(
            "Document detected: structure-aware extraction, OCR only if scanned, "
            "then summarisation"
        )

    else:
        reasoning.append("Unsupported input type")

    return {"models": models, "reasoning": reasoning}
