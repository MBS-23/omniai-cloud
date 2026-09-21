"""Explainability: a truthful account of what the system did.

The previous version only saw the models that *succeeded*. When every model
failed it reported "No AI models were executed because the input did not match
any supported processing pipeline", which is simply untrue: the input matched,
the pipeline ran, and the models failed for an unrelated reason. Telling a user
their file was unsupported when the real cause was an expired API key is worse
than saying nothing.

It now receives what was planned, what ran, and what went wrong, and says so.
"""

from __future__ import annotations

import re
from typing import Any

# Model identifier -> how to describe it to a person.
_MODEL_NAMES = {
    "image_classification": "Image classification",
    "object_detection": "Object detection",
    "wildlife_detection": "Wildlife detection",
    "ocr_paddle": "Text extraction (OCR)",
    "blip_vision_language": "Vision-language captioning",
    "document_ai": "Document analysis",
    "translation_ai": "Translation",
    "text_analytics": "Text analytics",
}

_INPUT_NAMES = {
    "image": "an image",
    "document": "a document",
    "text": "a block of text",
    "unknown": "an unrecognised input",
}


def _name(model: str) -> str:
    return _MODEL_NAMES.get(model, model.replace("_", " ").capitalize())


def _with(result: dict[str, Any] | None) -> str:
    """ " with YOLOv8x", naming the model a backend reports, or nothing.

    Backends report models as "YOLOv8x (Ultralytics/YOLOv8)"; the repository
    in brackets is detail for the results panel, not for a sentence.
    """
    model = str((result or {}).get("model") or "")
    model = re.sub(r"\s*\([^()]*/[^()]*\)", "", model).split(", ")[0].strip()
    return f" with {model}" if model else ""


def _list(items: list[str]) -> str:
    """Join names into readable prose: 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def generate_explanation(
    input_type: str,
    selected_models: list[str],
    results: dict[str, Any],
    *,
    planned_models: list[str] | None = None,
    warnings: list[str] | None = None,
    routing_reasons: list[str] | None = None,
) -> dict[str, str]:
    """Describe the run in plain language.

    Args:
        input_type: What the input was detected as.
        selected_models: Models that actually produced output.
        results: The aggregated results, used to quote specific findings.
        planned_models: What the agent chose to run, before any failures.
        warnings: Human-readable reasons a planned model produced nothing.
        routing_reasons: The agent's stated justification for its choices.
    """
    planned = planned_models or []
    warnings = warnings or []
    lines: list[str] = []

    # ---- 1. What arrived -------------------------------------------------
    lines.append(f"The system received {_INPUT_NAMES.get(input_type, input_type)}.")

    # ---- 2. What was chosen, and why ------------------------------------
    if planned:
        lines.append(
            f"The routing agent selected {len(planned)} model"
            f"{'s' if len(planned) != 1 else ''}: {_list([_name(m) for m in planned])}."
        )
        for reason in (routing_reasons or [])[:6]:
            lines.append(f"- {reason}")
    elif input_type == "unknown":
        lines.append(
            "No pipeline matched this input, so no models were run. "
            "Supported inputs are JPG, PNG, WebP, PDF, DOCX and plain text."
        )
        return {"explanation": "\n".join(lines)}

    lines.append("")

    # ---- 3. Failures, stated plainly ------------------------------------
    # This is the case the old version got wrong.
    if warnings:
        if not selected_models:
            lines.append(
                "None of the selected models returned a result. The pipeline ran "
                "correctly; the models themselves could not be reached:"
            )
        else:
            lines.append("Some selected models did not return a result:")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if not selected_models:
        lines.append(
            "Because of this, there is no analysis to report. This is a "
            "configuration or connectivity problem, not a problem with your input."
        )
        return {"explanation": "\n".join(lines)}

    # ---- 4. What each model contributed ----------------------------------
    lines.append("What the models that ran found:")
    lines.append("")

    classification = results.get("image_classification")
    if classification:
        lines.append(
            f"- Image classification{_with(classification)} identified the subject as "
            f"'{classification.get('label', 'unknown')}' with "
            f"{classification.get('confidence', 0):.2f}% confidence."
        )

    detection = results.get("object_detection")
    if detection:
        count = detection.get("object_count", len(detection.get("detections", [])))
        line = f"- Object detection{_with(detection)} located {count} object(s)."
        refinement = detection.get("refinement") or {}
        if refinement.get("applied"):
            changes = refinement.get("changes", [])
            if changes:
                change = changes[0]
                line += (
                    f" The label '{change.get('from')}' was refined to "
                    f"'{change.get('to')}' using the classifier's species result, "
                    "because the detector's training set has no class for it."
                )
        if detection.get("bbox_available") is False:
            line += " This backend does not return bounding boxes, so no overlay is shown."
        lines.append(line)

    ocr = results.get("ocr")
    if ocr:
        lines.append(
            f"- Text extraction{_with(ocr)} read {ocr.get('word_count', 0)} words "
            f"({len(ocr.get('text', ''))} characters) from the image."
        )

    if results.get("caption"):
        lines.append(
            f"- A vision-language model{_with(results['caption'])} described the scene in natural language."
        )

    document = results.get("document")
    if document:
        kind = document.get("type", "a document")
        reason = document.get("no_text_reason")

        if document.get("summary_generated"):
            lines.append(
                f"- Document analysis identified the file as {kind} and read it via "
                f"{document.get('extraction_method', 'automatic extraction')}."
            )
            summariser = _with({"model": document.get("summary_model")})
            lines.append(f"- A summary was generated from the extracted text{summariser.replace(' with', ' by', 1)}.")
        elif reason:
            # Extraction failed, so naming the method that did not work would
            # only repeat the reason back.
            lines.append(f"- Document analysis identified the file as {kind}, but {reason[0].lower()}{reason[1:]}")
        else:
            lines.append(
                f"- Document analysis identified the file as {kind} and read it via "
                f"{document.get('extraction_method', 'automatic extraction')}."
            )

    translation = results.get("translation")
    if translation:
        lines.append(
            f"- Translation{_with(translation)} converted the text from "
            f"{translation.get('source_language', 'the detected language')} to "
            f"{translation.get('target_language', 'the target language')}."
        )

    analytics = results.get("text_analytics")
    if analytics:
        entity_count = len(analytics.get("entities", []))
        lines.append(
            f"- Text analytics{_with(analytics)} found {analytics.get('sentiment', 'unknown').lower()} "
            f"sentiment ({analytics.get('sentiment_confidence', 0):.2f}% confidence) "
            f"and extracted {entity_count} named entit"
            f"{'y' if entity_count == 1 else 'ies'}."
        )

    # ---- 5. Models that were skipped by design ---------------------------
    skipped = [m for m in planned if m not in selected_models]
    # Detection is recorded under a different name when wildlife routing fires.
    if "object_detection" in skipped and "wildlife_detection" in selected_models:
        skipped.remove("object_detection")
    if skipped and not warnings:
        lines.append("")
        lines.append(
            f"{_list([_name(m) for m in skipped])} "
            f"{'was' if len(skipped) == 1 else 'were'} skipped: the pipeline "
            "determined they would add nothing for this input."
        )

    lines.append("")
    lines.append(
        "Every step above is recorded as it happens, so the result can be checked "
        "rather than taken on trust."
    )

    return {"explanation": "\n".join(lines)}
