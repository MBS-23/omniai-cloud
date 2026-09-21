"""Orchestration: routing decision -> model execution -> aggregated result.

This is the intelligence layer from section 3.B of the paper, lifted out of
the Flask route handlers so it can be unit-tested and streamed.

The routing rules themselves are unchanged from the original implementation:

* OCR runs before detection, so that a screenshot full of text skips object
  detection entirely.
* An image classified as an animal is re-routed to the high-capacity detector
  and skips OCR, because wildlife photographs carry no readable text.
* Detection labels are then refined against the classifier's species label.

What is new is that each step emits a :class:`ProgressEvent`. The interface
shows the pipeline's real state instead of a timer pretending to be one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from backend.agent.auto_selector import analyze_input, auto_model_selector
from backend.agent.taxonomy import is_animal_image
from backend.core.labels import refine_detections, refinement_summary
from backend.modules.explainability import generate_explanation
from backend.providers.base import Backend, BackendUnavailable
from backend.utils.languages import is_supported
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.pipeline")

Status = Literal["start", "done", "skipped", "error"]

ProgressHook = Callable[["ProgressEvent"], None]

# Enough extracted text to treat an image as a document rather than a scene.
_TEXT_CHAR_THRESHOLD = 50
_TEXT_WORD_THRESHOLD = 10
# Share of the image the text boxes must cover before an image counts as a
# page of text (a scan, a screenshot, a notice) rather than a photograph that
# happens to contain some words, such as a street sign or a bus destination.
_TEXT_COVERAGE_THRESHOLD = 0.10
# Past this many words the image is text whatever its layout.
_MANY_WORDS = 40
# A shorter reading still counts when the recogniser is this sure of it.
_CONFIDENT_OCR = 85.0
_SHORT_TEXT_CHARS = 20

_STAGE_LABELS = {
    "inspect": "Inspecting input",
    "route": "Selecting models",
    "image_classification": "Classifying image",
    "extract_text": "Reading text",
    "object_detection": "Detecting objects",
    "wildlife_detection": "Detecting wildlife",
    "caption_image": "Generating caption",
    "document_ai": "Processing document",
    "translate": "Translating",
    "analyze_text": "Analysing text",
    "explain": "Compiling explanation",
}


@dataclass
class ProgressEvent:
    stage: str
    status: Status
    percent: int
    label: str = ""
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = _STAGE_LABELS.get(self.stage, self.stage.replace("_", " ").title())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "percent": self.percent,
            "label": self.label,
            "detail": self.detail,
        }


@dataclass
class _Run:
    """Mutable state for one analysis, plus progress bookkeeping."""

    backend: Backend
    hook: ProgressHook | None = None
    results: dict[str, Any] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Set by the OCR step: the image is a page of text, not a scene, so scene
    # detection and captioning have nothing to add.
    text_image: bool = False
    _percent: int = 0

    def emit(self, stage: str, status: Status, percent: int | None = None, detail: str | None = None) -> None:
        if percent is not None:
            self._percent = max(self._percent, min(100, percent))
        if self.hook is not None:
            self.hook(ProgressEvent(stage=stage, status=status, percent=self._percent, detail=detail))

    def record(self, module: str) -> None:
        if module not in self.modules:
            self.modules.append(module)

    def warn(self, message: str) -> None:
        from backend.utils.redaction import redact

        # Warnings are surfaced in the interface and in the downloaded JSON.
        message = redact(message)
        logger.warning(message)
        if message not in self.warnings:
            self.warnings.append(message)


def _count(number: int, noun: str) -> str:
    """ "1 model", "4 models"."""
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _word_count(text: str) -> int:
    return len([word for word in (text or "").split() if word])


def _has_substantial_text(ocr_result: dict[str, Any]) -> bool:
    """Real text, as opposed to a few characters of recogniser noise.

    Ten words is a fair floor for English, but a Telugu or Hindi sentence
    often has fewer, longer words. A reading the recogniser is confident of
    is kept even when short, provided it is more than a fragment.
    """
    text = (ocr_result.get("text") or "").strip()
    if len(text) >= _TEXT_CHAR_THRESHOLD and _word_count(text) >= _TEXT_WORD_THRESHOLD:
        return True
    confidence = ocr_result.get("mean_confidence")
    return confidence is not None and confidence >= _CONFIDENT_OCR and len(text) >= _SHORT_TEXT_CHARS


def _text_coverage(ocr_result: dict[str, Any], path: str) -> float | None:
    """Fraction of the image covered by text boxes, or None without boxes."""
    boxes = [box.get("bbox") for box in ocr_result.get("boxes") or [] if box.get("bbox")]
    if not boxes:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            area = image.width * image.height
    except Exception:
        return None
    covered = sum(max(0.0, x2 - x1) * max(0.0, y2 - y1) for x1, y1, x2, y2 in (b[:4] for b in boxes))
    return min(1.0, covered / area) if area else None


def _is_text_image(ocr_result: dict[str, Any], path: str) -> bool:
    """A page of text, as opposed to a photograph with some words in it.

    Word count alone cannot tell the two apart: a street photograph can carry
    ten words of signage. Where the backend reports where the text is, the
    share of the image it covers decides; without boxes, the word count does.
    """
    words = _word_count(ocr_result.get("text", ""))
    coverage = _text_coverage(ocr_result, path)
    ocr_result["text_coverage"] = None if coverage is None else round(coverage * 100, 1)
    if coverage is None:
        return True
    return coverage >= _TEXT_COVERAGE_THRESHOLD or words >= _MANY_WORDS


# ======================================================================
# File pipeline
# ======================================================================
def run_file_analysis(
    file_path: Path | str,
    backend: Backend,
    *,
    on_event: ProgressHook | None = None,
) -> dict[str, Any]:
    """Analyse an image or document and return the aggregated result."""
    run = _Run(backend=backend, hook=on_event)
    path = str(file_path)

    run.emit("inspect", "start", 5)
    metadata = analyze_input(file_path=path)
    if metadata["input_type"] == "unknown":
        raise ValueError("Unsupported input type")

    if metadata["input_type"] == "document":
        _validate_document(path, metadata.get("file_extension"))
    run.emit("inspect", "done", 10, detail=metadata["input_type"])

    run.emit("route", "start", 12)
    decision = auto_model_selector(metadata)
    planned = decision["models"]
    run.reasons = decision["reasoning"]
    run.emit("route", "done", 18, detail=_count(len(planned), "model") + " selected")

    # Progress is distributed evenly across the planned models, leaving the
    # last 12% for explanation assembly.
    span = max(1, len(planned))
    step = (88 - 18) // span

    for index, model in enumerate(planned):
        floor = 18 + index * step
        if model == "image_classification":
            _step_classify(run, path, floor)
        elif model == "ocr_paddle":
            _step_ocr(run, path, floor)
        elif model == "object_detection":
            _step_detect(run, path, floor)
        elif model == "blip_vision_language":
            _step_caption(run, path, floor)
        elif model == "document_ai":
            _step_document(run, path, metadata, floor)

    # OCR may have revealed the language of text inside an image.
    ocr_result = run.results.get("ocr")
    if ocr_result and ocr_result.get("detected_language"):
        metadata["language"] = ocr_result["detected_language"]

    run.emit("explain", "start", 90)
    explanation = generate_explanation(
        input_type=metadata["input_type"],
        selected_models=run.modules,
        results=run.results,
        planned_models=planned,
        warnings=run.warnings,
        routing_reasons=run.reasons,
    )
    run.emit("explain", "done", 100)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "selected_modules": run.modules,
        "planned_modules": planned,
        "results": run.results,
        "explanation": explanation["explanation"],
        "backend": backend.name,
    }
    if run.warnings:
        payload["warnings"] = run.warnings
    return payload


def _validate_document(path: str, extension: str | None) -> None:
    from backend.utils.docx_utils import validate_docx_file
    from backend.utils.pdf_utils import validate_pdf_file

    if extension == ".pdf":
        validate_pdf_file(path)
    elif extension == ".docx":
        validate_docx_file(path)
    else:
        raise ValueError("Unsupported document type. Only PDF and DOCX are allowed")


def _step_classify(run: _Run, path: str, floor: int) -> None:
    run.emit("image_classification", "start", floor)
    try:
        result = run.backend.classify_image(path)
    except BackendUnavailable as error:
        run.warn(f"Image classification unavailable: {error}")
        run.emit("image_classification", "error", detail=str(error))
        return
    run.results["image_classification"] = result
    run.record("image_classification")
    run.emit(
        "image_classification",
        "done",
        floor + 6,
        detail=f"{result.get('label')} ({result.get('confidence')}%)",
    )


def _step_ocr(run: _Run, path: str, floor: int) -> None:
    classification = run.results.get("image_classification") or {}
    label = classification.get("label", "")

    # Wildlife photographs carry no readable text; skip the OCR cost.
    if is_animal_image(label):
        run.emit("extract_text", "skipped", floor + 6, detail=f"not expected in a {label} photo")
        return

    run.emit("extract_text", "start", floor)
    try:
        ocr_result = run.backend.extract_text(path)
    except BackendUnavailable as error:
        run.warn(f"Text extraction unavailable: {error}")
        run.emit("extract_text", "error", detail=str(error))
        return

    if not ocr_result.get("has_text"):
        run.emit("extract_text", "skipped", floor + 6, detail="no readable text found")
        return

    if not _has_substantial_text(ocr_result):
        words = _word_count(ocr_result.get("text", ""))
        run.emit("extract_text", "skipped", floor + 6, detail=f"only {words} words, below threshold")
        return

    text = (ocr_result.get("text") or "").strip()
    ocr_result["word_count"] = _word_count(text)

    try:
        from langdetect import detect

        ocr_result["detected_language"] = detect(text)
    except Exception:
        pass

    run.text_image = _is_text_image(ocr_result, path)
    coverage = ocr_result.get("text_coverage")
    if coverage is not None and not run.text_image:
        run.reasons.append(
            f"Text covers only {coverage}% of the image, so it was treated as a photograph "
            "with some words in it: objects and a caption were still produced."
        )

    run.results["ocr"] = ocr_result
    run.record("ocr_paddle")
    run.emit("extract_text", "done", floor + 6, detail=f"{ocr_result['word_count']} words")


def _step_detect(run: _Run, path: str, floor: int) -> None:
    # An image that turned out to be a page of text needs no object detection.
    if run.text_image:
        run.emit("object_detection", "skipped", floor + 6, detail="input is text, not a scene")
        return

    classification = run.results.get("image_classification") or {}
    label = classification.get("label", "")
    confidence = classification.get("confidence", 0.0)
    wildlife = is_animal_image(label)
    stage = "wildlife_detection" if wildlife else "object_detection"

    run.emit(stage, "start", floor)
    try:
        raw = run.backend.detect_objects(path, wildlife=wildlife)
    except BackendUnavailable as error:
        run.warn(f"Object detection unavailable: {error}")
        run.emit(stage, "error", detail=str(error))
        return

    # The pipeline does not own what a backend hands back. Copying before
    # refinement keeps a backend that caches or reuses a response object from
    # being corrupted, and stops one request's refinement leaking into the
    # next -- which is exactly what happens without it.
    result = dict(raw)

    if wildlife and result.get("detections"):
        result["detections"] = refine_detections(result["detections"], label, confidence)
        result["refinement"] = refinement_summary(result["detections"])

    run.results["object_detection"] = result
    run.record("wildlife_detection" if wildlife else "object_detection")
    run.emit(stage, "done", floor + 6, detail=_count(result.get("object_count", 0), "object"))


def _step_caption(run: _Run, path: str, floor: int) -> None:
    # An image that turned out to be a page of text does not need describing:
    # OCR already has its content, and a caption model given a document
    # produces a garbled paraphrase of the words it can partly read. This is
    # the same rule that suppresses object detection for text images.
    if run.text_image:
        run.emit("caption_image", "skipped", floor + 6, detail="text already extracted")
        return

    classification = run.results.get("image_classification") or {}
    label = classification.get("label", "")
    prompt = f"a photo of {label}" if is_animal_image(label) else None

    run.emit("caption_image", "start", floor)
    try:
        caption_result = run.backend.caption_image(path, prompt=prompt)
    except BackendUnavailable as error:
        run.warn(f"Captioning unavailable: {error}")
        run.emit("caption_image", "error", detail=str(error))
        return

    detections = (run.results.get("object_detection") or {}).get("detections", [])
    if is_animal_image(label) and detections:
        caption_result = _enrich_caption(caption_result, detections)

    run.results["caption"] = caption_result
    run.record("blip_vision_language")
    run.emit("caption_image", "done", floor + 6)


def _enrich_caption(caption_result: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Append detected species context to a caption, preserving the original."""
    original = caption_result.get("caption", "")
    species = list(dict.fromkeys(d["label"] for d in detections if d.get("label")))
    if not species:
        return caption_result

    if len(species) == 1:
        scene = f"{species[0]} in action"
    else:
        scene = ", ".join(species[:-1]) + f" and {species[-1]} interaction"

    caption_result["original_caption"] = original
    caption_result["enhanced_caption"] = f"{original} [Scene: {scene}]"
    caption_result["caption"] = caption_result["enhanced_caption"]
    return caption_result


def _step_document(run: _Run, path: str, metadata: dict[str, Any], floor: int) -> None:
    from backend.modules.document_ai import process_document

    run.emit("document_ai", "start", floor)
    try:
        doc_result = process_document(path, backend=run.backend)
    except BackendUnavailable as error:
        run.warn(f"Document processing unavailable: {error}")
        run.emit("document_ai", "error", detail=str(error))
        return

    if doc_result.get("language"):
        metadata["language"] = doc_result["language"]

    ocr_was_used = bool(doc_result.get("ocr_used"))
    summary = doc_result.get("summary", "")

    run.results["document"] = {
        "type": doc_result["document_type"],
        "extraction_method": doc_result["extraction_method"],
        # Raw text is returned only for scanned sources, where the reader may
        # need to check the OCR. For text PDFs the summary is the product.
        "extracted_text": doc_result.get("extracted_text", "") if ocr_was_used else "",
        "summary": summary,
        "summary_generated": bool(doc_result.get("summary_generated")),
        "summary_model": doc_result.get("summary_model"),
        "coverage_note": doc_result.get("coverage_note"),
        "no_text_reason": doc_result.get("no_text_reason"),
        "text_preview": summary[:1000],
        "insights": doc_result.get("insights", []),
        "page_count": doc_result.get("page_count"),
    }
    run.record("document_ai")
    if ocr_was_used:
        run.record("ocr_paddle")

    run.emit("document_ai", "done", floor + 6, detail=doc_result["document_type"])


# ======================================================================
# Text pipeline
# ======================================================================
def run_text_analysis(
    text: str,
    backend: Backend,
    *,
    source_language: str = "auto",
    target_language: str = "en",
    auto_detect_source: bool = True,
    on_event: ProgressHook | None = None,
) -> dict[str, Any]:
    """Analyse raw text: detect language, optionally translate, then analyse."""
    run = _Run(backend=backend, hook=on_event)

    run.emit("inspect", "start", 5)
    metadata = analyze_input(text=text)
    detected = metadata.get("language") or "unknown"
    run.emit("inspect", "done", 15, detail=f"language: {detected}")

    run.emit("route", "start", 18)
    decision = auto_model_selector(metadata)
    planned = decision["models"]
    run.reasons = decision["reasoning"]
    run.emit("route", "done", 25, detail=_count(len(planned), "model") + " selected")

    source = (source_language or "auto").strip().lower()
    target = (target_language or "en").strip().lower()
    effective_source = detected if (source == "auto" or auto_detect_source) else source

    should_translate = (
        bool(target)
        and effective_source not in {"", "unknown"}
        and effective_source != target
    )

    if should_translate and not is_supported(target):
        run.warn(f"'{target}' is not a supported translation target; skipping translation.")
        should_translate = False

    if should_translate:
        run.emit("translate", "start", 30)
        try:
            translation = backend.translate(
                text, source_lang=effective_source, target_lang=target
            )
        except BackendUnavailable as error:
            run.warn(f"Translation unavailable: {error}")
            run.emit("translate", "error", detail=str(error))
        else:
            if translation.get("translation_applied"):
                run.results["translation"] = translation
                run.record("translation_ai")
                run.emit("translate", "done", 60, detail=f"{effective_source} to {target}")
            else:
                # The backend handed back the input unchanged. Report that
                # rather than presenting untranslated text under a panel
                # labelled "Translation".
                reason = (
                    translation.get("translation_error")
                    or "the translation model was unavailable"
                )
                run.warn(f"Translation did not run: {reason}")
                run.emit("translate", "error", 60, detail=reason)
    else:
        run.emit("translate", "skipped", 60, detail="source and target match")

    metadata["source_language"] = effective_source
    metadata["target_language"] = target

    if "text_analytics" in decision["models"]:
        run.emit("analyze_text", "start", 65)
        try:
            analytics = backend.analyze_text(text)
        except BackendUnavailable as error:
            run.warn(f"Text analytics unavailable: {error}")
            run.emit("analyze_text", "error", detail=str(error))
        else:
            run.results["text_analytics"] = analytics
            run.record("text_analytics")
            run.emit(
                "analyze_text",
                "done",
                90,
                detail=f"{analytics.get('sentiment', 'unknown').lower()} sentiment",
            )

    run.emit("explain", "start", 92)
    explanation = generate_explanation(
        input_type="text",
        selected_models=run.modules,
        results=run.results,
        planned_models=planned,
        warnings=run.warnings,
        routing_reasons=run.reasons,
    )
    run.emit("explain", "done", 100)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "selected_modules": run.modules,
        "planned_modules": planned,
        "results": run.results,
        "explanation": explanation["explanation"],
        "backend": backend.name,
    }
    if run.warnings:
        payload["warnings"] = run.warnings
    return payload
