"""Hosted backend: every capability served by the Mistral API.

This is the ``lite`` profile. It carries no PyTorch dependency, starts in
under a second and is small enough to containerise and deploy. Where a hosted
model genuinely cannot match the local one -- object localisation is the only
such case -- the result says so explicitly rather than inventing coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.providers.base import (
    Backend,
    BackendUnavailable,
    CaptionResult,
    Classification,
    DetectionResult,
    OcrResult,
    TextAnalytics,
    TranslationResult,
)
from backend.providers.mistral_client import MistralClient
from backend.utils.languages import language_name
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.remote")

_SENTIMENT_LABELS = {"SAD", "NEUTRAL", "POSITIVE"}

_CLASSIFY_SYSTEM = (
    "You are an image classifier. Reply with a JSON object only, shaped "
    '{"label": string, "confidence": number}. The label must be the single most '
    "specific common name for the main subject, lowercase, in the style of an "
    "ImageNet class (for example 'african elephant', not 'animal'). confidence "
    "is your certainty from 0 to 100."
)

_DETECT_SYSTEM = (
    "You detect objects in images. Reply with a JSON object only, shaped "
    '{"objects": [{"label": string, "confidence": number}]}. Use lowercase '
    "singular labels and confidence from 0 to 100. List at most 15 objects, "
    "most prominent first. Return an empty list if there are no discrete objects."
)

_CAPTION_SYSTEM = (
    "You write image captions. Reply with one factual sentence of 15 to 30 words "
    "describing what is visibly present. No preamble, no speculation about mood "
    "or intent, no markdown."
)

_SUMMARY_SYSTEM = (
    "You are an expert summariser. Write one paragraph that captures the key "
    "points, main ideas and important details. Be thorough but do not repeat "
    "yourself. Plain prose only: no markdown, no heading, no 'Summary:' prefix."
)

_ADJUDICATE_SYSTEM = (
    "You settle disagreements between image classifiers. Study the image "
    "carefully, then reply with a JSON object only, shaped "
    '{"label": string, "confidence": number, "reason": string}. '
    "The label is the single most specific correct common name for the main "
    "subject, lowercase. confidence is 0 to 100. reason is one short sentence "
    "naming the visual evidence you used."
)

_ANALYTICS_SYSTEM = (
    "You analyse text. Reply with a JSON object only, shaped "
    '{"sentiment": "POSITIVE"|"NEUTRAL"|"SAD", "confidence": number, '
    '"entities": [{"text": string, "label": string}]}. Use SAD for negative or '
    "sorrowful sentiment. confidence is 0 to 100. Entity labels follow spaCy "
    "conventions: PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT, PRODUCT, "
    "EVENT, WORK_OF_ART, LANGUAGE, NORP, FAC, CARDINAL, ORDINAL, QUANTITY."
)


def _clamp_confidence(value: Any, default: float = 0.0) -> float:
    """Normalise a model-reported confidence to a 0-100 percentage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    # Accept either a 0-1 probability or a 0-100 percentage.
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return round(max(0.0, min(100.0, number)), 2)


def _detect_language(text: str) -> str:
    from langdetect import detect

    try:
        return detect(text)
    except Exception:
        # langdetect raises LangDetectException for short or scriptless input.
        return "unknown"


class RemoteBackend(Backend):
    name = "mistral"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = MistralClient(settings.mistral)

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        if not self._settings.mistral.configured:
            logger.warning(
                "Lite profile selected but no Mistral API key is set. "
                "Requests will fail until OMNIAI_MISTRAL_API_KEY is provided."
            )
            return
        logger.info(
            "Lite profile ready (text=%s, vision=%s, ocr=%s)",
            self._settings.mistral.text_model,
            self._settings.mistral.vision_model,
            self._settings.mistral.ocr_model,
        )

    def capabilities(self) -> set[str]:
        if not self._settings.mistral.configured:
            return set()
        return {
            "classify_image",
            "detect_objects",
            "extract_text",
            "caption_image",
            "summarize",
            "translate",
            "analyze_text",
        }

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def _classify_once(self, image_path: str, model: str) -> Classification:
        data = self._client.chat_json(
            system=_CLASSIFY_SYSTEM,
            user="Classify the main subject of this image.",
            image_path=image_path,
            model=model,
            max_tokens=120,
        )
        label = str(data.get("label") or "unknown").strip().lower()
        return {
            "label": label or "unknown",
            "confidence": _clamp_confidence(data.get("confidence")),
        }

    def classify_image(self, image_path: str) -> Classification:
        """Classify with two models and an adjudicator; see providers.ensemble."""
        from backend.providers.ensemble import run_ensemble
        from backend.providers.mistral_client import rate_limited

        primary_model = self._settings.mistral.vision_model

        # A second opinion costs a second call. While the key is being rate
        # limited, one good answer beats three failed ones.
        if not self._settings.mistral.ensemble or rate_limited():
            result = self._classify_once(image_path, primary_model)
            result["ensemble"] = {"agreement": "single", "models": [primary_model]}
            return result

        def adjudicate(first: Classification, second: Classification) -> dict[str, Any]:
            verdict = self._client.chat_json(
                system=_ADJUDICATE_SYSTEM,
                user=(
                    "Two classifiers disagree about the main subject of this image.\n"
                    f"Classifier A says: {first['label']}\n"
                    f"Classifier B says: {second['label']}\n"
                    "Look at the image and decide. You may return a third label "
                    "if both are wrong."
                ),
                image_path=image_path,
                model=self._settings.mistral.reasoning_model,
                max_tokens=400,
            )
            verdict["confidence"] = _clamp_confidence(verdict.get("confidence"), first["confidence"])
            return verdict

        return run_ensemble(
            classify_once=lambda model: self._classify_once(image_path, model),
            adjudicate=adjudicate,
            primary_model=primary_model,
            second_model=self._settings.mistral.fast_model,
            adjudicator_model=self._settings.mistral.reasoning_model,
            timeout_seconds=self._settings.mistral.timeout_seconds + 5,
        )

    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult:
        focus = (
            "Focus on animals; name each to species level where you can."
            if wildlife
            else "Include every distinct physical object you can identify."
        )
        data = self._client.chat_json(
            system=_DETECT_SYSTEM,
            user=f"Identify the objects in this image. {focus}",
            image_path=image_path,
            max_tokens=600,
        )

        detections: list[dict[str, Any]] = []
        raw_objects = data.get("objects")
        if isinstance(raw_objects, list):
            for item in raw_objects[:15]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip().lower()
                if not label:
                    continue
                detections.append(
                    {
                        "label": label,
                        "confidence": _clamp_confidence(item.get("confidence")),
                        "bbox": None,
                    }
                )

        return {
            "object_count": len(detections),
            "detections": detections,
            # The hosted model names objects reliably but does not return
            # trustworthy coordinates, so the UI must not draw boxes.
            "bbox_available": False,
        }

    def extract_text(self, image_path: str) -> OcrResult:
        is_pdf = Path(image_path).suffix.lower() == ".pdf"
        response = self._client.ocr(image_path, is_pdf=is_pdf)

        pages = response.get("pages") or []
        blocks = [str(page.get("markdown") or "").strip() for page in pages]
        text = "\n\n".join(block for block in blocks if block).strip()

        if not text:
            return {"has_text": False, "text": "", "lines": [], "language_hint": None}

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return {
            "has_text": True,
            "text": text,
            "lines": lines,
            "word_count": len([word for word in text.split() if word]),
            "language_hint": "auto",
        }

    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult:
        instruction = (
            f"Describe this image. Context hint: {prompt}."
            if prompt
            else "Describe this image."
        )
        caption = self._client.chat(
            system=_CAPTION_SYSTEM,
            user=instruction,
            image_path=image_path,
            max_tokens=120,
        )
        caption = caption.strip().strip('"')
        if caption:
            caption = caption[0].upper() + caption[1:]
        return {"caption": caption or "No caption could be generated for this image."}

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    def summarize(self, text: str, *, max_length: int = 200) -> str:
        return self._client.chat(
            system=_SUMMARY_SYSTEM,
            user=f"Summarise the following text.\n\n{text[:12000]}",
            max_tokens=max(150, min(max_length, 500)),
            temperature=0.3,
        ).strip()

    def translate(self, text: str, *, source_lang: str, target_lang: str) -> TranslationResult:
        source_name = language_name(source_lang)
        target_name = language_name(target_lang)
        system = (
            f"You are a professional translator. Translate the user's text from "
            f"{source_name} into {target_name}. Preserve meaning, tone, names, "
            "numbers and line breaks. Output only the translation."
        )
        try:
            translated = self._client.chat(
                system=system,
                user=text,
                max_tokens=1500,
                temperature=0.1,
            ).strip()
        except BackendUnavailable as error:
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "translation_applied": False,
                "translation_error": str(error),
            }

        return {
            "translated_text": translated or text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation_applied": bool(translated),
        }

    def analyze_text(self, text: str) -> TextAnalytics:
        language = _detect_language(text)

        data = self._client.chat_json(
            system=_ANALYTICS_SYSTEM,
            user=f"Analyse the sentiment and named entities of this text:\n\n{text[:6000]}",
            max_tokens=900,
        )

        sentiment = str(data.get("sentiment") or "NEUTRAL").strip().upper()
        if sentiment not in _SENTIMENT_LABELS:
            sentiment = "NEUTRAL"

        entities: list[dict[str, str]] = []
        raw_entities = data.get("entities")
        if isinstance(raw_entities, list):
            for item in raw_entities[:60]:
                if not isinstance(item, dict):
                    continue
                entity_text = str(item.get("text") or "").strip()
                entity_label = str(item.get("label") or "").strip().upper()
                if entity_text and entity_label:
                    entities.append({"text": entity_text, "label": entity_label})

        return {
            "original_language": language,
            "translated_text": None,
            "sentiment": sentiment,
            "sentiment_confidence": _clamp_confidence(data.get("confidence"), 50.0),
            "entities": entities,
            "explanation": (
                f"Detected input language as '{language}'. "
                f"Sentiment and {len(entities)} named entities were extracted by "
                f"{self._settings.mistral.text_model} in a single structured call."
            ),
        }
