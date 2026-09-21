"""Hosted backend served by the Gemini API.

Used by the ``lite`` profile when ``OMNIAI_HOSTED_PROVIDER=gemini``. This is
the profile the public Vercel deployment runs: it carries no PyTorch and no
model weights, so it fits a serverless function, while Gemini supplies vision,
OCR, translation and text analysis from one API.

Prompts are shared with the Mistral backend so the two providers produce
results in the same shape and the interface cannot tell them apart.
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
from backend.providers.ensemble import run_ensemble
from backend.providers.gemini_client import GeminiClient, rate_limited
from backend.providers.remote import (
    _ADJUDICATE_SYSTEM,
    _ANALYTICS_SYSTEM,
    _CAPTION_SYSTEM,
    _CLASSIFY_SYSTEM,
    _DETECT_SYSTEM,
    _SENTIMENT_LABELS,
    _SUMMARY_SYSTEM,
    _clamp_confidence,
    _detect_language,
)
from backend.utils.languages import language_name
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.gemini.backend")

_OCR_SYSTEM = (
    "You transcribe text from images and documents. Reproduce every piece of "
    "readable text exactly as written, in reading order, one line per line of "
    "the source. Do not summarise, translate, correct or describe. If there is "
    "no readable text, reply with nothing at all."
)


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._gemini = settings.gemini
        self._client = GeminiClient(settings.gemini)

    def warmup(self) -> None:
        if not self._gemini.configured:
            logger.warning(
                "Gemini selected but no key is set; requests will fail until "
                "OMNIAI_GEMINI_API_KEY is provided."
            )
            return
        logger.info(
            "Gemini ready (model=%s, fast=%s, reasoning=%s)",
            self._gemini.model, self._gemini.fast_model, self._gemini.reasoning_model,
        )

    def capabilities(self) -> set[str]:
        if not self._gemini.configured:
            return set()
        return {
            "classify_image", "detect_objects", "extract_text", "caption_image",
            "summarize", "translate", "analyze_text",
        }

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def _classify_once(self, image_path: str, model: str) -> Classification:
        data = self._client.generate_json(
            system=_CLASSIFY_SYSTEM,
            user="Classify the main subject of this image.",
            file_path=image_path,
            model=model,
            max_tokens=512,
        )
        label = str(data.get("label") or "unknown").strip().lower()
        return {"label": label or "unknown", "confidence": _clamp_confidence(data.get("confidence"))}

    def classify_image(self, image_path: str) -> Classification:
        primary = self._gemini.model

        if not self._settings.mistral.ensemble or rate_limited():
            result = self._classify_once(image_path, primary)
            result["ensemble"] = {"agreement": "single", "models": [primary]}
            return result

        def adjudicate(first: Classification, second: Classification) -> dict[str, Any]:
            verdict = self._client.generate_json(
                system=_ADJUDICATE_SYSTEM,
                user=(
                    "Two classifiers disagree about the main subject of this image.\n"
                    f"Classifier A says: {first['label']}\n"
                    f"Classifier B says: {second['label']}\n"
                    "Look at the image and decide. You may return a third label if "
                    "both are wrong."
                ),
                file_path=image_path,
                model=self._gemini.reasoning_model,
                # The adjudicator's job is to reason, so let it think at the
                # model's default depth and give it room to do so.
                thinking=None,
                max_tokens=4096,
            )
            verdict["confidence"] = _clamp_confidence(verdict.get("confidence"), first["confidence"])
            return verdict

        return run_ensemble(
            classify_once=lambda model: self._classify_once(image_path, model),
            adjudicate=adjudicate,
            primary_model=primary,
            second_model=self._gemini.fast_model,
            adjudicator_model=self._gemini.reasoning_model,
            timeout_seconds=self._gemini.timeout_seconds + 5,
        )

    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult:
        focus = (
            "Focus on animals; name each to species level where you can."
            if wildlife else
            "Include every distinct physical object you can identify."
        )
        data = self._client.generate_json(
            system=_DETECT_SYSTEM,
            user=f"Identify the objects in this image. {focus}",
            file_path=image_path,
            max_tokens=1500,
        )
        detections: list[dict[str, Any]] = []
        raw = data.get("objects")
        if isinstance(raw, list):
            for item in raw[:15]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip().lower()
                if label:
                    detections.append({
                        "label": label,
                        "confidence": _clamp_confidence(item.get("confidence")),
                        "bbox": None,
                    })
        return {"object_count": len(detections), "detections": detections, "bbox_available": False}

    def extract_text(self, image_path: str) -> OcrResult:
        text = self._client.generate(
            system=_OCR_SYSTEM,
            user="Transcribe all readable text.",
            file_path=image_path,
            model=self._gemini.ocr_model,
            # A full multi-page document can run to thousands of words.
            max_tokens=16384,
            temperature=0.0,
        ).strip()

        if not text:
            return {"has_text": False, "text": "", "lines": [], "language_hint": None}

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return {
            "has_text": True,
            "text": "\n".join(lines),
            "lines": lines,
            "word_count": len([w for w in text.split() if w]),
            "language_hint": "auto",
        }

    def extract_text_from_pdf(self, pdf_path: str) -> OcrResult:
        """OCR a whole PDF in one request, with no local rasteriser.

        On serverless hosts there is no Poppler to turn a scanned PDF into page
        images, so the document goes to the model directly instead.
        """
        if Path(pdf_path).suffix.lower() != ".pdf":
            raise BackendUnavailable("extract_text_from_pdf expects a PDF")
        return self.extract_text(pdf_path)

    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult:
        instruction = f"Describe this image. Context hint: {prompt}." if prompt else "Describe this image."
        caption = self._client.generate(
            system=_CAPTION_SYSTEM, user=instruction, file_path=image_path, max_tokens=400,
        ).strip().strip('"')
        if caption:
            caption = caption[0].upper() + caption[1:]
        return {"caption": caption or "No caption could be generated for this image."}

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    def summarize(self, text: str, *, max_length: int = 200) -> str:
        return self._client.generate(
            system=_SUMMARY_SYSTEM,
            user=f"Summarise the following text.\n\n{text[:30000]}",
            max_tokens=1500,
            temperature=0.3,
        ).strip()

    def translate(self, text: str, *, source_lang: str, target_lang: str) -> TranslationResult:
        system = (
            f"You are a professional translator. Translate the user's text from "
            f"{language_name(source_lang)} into {language_name(target_lang)}. Preserve "
            "meaning, tone, names, numbers and line breaks. Output only the translation."
        )
        try:
            translated = self._client.generate(
                system=system, user=text, max_tokens=4096, temperature=0.1,
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
        data = self._client.generate_json(
            system=_ANALYTICS_SYSTEM,
            user=f"Analyse the sentiment and named entities of this text:\n\n{text[:8000]}",
            max_tokens=2048,
        )

        sentiment = str(data.get("sentiment") or "NEUTRAL").strip().upper()
        if sentiment not in _SENTIMENT_LABELS:
            sentiment = "NEUTRAL"

        entities = []
        raw = data.get("entities")
        if isinstance(raw, list):
            for item in raw[:60]:
                if isinstance(item, dict):
                    value = str(item.get("text") or "").strip()
                    tag = str(item.get("label") or "").strip().upper()
                    if value and tag:
                        entities.append({"text": value, "label": tag})

        return {
            "original_language": language,
            "translated_text": None,
            "sentiment": sentiment,
            "sentiment_confidence": _clamp_confidence(data.get("confidence"), 50.0),
            "entities": entities,
            "explanation": (
                f"Detected input language as '{language}'. Sentiment and "
                f"{len(entities)} named entities were extracted by {self._gemini.model}."
            ),
        }
