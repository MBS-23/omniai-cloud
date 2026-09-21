"""Local backend: the models described in the OmniAI Cloud paper.

ResNet50, YOLOv8n/YOLOv8x, PaddleOCR, BLIP, NLLB-200, spaCy and a RoBERTa
sentiment head, all executing in-process. This is the ``full`` profile.

Every heavyweight import happens inside a method, not at module scope. That is
what allows the ``lite`` profile to run on a machine with no PyTorch installed
while this file still sits in the package.
"""

from __future__ import annotations

import threading
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
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.local")


class LocalBackend(Backend):
    name = "local"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._warm = False
        # Guards warmup. The original code used a bare module-level boolean,
        # which two concurrent first-requests could both read as False and then
        # both load several gigabytes of weights.
        self._lock = threading.Lock()
        self._summariser = None

    # ------------------------------------------------------------------
    def warmup(self) -> None:
        """Pre-load the slow models.

        The paper's ablation (Table, section 6.3) measures this as the
        difference between a 84-97 s first request and a 1.8-2.5 s one, so it
        runs eagerly rather than on demand.
        """
        if self._warm:
            return
        with self._lock:
            if self._warm:
                return

            from backend.modules.blip_vision_language import initialize_blip_model
            from backend.modules.ocr_paddle import initialize_ocr_engines

            logger.info("Warming local models (this is the slow path, once per process)")
            initialize_ocr_engines()
            initialize_blip_model()
            self._warm = True
            logger.info("Local models ready")

    def capabilities(self) -> set[str]:
        caps = {
            "classify_image",
            "detect_objects",
            "extract_text",
            "caption_image",
            "translate",
            "analyze_text",
        }
        # Summarisation is abstractive and needs the hosted model; without a
        # key it degrades to an extractive fallback, which still counts.
        caps.add("summarize")
        return caps

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def classify_image(self, image_path: str) -> Classification:
        from backend.modules.image_classification import classify_image

        return classify_image(image_path)

    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult:
        if wildlife:
            from backend.modules.wildlife_detection import detect_wildlife

            result: dict[str, Any] = detect_wildlife(image_path)
        else:
            from backend.modules.object_detection import detect_objects

            result = detect_objects(image_path)

        # YOLO always returns pixel coordinates, so the overlay is trustworthy.
        return {**result, "bbox_available": True}  # type: ignore[return-value]

    def extract_text(self, image_path: str) -> OcrResult:
        from backend.modules.ocr_paddle import extract_text_paddle

        return extract_text_paddle(image_path)  # type: ignore[return-value]

    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult:
        from backend.modules.blip_vision_language import generate_image_caption

        return generate_image_caption(image_path, prompt=prompt)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    def summarize(self, text: str, *, max_length: int = 200) -> str:
        from backend.modules.summarization import generate_summary

        return generate_summary(text, max_length=max_length)

    def translate(self, text: str, *, source_lang: str, target_lang: str) -> TranslationResult:
        from backend.modules.translation_ai import translate_text

        return translate_text(  # type: ignore[return-value]
            text, source_lang=source_lang, target_lang=target_lang
        )

    def analyze_text(self, text: str) -> TextAnalytics:
        from backend.modules.text_analytics import analyze_text

        return analyze_text(text)  # type: ignore[return-value]


def ensure_local_importable() -> None:
    """Fail fast with a readable message if the full profile cannot run.

    Without this, selecting ``full`` on a machine that only has the lite
    dependencies produces a raw ModuleNotFoundError from somewhere deep in a
    request handler.
    """
    missing: list[str] = []
    for module, package in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("ultralytics", "ultralytics"),
        ("paddleocr", "paddleocr"),
        ("spacy", "spacy"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        raise BackendUnavailable(
            "The 'full' profile needs packages that are not installed: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements/full.txt` on "
            "Python 3.11, or set OMNIAI_PROFILE=lite to use the hosted backend."
        )
