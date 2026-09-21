"""The paper profile: the models published in the OmniAI Cloud paper.

=============  ======================================  ==========================
Capability     Model                                   Hugging Face repository
=============  ======================================  ==========================
Classify       ResNet50 (torchvision ImageNet V2)      timm/resnet50.tv2_in1k
Detect         YOLOv8n                                 Ultralytics/YOLOv8
Wildlife       YOLOv8x                                 Ultralytics/YOLOv8
OCR            PaddleOCR 3, PP-OCRv5                   PaddlePaddle/PP-OCRv5_*
Caption        BLIP (large)                            Salesforce/blip-image-captioning-large
Translate      NLLB-200 (distilled 600M)               facebook/nllb-200-distilled-600M
Sentiment      RoBERTa                                 cardiffnlp/twitter-roberta-base-sentiment-latest
Entities       spaCy en_core_web_sm                    spacy/en_core_web_sm
Summarise      T5 (base)                               google-t5/t5-base
=============  ======================================  ==========================

Every model executes in this process. There is no API key, no quota and so no
rate limit: the only limit is the CPU. Each model is loaded once and kept (the
paper's singleton pattern), and the slow ones are loaded at startup on a
background thread (its pre-initialisation), so the first visitor does not pay
for them and the server answers health checks while they load.
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from backend.config import Settings
from backend.modules.hub import model_label
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

logger = get_logger("OmniAI.paper")

_T = TypeVar("_T")


def _reported(capability: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Turn an unexpected model error into a failure of that one step.

    The pipeline treats BackendUnavailable as "this capability did not run",
    records a warning and carries on with the other models. Any other
    exception would abort the whole analysis with a 500, so one misbehaving
    model would cost the user every other result as well.
    """

    def decorate(method: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            try:
                return method(*args, **kwargs)
            except BackendUnavailable:
                raise
            except Exception as error:
                logger.exception("%s failed", capability)
                raise BackendUnavailable(
                    f"{capability} failed inside the model ({type(error).__name__}). "
                    "The other results are unaffected."
                ) from error

        return wrapper

    return decorate


# Pre-initialisation order: what a first request most likely needs, first.
_CORE = ("ocr", "translate")
_ALL = ("classify", "detect", "ocr", "caption", "sentiment", "ner", "translate", "summarize", "wildlife")


class PaperBackend(Backend):
    name = "paper"
    ocr_label = "PaddleOCR (PP-OCRv5)"

    def _label(self, key: str) -> str:
        """The model's name, marked int8 where it runs quantised."""
        label = model_label(key)
        if key in {"caption", "translate", "summarize"} and self._settings.quantize:
            label += ", int8 on CPU"
        return label

    @property
    def summary_label(self) -> str:
        if self._settings.summarizer == "mistral" and self._settings.mistral.configured:
            return f"Mistral ({self._settings.mistral.text_model}), T5 as fallback"
        return self._label("summarize")

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._preload_thread: threading.Thread | None = None
        self._preload_failures: dict[str, str] = {}
        _quieten_libraries()
        _size_thread_pools()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def model_handle(key: str) -> Any:
        """The lazy handle for one model key, imported on demand."""
        if key == "classify":
            from backend.modules.image_classification import MODEL
        elif key == "detect":
            from backend.modules.object_detection import MODEL
        elif key == "wildlife":
            from backend.modules.wildlife_detection import MODEL
        elif key == "caption":
            from backend.modules.blip_vision_language import MODEL
        elif key == "translate":
            from backend.modules.translation_ai import MODEL
        elif key == "summarize":
            from backend.modules.summarization import MODEL
        elif key == "sentiment":
            from backend.modules.text_analytics import SENTIMENT as MODEL
        elif key == "ner":
            from backend.modules.text_analytics import NER as MODEL
        else:
            raise KeyError(key)
        return MODEL

    def _preload(self, keys: tuple[str, ...]) -> None:
        for key in keys:
            try:
                if key == "ocr":
                    from backend.modules.ocr_paddle import initialize_ocr_engines

                    initialize_ocr_engines(self._settings.ocr_langs)
                else:
                    self.model_handle(key).get()
            except Exception as error:
                # A model that fails here is retried on first use, where the
                # failure is reported against the request that needed it.
                self._preload_failures[key] = str(error)[:300]
                logger.exception("Preloading %s failed", key)
        logger.info("Paper profile ready")

    def warmup(self) -> None:
        keys = {"all": _ALL, "core": _CORE}.get(self._settings.preload, ())
        if not keys or self._preload_thread is not None:
            return
        logger.info("Pre-initialising %s in the background", ", ".join(keys))
        self._preload_thread = threading.Thread(
            target=self._preload, args=(keys,), name="omniai-preload", daemon=True
        )
        self._preload_thread.start()

    def status(self) -> dict[str, Any]:
        """Which models are in memory, for the health endpoint."""
        from backend.modules.lazy import loaded_models

        loaded = loaded_models()
        preloading = bool(self._preload_thread and self._preload_thread.is_alive())
        return {
            "models_loaded": sorted(loaded),
            "load_seconds": loaded,
            "preloading": preloading,
            "preload_failures": dict(self._preload_failures),
        }

    def capabilities(self) -> set[str]:
        return {
            "classify_image", "detect_objects", "extract_text", "caption_image",
            "summarize", "translate", "analyze_text",
        }

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    @_reported("Image classification")
    def classify_image(self, image_path: str) -> Classification:
        from backend.modules.image_classification import classify_image

        result: dict[str, Any] = classify_image(image_path)
        result["ensemble"] = {"agreement": "single", "models": [model_label("classify")]}
        result["model"] = model_label("classify")
        return result  # type: ignore[return-value]

    @_reported("Object detection")
    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult:
        if wildlife:
            from backend.modules.wildlife_detection import detect_wildlife

            result: dict[str, Any] = detect_wildlife(image_path)
            key = "wildlife"
        else:
            from backend.modules.object_detection import detect_objects

            result = detect_objects(image_path)
            key = "detect"

        # YOLO returns pixel coordinates, so the overlay can be drawn.
        return {**result, "bbox_available": True, "model": model_label(key)}  # type: ignore[return-value]

    @_reported("Text extraction")
    def extract_text(self, image_path: str) -> OcrResult:
        from backend.modules.ocr_paddle import extract_text_paddle

        result = extract_text_paddle(image_path, self._settings.ocr_langs)
        result["model"] = f"{self.ocr_label}, PaddlePaddle on Hugging Face"
        return result  # type: ignore[return-value]

    @_reported("Captioning")
    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult:
        from backend.modules.blip_vision_language import generate_image_caption

        result: dict[str, Any] = generate_image_caption(image_path, prompt=prompt)
        result["model"] = self._label("caption")
        return result  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    @_reported("Summarisation")
    def summarize(self, text: str, *, max_length: int = 200) -> str:
        from backend.modules.summarization import generate_summary

        return generate_summary(text, max_length=max_length)

    @_reported("Translation")
    def translate(self, text: str, *, source_lang: str, target_lang: str) -> TranslationResult:
        from backend.modules.translation_ai import translate_text

        result: dict[str, Any] = translate_text(text, source_lang=source_lang, target_lang=target_lang)
        result["model"] = self._label("translate")
        return result  # type: ignore[return-value]

    @_reported("Text analytics")
    def analyze_text(self, text: str) -> TextAnalytics:
        from backend.modules.text_analytics import analyze_text

        result: dict[str, Any] = analyze_text(text)
        result["model"] = f"{model_label('ner')} + {model_label('sentiment')}"
        return result  # type: ignore[return-value]


def _quieten_libraries() -> None:
    """Keep library progress bars and load reports out of the service log."""
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def _size_thread_pools() -> None:
    """Run as many inference threads as the container has CPUs, no more."""
    from backend.utils.cpu import effective_cpus

    threads = str(effective_cpus())
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, threads)
    try:
        import torch

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        logger.info("Inference threads: %s", torch.get_num_threads())
    except Exception:
        pass


def ensure_paper_importable() -> None:
    """Fail fast, with a readable message, if the paper stack is not installed."""
    from backend.config import PAPER_MODULES, _installed

    missing = [module for module in PAPER_MODULES if not _installed(module)]
    if missing:
        raise BackendUnavailable(
            "The 'paper' profile needs packages that are not installed: "
            + ", ".join(missing)
            + ". Run it with Docker (`docker compose up --build`), or install "
            "requirements/paper.txt on Python 3.10 to 3.13. PaddlePaddle publishes "
            "no wheels for Python 3.14."
        )
