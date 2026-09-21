"""Offline backend: every model runs on this machine, no API key required.

This is the ``offline`` profile. It uses Hugging Face weights cached locally,
so once ``backend.models.download_models`` has run there is no network call at
request time, no rate limit and no per-request cost.

Model choices track the architectures named in the paper where a current,
maintained checkpoint exists:

=============  ==========================================  ==================
Capability     Model                                       Note
=============  ==========================================  ==================
Classify       microsoft/resnet-50                         ResNet50, ImageNet-1k
Detect         facebook/detr-resnet-50                     COCO classes, real boxes
OCR            EasyOCR (CRAFT detector + CRNN recogniser)  full-page, 80+ scripts
Caption        Salesforce/blip-image-captioning-base       BLIP
Translate      facebook/nllb-200-distilled-600M            NLLB-200
Sentiment      cardiffnlp/twitter-roberta-base-sentiment   RoBERTa
Entities       dslim/bert-base-NER                         replaces spaCy
Summarise      sshleifer/distilbart-cnn-12-6               replaces the hosted model
=============  ==========================================  ==================

PaddleOCR is not used: it has no wheels for current Python versions. EasyOCR
fills the same role and the original project already had an adapter for it.

Because DETR emits COCO class names, the label-refinement step works exactly
as described in the paper -- COCO's coarse ``cat``/``cow``/``bird`` labels are
still what needs refining against the classifier's species output.
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
from backend.utils.languages import nllb_code
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.offline")


def _quieten_transformers() -> None:
    """Silence library chatter that is not actionable for this application.

    transformers 5.x prints a weight-load report for every model and warns
    whenever a generation argument overrides a config default. Both are
    expected here, and both drown the pipeline's own log lines.
    """
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

MODELS = {
    "classify": "microsoft/resnet-50",
    "detect": "facebook/detr-resnet-50",
    "caption": "Salesforce/blip-image-captioning-base",
    "translate": "facebook/nllb-200-distilled-600M",
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "ner": "dslim/bert-base-NER",
    "summarize": "sshleifer/distilbart-cnn-12-6",
}

# Detection confidence floors. The wildlife path trades precision for recall,
# which is the same intent as the paper's YOLOv8n -> YOLOv8x switch: catch the
# animal even when it is partly occluded or far from the camera.
DETECT_THRESHOLD = 0.60
WILDLIFE_THRESHOLD = 0.30
MAX_DETECTIONS = 15

# ImageNet class names are comma-separated synonym lists, for example
# "African elephant, Loxodonta africana". Only the first is a common name.
def primary_label(raw: str) -> str:
    return (raw or "").split(",")[0].strip().lower()


class _Registry:
    """Loads each model once, on first use, and hands out the cached object."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str, build: Any) -> Any:
        if key in self._items:
            return self._items[key]
        with self._lock:
            if key not in self._items:
                logger.info("Loading %s (%s)", key, MODELS.get(key, key))
                self._items[key] = build()
                logger.info("Loaded %s", key)
            return self._items[key]

    def loaded(self) -> set[str]:
        return set(self._items)


class OfflineBackend(Backend):
    name = "offline"

    def __init__(self, settings: Settings) -> None:
        _quieten_transformers()
        self._settings = settings
        self._registry = _Registry()
        self._device = self._resolve_device()

    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        configured = self._settings.torch_device
        if configured in {"cpu", "cuda"}:
            return configured
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def warmup(self) -> None:
        """Load the models a first request is most likely to need.

        The paper's ablation measures this as the difference between an
        84-97 s first request and a 1.8-2.5 s one.
        """
        logger.info("Offline profile warming up on %s", self._device)
        try:
            self._classifier()
            self._ocr()
        except Exception:
            logger.exception("Warmup failed; models will load on first use instead")
        logger.info("Offline profile ready (loaded: %s)", ", ".join(sorted(self._registry.loaded())))

    def capabilities(self) -> set[str]:
        return {
            "classify_image", "detect_objects", "extract_text", "caption_image",
            "summarize", "translate", "analyze_text",
        }

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------
    def _classifier(self):
        def build():
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            repo = MODELS["classify"]
            processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForImageClassification.from_pretrained(repo)
            return processor, model.to(self._device).eval()

        return self._registry.get("classify", build)

    def _detector(self):
        def build():
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            repo = MODELS["detect"]
            processor = AutoImageProcessor.from_pretrained(repo)
            model = AutoModelForObjectDetection.from_pretrained(repo)
            return processor, model.to(self._device).eval()

        return self._registry.get("detect", build)

    def _captioner(self):
        def build():
            from transformers import BlipForConditionalGeneration, BlipProcessor

            repo = MODELS["caption"]
            processor = BlipProcessor.from_pretrained(repo)
            model = BlipForConditionalGeneration.from_pretrained(repo)
            return processor, model.to(self._device).eval()

        return self._registry.get("caption", build)

    def _ocr(self):
        def build():
            import easyocr

            # EasyOCR downloads its detector and recogniser on first use.
            return easyocr.Reader(["en"], gpu=(self._device == "cuda"), verbose=False)

        return self._registry.get("ocr", build)

    def _translator(self):
        def build():
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            repo = MODELS["translate"]
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSeq2SeqLM.from_pretrained(repo)
            return tokenizer, model.to(self._device).eval()

        return self._registry.get("translate", build)

    def _sentiment(self):
        def build():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            repo = MODELS["sentiment"]
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSequenceClassification.from_pretrained(repo)
            return tokenizer, model.to(self._device).eval()

        return self._registry.get("sentiment", build)

    def _ner(self):
        def build():
            from transformers import pipeline

            return pipeline(
                "token-classification",
                model=MODELS["ner"],
                aggregation_strategy="simple",
                device=0 if self._device == "cuda" else -1,
            )

        return self._registry.get("ner", build)

    def _summariser(self):
        def build():
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            repo = MODELS["summarize"]
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForSeq2SeqLM.from_pretrained(repo)
            return tokenizer, model.to(self._device).eval()

        return self._registry.get("summarize", build)

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def classify_image(self, image_path: str) -> Classification:
        import torch
        from PIL import Image

        processor, model = self._classifier()
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(self._device)
            probabilities = model(**inputs).logits.softmax(-1)[0]

        score, index = probabilities.max(dim=-1)
        raw = model.config.id2label[index.item()]

        return {
            "label": primary_label(raw),
            "confidence": round(score.item() * 100, 2),
            "ensemble": {
                "agreement": "single",
                "models": [MODELS["classify"]],
                "synonyms": raw,
            },
        }

    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult:
        import torch
        from PIL import Image

        processor, model = self._detector()
        image = Image.open(image_path).convert("RGB")
        threshold = WILDLIFE_THRESHOLD if wildlife else DETECT_THRESHOLD

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(self._device)
            outputs = model(**inputs)

        processed = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[image.height, image.width]]).to(self._device),
            threshold=threshold,
        )[0]

        detections: list[dict[str, Any]] = []
        order = processed["scores"].argsort(descending=True)
        for position in order[:MAX_DETECTIONS]:
            box = [round(float(v), 2) for v in processed["boxes"][position].tolist()]
            detections.append({
                "label": primary_label(model.config.id2label[processed["labels"][position].item()]),
                "confidence": round(float(processed["scores"][position]) * 100, 2),
                "bbox": box,
            })

        return {
            "object_count": len(detections),
            "detections": detections,
            # DETR returns pixel coordinates, so the overlay is trustworthy.
            "bbox_available": True,
            "threshold": threshold,
        }

    def extract_text(self, image_path: str) -> OcrResult:
        reader = self._ocr()
        # detail=1 returns (box, text, confidence) per region.
        regions = reader.readtext(image_path, detail=1, paragraph=False)

        lines: list[str] = []
        boxes: list[dict[str, Any]] = []
        for box, text, confidence in regions:
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            lines.append(cleaned)
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            boxes.append({
                "text": cleaned,
                "confidence": round(float(confidence) * 100, 2),
                "bbox": [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)],
            })

        if not lines:
            return {"has_text": False, "text": "", "lines": [], "language_hint": None}

        text = "\n".join(lines)
        return {
            "has_text": True,
            "text": text,
            "lines": lines,
            "word_count": len([word for word in text.split() if word]),
            "language_hint": "latin",
            "boxes": boxes,
        }

    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult:
        import torch
        from PIL import Image

        processor, model = self._captioner()
        image = Image.open(image_path).convert("RGB")

        inputs = (
            processor(images=image, text=prompt, return_tensors="pt")
            if prompt else
            processor(images=image, return_tensors="pt")
        ).to(self._device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                min_new_tokens=8,
                num_beams=4,
                length_penalty=1.1,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        # A prompted BLIP caption repeats the prompt back; drop it.
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip(" ,.")
        if caption:
            caption = caption[0].upper() + caption[1:]

        return {"caption": caption or "No caption could be generated for this image."}

    # ------------------------------------------------------------------
    # Language
    # ------------------------------------------------------------------
    def summarize(self, text: str, *, max_length: int = 200) -> str:
        import torch

        tokenizer, model = self._summariser()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._device)

        # The checkpoint's generation config carries its own max_length and
        # min_length. Passing max_new_tokens alongside them makes transformers
        # warn on every call, so clear the pair we are overriding.
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=min(max(60, max_length), 300),
                min_new_tokens=30,
                max_length=None,
                min_length=None,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def translate(self, text: str, *, source_lang: str, target_lang: str) -> TranslationResult:
        import torch

        tokenizer, model = self._translator()
        tokenizer.src_lang = nllb_code(source_lang)
        target_token = nllb_code(target_lang)

        forced_bos = tokenizer.convert_tokens_to_ids(target_token)
        if forced_bos is None or forced_bos == tokenizer.unk_token_id:
            return {
                "translated_text": text,
                "source_language": source_lang,
                "target_language": target_lang,
                "translation_applied": False,
                "translation_error": f"'{target_lang}' is not a supported translation target.",
            }

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs, forced_bos_token_id=forced_bos, max_length=512, num_beams=4
            )

        translated = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        return {
            "translated_text": translated or text,
            "source_language": source_lang,
            "target_language": target_lang,
            "translation_applied": bool(translated),
        }

    def analyze_text(self, text: str) -> TextAnalytics:
        import torch

        language = _detect_language(text)

        tokenizer, model = self._sentiment()
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            scores = model(**inputs).logits.softmax(-1)[0]

        best = int(scores.argmax())
        raw_label = model.config.id2label[best].lower()
        sentiment = (
            "POSITIVE" if "pos" in raw_label
            else "SAD" if "neg" in raw_label
            else "NEUTRAL"
        )

        entities = []
        try:
            for item in self._ner()(text[:2000]):
                value = str(item.get("word", "")).strip()
                group = str(item.get("entity_group", "")).upper()
                if value and group:
                    entities.append({"text": value, "label": _map_entity(group)})
        except Exception as error:
            logger.warning("Entity recognition failed: %s", error)

        return {
            "original_language": language,
            "translated_text": None,
            "sentiment": sentiment,
            "sentiment_confidence": round(float(scores[best]) * 100, 2),
            "entities": entities,
            "explanation": (
                f"Detected input language as '{language}'. Sentiment from "
                f"{MODELS['sentiment'].split('/')[-1]}, entities from "
                f"{MODELS['ner'].split('/')[-1]}, both running locally."
            ),
        }


# CoNLL tags to the spaCy-style labels the interface already renders.
_ENTITY_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE", "MISC": "MISC"}


def _map_entity(group: str) -> str:
    return _ENTITY_MAP.get(group, group)


def _detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return "unknown"


def ensure_offline_importable() -> None:
    """Fail fast, with a readable message, if the offline stack is missing."""
    missing: list[str] = []
    for module, package in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("easyocr", "easyocr"),
        ("PIL", "Pillow"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        raise BackendUnavailable(
            "The 'offline' profile needs packages that are not installed: "
            + ", ".join(missing)
            + ". Install them with `pip install -r requirements/offline.txt`, then "
            "run `python -m backend.models.download_models`."
        )
