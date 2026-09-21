"""Server-rendered pages."""

from __future__ import annotations

from flask import Blueprint, render_template

from backend.agent.taxonomy import SPECIES_COUNT
from backend.config import get_settings
from backend.providers import get_backend
from backend.utils.languages import display_names, translation_codes

views = Blueprint("views", __name__)

# How the header names the model stack serving this page.
_STACK_LABELS = {
    "paper": "Paper models",
    "offline": "Local models",
    "lite": "Hosted models",
}

_OCR_LANGUAGE_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada"}


def _lineup(settings, languages: int) -> tuple[str, list[tuple[str, str]]]:
    """What the hero says runs here: every model, and the job each does."""
    if settings.profile == "paper":
        scripts = [_OCR_LANGUAGE_NAMES.get(code, code) for code in settings.ocr_langs]
        reads = scripts[0] if len(scripts) == 1 else ", ".join(scripts[:-1]) + " and " + scripts[-1]
        return (
            "The nine models from the paper, running on this machine. "
            "No outside AI service sees what you upload.",
            [
                ("ResNet50", "classifies the image"),
                ("YOLOv8n", "finds everyday objects"),
                ("YOLOv8x", "takes over for animals"),
                ("PaddleOCR", f"reads {reads}"),
                ("BLIP-large", "describes the scene"),
                ("NLLB-200", f"translates between {languages} languages"),
                ("RoBERTa", "scores sentiment"),
                ("spaCy", "finds names, places and organisations"),
                ("T5", "summarises documents"),
            ],
        )
    if settings.profile == "offline":
        return (
            "Local models, no API key: substitutes that install on Python 3.14.",
            [
                ("ResNet-50", "classifies the image"), ("DETR", "finds objects"),
                ("EasyOCR", "reads text"), ("BLIP-base", "describes the scene"),
                ("NLLB-200", f"translates between {languages} languages"), ("RoBERTa", "scores sentiment"),
                ("BERT-NER", "finds names"), ("DistilBART", "summarises documents"),
            ],
        )
    provider = "Gemini" if settings.hosted_provider == "gemini" else "Mistral"
    return (
        f"Hosted models, through the {provider} API.",
        [(provider, "classifies, detects, reads, describes, translates and summarises")],
    )


@views.get("/")
def home() -> str:
    settings = get_settings()
    backend = get_backend()
    lineup_intro, lineup = _lineup(settings, len(translation_codes()))

    return render_template(
        "index.html",
        app_title="OmniAI Cloud",
        app_subtitle="Unified multi-modal AI with automatic model selection",
        language_names=display_names(),
        translation_language_codes=translation_codes(),
        stack_label=_STACK_LABELS.get(settings.profile, "Models"),
        backend_ready=bool(backend.capabilities()),
        # Configuration instructions only for a developer running it locally.
        setup_hint=(
            settings.setup_hint() if settings.debug
            else "Analysis is temporarily unavailable. Please try again later."
        ),
        species_count=SPECIES_COUNT,
        lineup_intro=lineup_intro,
        lineup=lineup,
        max_upload_mb=round(settings.limits.max_upload_bytes / (1024 * 1024)),
        max_text_chars=settings.limits.max_text_chars,
    )
