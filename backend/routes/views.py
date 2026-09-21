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


@views.get("/")
def home() -> str:
    settings = get_settings()
    backend = get_backend()

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
        capability_count=len(backend.capabilities()),
        language_count=len(translation_codes()),
        max_upload_mb=round(settings.limits.max_upload_bytes / (1024 * 1024)),
        max_text_chars=settings.limits.max_text_chars,
    )
