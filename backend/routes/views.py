"""Server-rendered pages."""

from __future__ import annotations

from flask import Blueprint, render_template

from backend.agent.taxonomy import SPECIES_COUNT
from backend.config import get_settings
from backend.providers import get_backend
from backend.utils.languages import display_names, translation_codes

views = Blueprint("views", __name__)


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
        profile=settings.profile,
        backend_name=backend.name,
        backend_ready=bool(backend.capabilities()),
        setup_hint=settings.setup_hint(),
        species_count=SPECIES_COUNT,
        max_upload_mb=round(settings.limits.max_upload_bytes / (1024 * 1024)),
        max_text_chars=settings.limits.max_text_chars,
    )
