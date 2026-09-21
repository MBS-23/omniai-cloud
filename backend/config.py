"""Central configuration for OmniAI Cloud.

Everything the application needs to know about its environment lives here, so
no module has to reach for ``os.getenv`` on its own. Settings are read once at
import time and exposed through :func:`get_settings`.

Two runtime profiles are supported:

``full``
    Every model runs locally (ResNet50, YOLOv8, PaddleOCR, BLIP, NLLB, spaCy).
    This is the configuration described in the OmniAI Cloud paper. It needs
    Python 3.11 and roughly 6 GB of downloaded weights.

``lite``
    The same pipeline, with model execution delegated to the Mistral API.
    Small enough to deploy on ordinary hosting and to run on any Python 3.11+.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

Profile = Literal["offline", "full", "lite"]

# Keys are read as OMNIAI_<NAME>; the bare <NAME> is accepted as a fallback so
# that .env files written before the namespacing was introduced keep working.
_LEGACY_ALIASES = {
    "OMNIAI_MISTRAL_API_KEY": "MISTRAL_API_KEY",
    "OMNIAI_PROFILE": "PROFILE",
}


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None and name in _LEGACY_ALIASES:
        value = os.getenv(_LEGACY_ALIASES[name])
    if value is None:
        return None
    value = value.strip().strip('"').strip("'")
    return value or None


def _str(name: str, default: str) -> str:
    return _raw(name) or default


def _int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _csv(name: str) -> list[str]:
    raw = _raw(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class MistralSettings:
    """Credentials and model IDs for the hosted Mistral backend.

    Defaults target models available on a standard key. ``mistral-large`` is
    deliberately not used: it is absent from ordinary tiers, and selecting a
    model the key cannot reach fails every request with an unhelpful 400.
    """

    api_key: str | None
    text_model: str
    vision_model: str
    ocr_model: str
    # A reasoning model, used to adjudicate when the ensemble disagrees.
    reasoning_model: str = "magistral-medium-latest"
    # A cheaper, faster model for the second opinion in the ensemble.
    fast_model: str = "mistral-small-latest"
    # Run vision classification through two models and reconcile them.
    ensemble: bool = True
    base_url: str = "https://api.mistral.ai/v1"
    timeout_seconds: int = 90

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Limits:
    """Hard ceilings applied to every request, before any model runs."""

    max_upload_bytes: int
    max_pdf_pages: int
    max_image_pixels: int
    max_text_chars: int
    rate_limit_per_minute: int


@dataclass(frozen=True)
class Settings:
    profile: Profile
    host: str
    port: int
    debug: bool
    cors_origins: list[str]
    limits: Limits
    mistral: MistralSettings
    model_dir: Path
    poppler_path: str | None
    torch_device: str
    project_root: Path = field(default=PROJECT_ROOT)

    @property
    def is_lite(self) -> bool:
        return self.profile == "lite"

    @property
    def is_offline(self) -> bool:
        """True when no network call is made to serve a request."""
        return self.profile in {"offline", "full"}

    @property
    def upload_dir(self) -> Path:
        return self.project_root / "temp_uploads"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    def describe(self) -> dict[str, object]:
        """Non-secret summary, safe to log and to expose on /api/health."""
        return {
            "profile": self.profile,
            "debug": self.debug,
            "mistral_configured": self.mistral.configured,
            "torch_device": self.torch_device,
            "max_upload_mb": round(self.limits.max_upload_bytes / (1024 * 1024), 1),
        }


def _resolve_profile() -> Profile:
    """Which backend serves requests.

    offline : every model runs locally from cached Hugging Face weights.
              No API key, no rate limits, no network at request time.
    full    : the original PyTorch + PaddleOCR stack. Needs Python 3.11.
    lite    : models served by the Mistral API. Smallest install, deployable.
    """
    raw = _str("OMNIAI_PROFILE", "offline").lower()
    if raw in {"offline", "local"}:
        return "offline"
    if raw == "full":
        return "full"
    return "lite"


def _build_settings() -> Settings:
    model_dir = Path(_str("OMNIAI_MODEL_DIR", "backend/models/downloaded_models"))
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir

    return Settings(
        profile=_resolve_profile(),
        host=_str("OMNIAI_HOST", "127.0.0.1"),
        port=_int("OMNIAI_PORT", 8000),
        debug=_bool("OMNIAI_DEBUG", False),
        cors_origins=_csv("OMNIAI_CORS_ORIGINS"),
        limits=Limits(
            max_upload_bytes=_int("OMNIAI_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
            max_pdf_pages=_int("OMNIAI_MAX_PDF_PAGES", 200),
            max_image_pixels=_int("OMNIAI_MAX_IMAGE_PIXELS", 50_000_000),
            max_text_chars=_int("OMNIAI_MAX_TEXT_CHARS", 20_000),
            rate_limit_per_minute=_int("OMNIAI_RATE_LIMIT_PER_MINUTE", 20),
        ),
        mistral=MistralSettings(
            api_key=_raw("OMNIAI_MISTRAL_API_KEY"),
            text_model=_str("OMNIAI_MISTRAL_TEXT_MODEL", "mistral-medium-latest"),
            vision_model=_str("OMNIAI_MISTRAL_VISION_MODEL", "mistral-medium-latest"),
            ocr_model=_str("OMNIAI_MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
            reasoning_model=_str("OMNIAI_MISTRAL_REASONING_MODEL", "magistral-medium-latest"),
            fast_model=_str("OMNIAI_MISTRAL_FAST_MODEL", "mistral-small-latest"),
            ensemble=_bool("OMNIAI_ENSEMBLE", True),
        ),
        model_dir=model_dir,
        poppler_path=_raw("OMNIAI_POPPLER_PATH"),
        torch_device=_str("OMNIAI_TORCH_DEVICE", "auto").lower(),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings, building them on first use."""
    global _settings
    if _settings is None:
        _settings = _build_settings()
    return _settings


def reload_settings() -> Settings:
    """Rebuild settings from the current environment. Used by tests."""
    global _settings
    _settings = None
    return get_settings()
