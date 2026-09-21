"""Backend resolution.

One process runs exactly one backend, chosen by ``OMNIAI_PROFILE``.
"""

from __future__ import annotations

from backend.config import Settings, get_settings
from backend.providers.base import Backend, BackendUnavailable
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.providers")

_backend: Backend | None = None


def build_backend(settings: Settings | None = None) -> Backend:
    """Construct the backend for the configured profile."""
    settings = settings or get_settings()

    if settings.profile == "offline":
        from backend.providers.offline import OfflineBackend, ensure_offline_importable

        ensure_offline_importable()
        logger.info("Backend: offline (local Hugging Face models, no API key)")
        return OfflineBackend(settings)

    if settings.profile == "paper":
        from backend.providers.paper import PaperBackend, ensure_paper_importable

        ensure_paper_importable()
        logger.info("Backend: paper (the paper's own models, from Hugging Face, no API key)")
        return PaperBackend(settings)

    if settings.hosted_provider == "gemini":
        from backend.providers.gemini import GeminiBackend

        logger.info("Backend: gemini (lite profile)")
        return GeminiBackend(settings)

    from backend.providers.remote import RemoteBackend

    logger.info("Backend: mistral (lite profile)")
    return RemoteBackend(settings)


def get_backend() -> Backend:
    """Return the process-wide backend, building it on first use."""
    global _backend
    if _backend is None:
        _backend = build_backend()
    return _backend


def set_backend(backend: Backend) -> None:
    """Install an already-constructed backend as the process-wide one.

    The app factory uses this so that startup does not build the backend a
    second time on the first request.
    """
    global _backend
    _backend = backend


def reset_backend() -> None:
    """Drop the cached backend. Used by tests and by profile switches."""
    global _backend
    _backend = None


__all__ = [
    "Backend",
    "BackendUnavailable",
    "build_backend",
    "get_backend",
    "reset_backend",
]
