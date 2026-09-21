"""Application factory.

The original module created a Flask app at import time and loaded every model
inside the ``/`` route handler, so the first page view blocked on several
gigabytes of weights and two concurrent views could both start loading them.
Construction and warmup are now explicit and separable, which is also what
makes the app testable.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from backend.config import Settings, get_settings
from backend.providers import build_backend, set_backend
from backend.utils.file_utils import purge_stale_uploads
from backend.utils.logger import get_logger
from backend.utils.security import SlidingWindowRateLimiter

logger = get_logger("OmniAI")


def create_app(settings: Settings | None = None, *, warm: bool = True) -> Flask:
    """Build the application.

    Args:
        settings: Override configuration. Defaults to the environment.
        warm: Pre-load models. Tests pass ``False`` to skip the slow path.
    """
    settings = settings or get_settings()

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.update(
        MAX_CONTENT_LENGTH=settings.limits.max_upload_bytes,
        JSON_SORT_KEYS=False,
        OMNIAI_SETTINGS=settings,
    )

    app.extensions["omniai_rate_limiter"] = SlidingWindowRateLimiter(
        settings.limits.rate_limit_per_minute
    )

    _register_blueprints(app)
    _register_error_handlers(app)
    _register_security_headers(app, settings)

    purge_stale_uploads()

    backend = build_backend(settings)
    set_backend(backend)
    if warm:
        try:
            backend.warmup()
        except Exception:
            # A warmup failure must not prevent the app from starting; the
            # health endpoint will report the degraded state.
            logger.exception("Backend warmup failed; continuing in degraded state")

    logger.info("OmniAI Cloud ready: %s", settings.describe())
    return app


def _register_blueprints(app: Flask) -> None:
    from backend.routes.api import api
    from backend.routes.views import views

    app.register_blueprint(views)
    app.register_blueprint(api)


def _register_error_handlers(app: Flask) -> None:
    def wants_json() -> bool:
        return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"

    @app.errorhandler(404)
    def not_found(_: Any):
        if wants_json():
            return jsonify({"detail": "Not found"}), 404
        return jsonify({"detail": "Not found"}), 404

    @app.errorhandler(413)
    def too_large(_: Any):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"detail": f"File is too large. The maximum is {limit} MB."}), 413

    @app.errorhandler(500)
    def server_error(error: Any):
        logger.exception("Unhandled error: %s", error)
        return jsonify({"detail": "Internal server error"}), 500


def _register_security_headers(app: Flask, settings: Settings) -> None:
    # Everything the page needs is served from this origin. Bootstrap and its
    # CDN dependency were removed during the interface rebuild, so no
    # third-party script or style host has to be allowed.
    #
    # script-src stays strict, which is where cross-site scripting actually
    # lives. style-src permits inline because the progress meter and the
    # confidence bars set a width per render, and a CSS class cannot express a
    # continuous value. No untrusted input reaches a style attribute: every
    # model-produced string is written with textContent.
    #
    # Framing is refused everywhere except inside a Hugging Face Space, whose
    # page on huggingface.co shows the app in an iframe. There, and only
    # there, huggingface.co may frame it; any other site still may not.
    embeddable = settings.on_hf_space
    frame_ancestors = "'self' https://huggingface.co" if embeddable else "'none'"
    csp = (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        f"frame-ancestors {frame_ancestors}; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    @app.after_request
    def apply_headers(response):
        response.headers.setdefault("Content-Security-Policy", csp)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if not embeddable:
            # X-Frame-Options cannot name an allowed parent, so on a Space the
            # CSP frame-ancestors directive above is the control instead.
            response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        if settings.cors_origins:
            origin = request.headers.get("Origin")
            if origin in settings.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
        return response
