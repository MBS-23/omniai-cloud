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
from werkzeug.exceptions import HTTPException

from backend.config import Settings, get_settings
from backend.providers import build_backend, set_backend
from backend.utils.file_utils import purge_stale_uploads
from backend.utils.logger import get_logger
from backend.utils.security import AnalysisSlots, SlidingWindowRateLimiter

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
        # A multipart body is parsed before any view runs. Bound how many parts
        # and how much non-file form data it may carry, so a request cannot
        # make the parser do unbounded work.
        MAX_FORM_PARTS=16,
        MAX_FORM_MEMORY_SIZE=512 * 1024,
        JSON_SORT_KEYS=False,
        OMNIAI_SETTINGS=settings,
    )

    app.extensions["omniai_rate_limiter"] = SlidingWindowRateLimiter(
        settings.limits.rate_limit_per_minute
    )
    app.extensions["omniai_analysis_slots"] = AnalysisSlots(settings.max_concurrent)

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


# What a client is told for each HTTP error: enough to act on, nothing about
# how the server is built.
_ERROR_MESSAGES = {
    400: "The request could not be understood.",
    404: "Not found.",
    405: "That method is not allowed here.",
    413: "The upload is too large.",
    429: "Too many requests. Please wait a moment and try again.",
}


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(413)
    def too_large(_: Any):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"detail": f"File is too large. The maximum is {limit} MB."}), 413

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        code = error.code or 500
        return jsonify({"detail": _ERROR_MESSAGES.get(code, "The request could not be completed.")}), code

    @app.errorhandler(Exception)
    def server_error(error: Exception):
        logger.exception("Unhandled error: %s", error)
        return jsonify({"detail": "Something went wrong. Please try again."}), 500


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
            "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        )
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        response.headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")
        if request.path.startswith("/api/"):
            # Results describe what someone uploaded; no cache should keep them.
            response.headers.setdefault("Cache-Control", "no-store")
        forwarded_https = (
            settings.trusted_proxies > 0
            and request.headers.get("X-Forwarded-Proto", "").lower() == "https"
        )
        if request.is_secure or forwarded_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        if settings.cors_origins:
            origin = request.headers.get("Origin")
            if origin in settings.cors_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Vary"] = "Origin"
        return response
