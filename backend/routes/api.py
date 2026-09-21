"""JSON and streaming endpoints."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from backend.config import get_settings
from backend.core.pipeline import ProgressEvent, run_file_analysis, run_text_analysis
from backend.providers import get_backend
from backend.providers.base import BackendUnavailable
from backend.utils.file_utils import delete_file, save_uploaded_file
from backend.utils.logger import get_logger
from backend.utils.redaction import redact
from backend.utils.security import (
    IMAGE_EXTENSIONS,
    AnalysisSlots,
    RateLimitExceeded,
    ServerBusy,
    UploadRejected,
    client_identifier,
    validate_extension,
    validate_image_dimensions,
    validate_signature,
    validate_size,
    validate_text,
)

logger = get_logger("OmniAI.api")

api = Blueprint("api", __name__, url_prefix="/api")

# Sentinel pushed onto the event queue when the worker thread finishes.
_DONE = object()


def _error(message: str, status: int = 400, **extra: Any) -> tuple[Response, int]:
    # Errors frequently quote an upstream provider's message. Scrub before it
    # reaches the browser rather than trusting every path that builds one.
    payload = {"detail": redact(message), **extra}
    return jsonify(payload), status


def _rate_limit() -> None:
    limiter = current_app.extensions["omniai_rate_limiter"]
    limiter.check(
        client_identifier(
            request.remote_addr,
            request.headers.get("X-Forwarded-For"),
            get_settings().trusted_proxies,
        )
    )


def _slots() -> AnalysisSlots:
    return current_app.extensions["omniai_analysis_slots"]


def _same_site() -> bool:
    """Whether the request comes from this site's own page, or from no page.

    Browsers label every request with where it came from. A form on another
    site posting here would be "cross-site"; the app's own page is
    "same-origin", and tools such as curl send nothing, which is allowed.
    """
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return False
    origin = request.headers.get("Origin")
    if origin and origin != "null":
        allowed = {request.host_url.rstrip("/"), *get_settings().cors_origins}
        forwarded_host = request.headers.get("X-Forwarded-Host")
        if forwarded_host and get_settings().trusted_proxies:
            allowed.add(f"https://{forwarded_host}")
        return origin.rstrip("/") in allowed
    return True


def _guard() -> tuple[Response, int] | None:
    """Checks every analysis request passes before any work is done."""
    if not _same_site():
        return _error("Requests from other sites are not accepted.", 403)
    try:
        _rate_limit()
    except RateLimitExceeded as error:
        return _error(str(error), 429, retry_after=error.retry_after)
    return None


def _prepare_upload() -> tuple[Path, str]:
    """Validate the incoming file and write it to a private temp path."""
    file_storage = request.files.get("file")
    if file_storage is None:
        raise UploadRejected("Select a file to analyse.")

    extension = validate_extension(file_storage.filename)
    validate_size(file_storage.stream)
    validate_signature(file_storage.stream, extension)

    _, file_path = save_uploaded_file(file_storage.filename, file_storage.stream)

    if extension in IMAGE_EXTENSIONS:
        try:
            validate_image_dimensions(file_path)
        except Exception:
            delete_file(file_path)
            raise

    return file_path, extension


def _text_arguments() -> dict[str, Any]:
    payload = request.get_json(silent=True) or request.form
    raw_auto = str(payload.get("auto_detect_source", "true")).lower()
    return {
        "text": validate_text(payload.get("text", "")),
        "source_language": (payload.get("source_language") or "auto").lower(),
        "target_language": (payload.get("target_language") or "en").lower(),
        "auto_detect_source": raw_auto in {"true", "1", "yes", "on"},
    }


# ======================================================================
# Non-streaming analysis
# ======================================================================
@api.post("/analyze/file")
def analyze_file() -> Any:
    if (refused := _guard()) is not None:
        return refused

    file_path: Path | None = None
    slot = False
    try:
        file_path, _ = _prepare_upload()
        _slots().acquire()
        slot = True
        result = run_file_analysis(file_path, get_backend())
        return jsonify(result)
    except ServerBusy as error:
        return _error(str(error), 503, retry_after=error.retry_after)
    except (UploadRejected, ValueError) as error:
        return _error(str(error), 400)
    except BackendUnavailable as error:
        return _error(str(error), 503)
    except Exception:
        logger.exception("File analysis failed")
        return _error("Something went wrong while analysing this file.", 500)
    finally:
        if slot:
            _slots().release()
        if file_path is not None:
            delete_file(file_path)


@api.post("/analyze/text")
def analyze_text() -> Any:
    if (refused := _guard()) is not None:
        return refused

    slot = False
    try:
        arguments = _text_arguments()
        _slots().acquire()
        slot = True
        result = run_text_analysis(backend=get_backend(), **arguments)
        return jsonify(result)
    except ServerBusy as error:
        return _error(str(error), 503, retry_after=error.retry_after)
    except (UploadRejected, ValueError) as error:
        return _error(str(error), 400)
    except BackendUnavailable as error:
        return _error(str(error), 503)
    except Exception:
        logger.exception("Text analysis failed")
        return _error("Something went wrong while analysing this text.", 500)
    finally:
        if slot:
            _slots().release()


# ======================================================================
# Streaming analysis (Server-Sent Events)
# ======================================================================
def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream(work: Callable[[Callable[[ProgressEvent], None]], dict[str, Any]],
            cleanup: Callable[[], None] | None = None,
            release: Callable[[], None] | None = None) -> Response:
    """Run ``work`` on a worker thread, relaying its progress events as SSE.

    The pipeline is synchronous and CPU- or network-bound, so it runs off the
    generator thread. Events travel over a queue; the generator only formats
    and yields them.
    """
    events: queue.Queue[Any] = queue.Queue()
    outcome: dict[str, Any] = {}

    def on_event(event: ProgressEvent) -> None:
        events.put(("progress", event.to_dict()))

    def worker() -> None:
        try:
            outcome["result"] = work(on_event)
        except (UploadRejected, ValueError) as error:
            outcome["error"] = {"detail": redact(error), "status": 400}
        except BackendUnavailable as error:
            outcome["error"] = {"detail": redact(error), "status": 503}
        except Exception:
            logger.exception("Streaming analysis failed")
            outcome["error"] = {
                "detail": "Something went wrong during analysis.",
                "status": 500,
            }
        finally:
            # The analysis slot is given back when the work ends, whether or
            # not the browser is still listening.
            if release is not None:
                release()
            events.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def generate() -> Iterator[str]:
        try:
            while True:
                item = events.get()
                if item is _DONE:
                    break
                kind, payload = item
                yield _sse(kind, payload)

            thread.join(timeout=5)

            if "error" in outcome:
                yield _sse("failed", outcome["error"])
            else:
                yield _sse("result", outcome.get("result", {}))
        finally:
            if cleanup is not None:
                cleanup()

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            # Tells nginx not to buffer the stream; harmless elsewhere.
            "X-Accel-Buffering": "no",
        },
        # No "Connection" header here. It is hop-by-hop, and PEP 3333 forbids
        # a WSGI application from setting one: waitress and gunicorn reject it
        # with an AssertionError, turning every stream into a 500. Flask's
        # development server tolerates it, which is what hid this. Keep-alive
        # is the HTTP/1.1 default and is the server's business, not ours.
    )


@api.post("/analyze/file/stream")
def analyze_file_stream() -> Any:
    if (refused := _guard()) is not None:
        return refused

    try:
        file_path, _ = _prepare_upload()
    except (UploadRejected, ValueError) as error:
        return _error(str(error), 400)

    slots = _slots()
    try:
        slots.acquire()
    except ServerBusy as error:
        delete_file(file_path)
        return _error(str(error), 503, retry_after=error.retry_after)

    backend = get_backend()
    return _stream(
        lambda on_event: run_file_analysis(file_path, backend, on_event=on_event),
        cleanup=lambda: delete_file(file_path),
        release=slots.release,
    )


@api.post("/analyze/text/stream")
def analyze_text_stream() -> Any:
    if (refused := _guard()) is not None:
        return refused

    try:
        arguments = _text_arguments()
    except (UploadRejected, ValueError) as error:
        return _error(str(error), 400)

    slots = _slots()
    try:
        slots.acquire()
    except ServerBusy as error:
        return _error(str(error), 503, retry_after=error.retry_after)

    backend = get_backend()
    return _stream(
        lambda on_event: run_text_analysis(backend=backend, on_event=on_event, **arguments),
        release=slots.release,
    )


# ======================================================================
# Health
# ======================================================================
@api.get("/health")
def health() -> Any:
    """Liveness probe for the container and for the page's warm-up indicator.

    Deliberately says nothing about configuration: that goes to the
    operator's log, not to whoever is asking.
    """
    backend = get_backend()
    ready = bool(backend.capabilities())
    payload: dict[str, Any] = {"status": "ok" if ready else "unavailable"}

    # Local backends report how many models are in memory, which the page
    # uses to show progress while a cold start loads them.
    status = getattr(backend, "status", None)
    if callable(status):
        models = status()
        payload["models"] = {
            "ready": len(models.get("models_loaded", [])),
            "loading": bool(models.get("preloading")),
        }

    if not ready:
        logger.warning("Health check: backend unavailable. %s", get_settings().setup_hint())
    return jsonify(payload), (200 if ready else 503)
