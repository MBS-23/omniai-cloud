"""Thin client for the Gemini API.

Mirrors :mod:`backend.providers.mistral_client` deliberately, so the two hosted
providers behave the same way under failure: retries only on 429 and 5xx,
patient backoff after a rate limit, a process-wide throttle so concurrent
requests do not turn one 429 into a burst of them, and errors that name the
cause without ever including the key.

The key travels in the ``x-goog-api-key`` header rather than the ``?key=``
query parameter the quickstarts show. A key in a URL ends up in proxy logs,
browser history and error messages; a header does not.
"""

from __future__ import annotations

import base64
import mimetypes
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

from backend.config import GeminiSettings
from backend.providers.base import BackendUnavailable
from backend.providers.mistral_client import extract_json_object
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.gemini")

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4
_RATE_LIMIT_BACKOFF = (2.0, 6.0, 15.0)
_COOLDOWN_SECONDS = 20.0

_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}


class _Throttle:
    """After a 429, space every request out for a cooldown window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cooling_until = 0.0
        self._next_allowed = 0.0

    @property
    def cooling(self) -> bool:
        return time.monotonic() < self._cooling_until

    def note_rate_limit(self, retry_after: float | None) -> None:
        with self._lock:
            wait = retry_after if retry_after is not None else _COOLDOWN_SECONDS
            self._cooling_until = max(self._cooling_until, time.monotonic() + wait)

    def wait_turn(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                interval = 1.2 if now < self._cooling_until else 0.0
                if now >= self._next_allowed:
                    self._next_allowed = now + interval
                    return
                sleep_for = self._next_allowed - now
            time.sleep(min(sleep_for, 5.0))


_throttle = _Throttle()


def rate_limited() -> bool:
    return _throttle.cooling


def _inline_part(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    mime = _MIME.get(file_path.suffix.lower()) or mimetypes.guess_type(str(file_path))[0]
    if mime is None:
        raise BackendUnavailable(f"Unsupported file type for Gemini: {file_path.suffix}")
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(file_path.read_bytes()).decode("ascii"),
        }
    }


def _retry_after(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class GeminiClient:
    def __init__(self, settings: GeminiSettings) -> None:
        self._settings = settings
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        if not self._settings.api_key:
            raise BackendUnavailable(
                "No Gemini API key is configured. Add OMNIAI_GEMINI_API_KEY to your "
                "environment (a free key is available at aistudio.google.com/apikey)."
            )
        return {
            "x-goog-api-key": self._settings.api_key,
            "Content-Type": "application/json",
        }

    def _post(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{_BASE_URL}/models/{model}:generateContent"
        headers = self._headers()
        last_error: Exception | None = None
        hit_rate_limit = False

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _throttle.wait_turn()
            backoff = 0.0
            try:
                response = self._session.post(
                    url, headers=headers, json=body, timeout=self._settings.timeout_seconds
                )
            except requests.RequestException as error:
                last_error = error
                logger.warning("Gemini request failed (attempt %s): %s", attempt, error)
                backoff = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            else:
                if response.status_code < 400:
                    return response.json()

                if response.status_code == 429:
                    hit_rate_limit = True
                    retry_after = _retry_after(response)
                    _throttle.note_rate_limit(retry_after)
                    index = min(attempt - 1, len(_RATE_LIMIT_BACKOFF) - 1)
                    backoff = retry_after if retry_after is not None else _RATE_LIMIT_BACKOFF[index]
                    last_error = RuntimeError("rate limited")
                    logger.warning("Gemini rate limit; waiting %.1fs", backoff)
                elif response.status_code in _RETRY_STATUS:
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                    backoff = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
                else:
                    try:
                        detail = str(response.json().get("error", {}).get("message", ""))[:220]
                    except Exception:
                        detail = response.text[:220]
                    raise BackendUnavailable(
                        f"Gemini API error {response.status_code}. {detail}".strip()
                    )

            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(backoff, 30.0))

        if hit_rate_limit:
            raise BackendUnavailable(
                "The Gemini API is rate limiting this key. Wait a moment and try again, "
                "or check your quota at aistudio.google.com."
            )
        raise BackendUnavailable(
            f"Could not reach the Gemini API after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    @staticmethod
    def _text_of(response: dict[str, Any]) -> str:
        """Pull the generated text out of a response, or explain its absence."""
        candidates = response.get("candidates") or []
        if not candidates:
            feedback = response.get("promptFeedback") or {}
            reason = feedback.get("blockReason")
            if reason:
                raise BackendUnavailable(f"Gemini declined the request ({reason}).")
            return ""

        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(str(part.get("text", "")) for part in parts).strip()

    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        file_path: str | Path | None = None,
        json_output: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> str:
        parts: list[dict[str, Any]] = [{"text": user}]
        if file_path is not None:
            parts.append(_inline_part(file_path))

        config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_output:
            config["responseMimeType"] = "application/json"

        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
        }
        return self._text_of(self._post(model or self._settings.model, body))

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("json_output", True)
        return extract_json_object(self.generate(**kwargs))
