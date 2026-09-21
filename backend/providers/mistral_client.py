"""Thin, dependency-light client for the Mistral API.

Only the three surfaces OmniAI needs are implemented: chat completion, chat
completion with an image attached, and the dedicated OCR endpoint. Retries use
exponential backoff and are limited to the status codes where retrying is
actually correct (429 and 5xx).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

from backend.config import MistralSettings
from backend.providers.base import BackendUnavailable
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.mistral")

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4

# Rate limiting is a first-class condition on a free tier, not an edge case.
# A 429 is retried more patiently than a server error, and the whole process
# slows down for a while afterwards rather than charging back in.
_RATE_LIMIT_BACKOFF = (2.0, 6.0, 15.0)
_COOLDOWN_SECONDS = 20.0
_MIN_INTERVAL_NORMAL = 0.0
_MIN_INTERVAL_COOLING = 1.2


class _Throttle:
    """Process-wide pacing, shared by every client instance.

    After a 429 the whole application spaces its requests out for a cooldown
    window. Without this, several in-flight requests each retry independently
    and turn one rate-limit response into a burst of them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._cooling_until = 0.0

    @property
    def cooling(self) -> bool:
        return time.monotonic() < self._cooling_until

    def note_rate_limit(self, retry_after: float | None) -> None:
        with self._lock:
            wait = retry_after if retry_after is not None else _COOLDOWN_SECONDS
            self._cooling_until = max(self._cooling_until, time.monotonic() + wait)

    def wait_turn(self) -> None:
        """Block until this caller may issue its request."""
        while True:
            with self._lock:
                now = time.monotonic()
                interval = _MIN_INTERVAL_COOLING if now < self._cooling_until else _MIN_INTERVAL_NORMAL
                if now >= self._next_allowed:
                    self._next_allowed = now + interval
                    return
                sleep_for = self._next_allowed - now
            time.sleep(min(sleep_for, 5.0))


_throttle = _Throttle()


def rate_limited() -> bool:
    """True while the process is pacing itself after a rate-limit response."""
    return _throttle.cooling


def _retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None

# Mistral accepts these for inline image payloads.
_IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".avif": "image/avif",
}


def encode_data_uri(path: str | Path) -> str:
    """Read a local file and return it as a base64 ``data:`` URI."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    mime = _IMAGE_MIME.get(suffix) or mimetypes.guess_type(str(file_path))[0]
    if mime is None:
        mime = "application/octet-stream"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models occasionally wrap JSON in prose or a fenced block even when asked
    not to, so fall back to the first balanced ``{...}`` span before giving up.
    """
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


class MistralClient:
    def __init__(self, settings: MistralSettings) -> None:
        self._settings = settings
        self._session = requests.Session()

    @property
    def settings(self) -> MistralSettings:
        return self._settings

    def _headers(self) -> dict[str, str]:
        if not self._settings.api_key:
            raise BackendUnavailable(
                "No Mistral API key is configured. Add OMNIAI_MISTRAL_API_KEY to "
                "your .env file, or switch to the 'paper' profile to run the paper's models locally."
            )
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = self._headers()
        last_error: Exception | None = None

        hit_rate_limit = False

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            _throttle.wait_turn()
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._settings.timeout_seconds,
                )
            except requests.RequestException as error:
                last_error = error
                logger.warning("Mistral request failed (attempt %s): %s", attempt, error)
                backoff = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            else:
                if response.status_code < 400:
                    return response.json()

                if response.status_code == 429:
                    hit_rate_limit = True
                    retry_after = _retry_after_seconds(response)
                    _throttle.note_rate_limit(retry_after)
                    index = min(attempt - 1, len(_RATE_LIMIT_BACKOFF) - 1)
                    backoff = retry_after if retry_after is not None else _RATE_LIMIT_BACKOFF[index]
                    last_error = RuntimeError("rate limited")
                    logger.warning(
                        "Mistral rate limit hit; waiting %.1fs before attempt %s",
                        backoff,
                        attempt + 1,
                    )
                elif response.status_code in _RETRY_STATUS:
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                    backoff = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
                    logger.warning(
                        "Mistral returned %s, retrying (attempt %s)",
                        response.status_code,
                        attempt,
                    )
                else:
                    # Surface the provider's own message, never the API key.
                    try:
                        detail = str(response.json().get("message", ""))[:200]
                    except Exception:
                        detail = response.text[:200]
                    raise BackendUnavailable(
                        f"Mistral API error {response.status_code}. {detail}".strip()
                    )

            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(backoff, 30.0))

        if hit_rate_limit:
            raise BackendUnavailable(
                "The Mistral API is rate limiting this key. Free-tier keys allow only "
                "a few requests per minute. Wait a moment and try again, set "
                "OMNIAI_ENSEMBLE=false to halve the number of calls per image, or "
                "upgrade the key at console.mistral.ai."
            )

        raise BackendUnavailable(
            f"Could not reach the Mistral API after {_MAX_ATTEMPTS} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Public surfaces
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        image_path: str | Path | None = None,
        json_output: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """Run a chat completion, optionally with one image attached."""
        content: Any
        if image_path is not None:
            content = [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": encode_data_uri(image_path)},
            ]
            chosen_model = model or self._settings.vision_model
        else:
            content = user
            chosen_model = model or self._settings.text_model

        payload: dict[str, Any] = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}

        data = self._post("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message", {}).get("content") or "").strip()

    def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        """Chat completion constrained to a JSON object, already parsed."""
        kwargs.setdefault("json_output", True)
        return extract_json_object(self.chat(**kwargs))

    def ocr(self, file_path: str | Path, *, is_pdf: bool = False) -> dict[str, Any]:
        """Run the dedicated OCR model over an image or PDF."""
        document = (
            {"type": "document_url", "document_url": encode_data_uri(file_path)}
            if is_pdf
            else {"type": "image_url", "image_url": encode_data_uri(file_path)}
        )
        return self._post(
            "ocr",
            {
                "model": self._settings.ocr_model,
                "document": document,
                "include_image_base64": False,
            },
        )
