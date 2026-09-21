"""Request-level protections applied before any model runs.

The original app accepted uploads with no rate limit, no magic-byte check and
no pixel budget, which left three cheap denial-of-service routes open: flood
the endpoint, send a renamed executable, or send a 60-megapixel PNG that
expands to gigabytes in memory.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import BinaryIO

from backend.config import get_settings

# Magic bytes for the formats we accept. An extension alone proves nothing.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".webp": (b"RIFF",),
    ".pdf": (b"%PDF-",),
    # DOCX is a ZIP container.
    ".docx": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
}

ALLOWED_EXTENSIONS = frozenset(_SIGNATURES)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class UploadRejected(ValueError):
    """The upload failed validation. The message is safe to show the user."""


class RateLimitExceeded(RuntimeError):
    def __init__(self, retry_after: int) -> None:
        super().__init__("Too many requests. Please wait a moment and try again.")
        self.retry_after = retry_after


class SlidingWindowRateLimiter:
    """Per-client sliding window, held in memory.

    Adequate for a single-process deployment. A multi-process or multi-replica
    deployment should move this to Redis; see docs/deployment.md.
    """

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_id: str) -> None:
        if self._max <= 0:
            return

        now = time.monotonic()
        cutoff = now - self._window

        with self._lock:
            hits = self._hits[client_id]
            while hits and hits[0] < cutoff:
                hits.popleft()

            if len(hits) >= self._max:
                retry_after = max(1, int(self._window - (now - hits[0])) + 1)
                raise RateLimitExceeded(retry_after)

            hits.append(now)

            # Keep the dict from growing without bound on a long-lived process:
            # drop every client whose most recent request has left the window.
            if len(self._hits) > 4096:
                for key in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                    del self._hits[key]


def validate_extension(filename: str | None) -> str:
    """Return the normalised extension, or reject the upload."""
    name = (filename or "").strip()
    if not name:
        raise UploadRejected("A filename is required.")

    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadRejected(f"Unsupported file type. Allowed formats: {allowed}")
    return extension


def validate_signature(stream: BinaryIO, extension: str) -> None:
    """Confirm the bytes match the claimed extension, then rewind."""
    position = stream.tell()
    header = stream.read(8)
    stream.seek(position)

    expected = _SIGNATURES.get(extension, ())
    if expected and not any(header.startswith(signature) for signature in expected):
        raise UploadRejected(
            f"This file does not look like a valid {extension} file. "
            "It may be corrupted or renamed."
        )


def validate_size(stream: BinaryIO) -> int:
    """Confirm the upload is within the configured budget, then rewind."""
    limit = get_settings().limits.max_upload_bytes
    position = stream.tell()
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(position)

    if size <= 0:
        raise UploadRejected("The uploaded file is empty.")

    if size > limit:
        raise UploadRejected(
            f"File is too large. The maximum is {limit // (1024 * 1024)} MB."
        )
    return size


def validate_image_dimensions(image_path: str | Path) -> None:
    """Reject decompression bombs before any model touches the pixels.

    A 30,000 x 30,000 PNG is a few hundred kilobytes on disk and several
    gigabytes once decoded.
    """
    from PIL import Image

    limit = get_settings().limits.max_image_pixels

    # Raise Pillow's own guard above ours so we produce the readable error.
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except UploadRejected:
        raise
    except Exception as error:
        raise UploadRejected(
            "The image could not be read. It may be damaged or in an unsupported format."
        ) from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous

    if width * height > limit:
        raise UploadRejected(
            f"Image is too large to process ({width} x {height} pixels). "
            f"The maximum is {limit // 1_000_000} megapixels."
        )


def validate_text(text: str) -> str:
    """Trim and bound a text submission."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise UploadRejected("Enter some text to analyse.")

    limit = get_settings().limits.max_text_chars
    if len(cleaned) > limit:
        raise UploadRejected(
            f"Text is too long ({len(cleaned):,} characters). The maximum is {limit:,}."
        )
    return cleaned


def client_identifier(
    remote_addr: str | None, forwarded_for: str | None, trusted_proxies: int = 0
) -> str:
    """Identify the caller for rate limiting.

    Each proxy appends the address it received the request from, so only the
    entries added by proxies we run behind can be believed; anything to their
    left was written by the client and may be invented. With N trusted
    proxies the caller is the Nth entry from the right. With none, the
    header is ignored entirely.
    """
    if trusted_proxies > 0 and forwarded_for:
        entries = [entry.strip() for entry in forwarded_for.split(",") if entry.strip()]
        if len(entries) >= trusted_proxies:
            return entries[-trusted_proxies]
    return remote_addr or "unknown"


class ServerBusy(RuntimeError):
    """Every analysis slot is taken. The message is safe to show the user."""

    retry_after = 20

    def __init__(self) -> None:
        super().__init__(
            "The server is busy with other analyses. Please try again in a few seconds."
        )


class AnalysisSlots:
    """Bounds how many analyses run at once.

    Each analysis keeps the CPU busy for seconds. Without a bound, a burst of
    requests would all run at once and every one of them would slow to a
    crawl; with one, the excess are told to retry instead.
    """

    def __init__(self, limit: int, wait_seconds: float = 3.0) -> None:
        self._slots = threading.BoundedSemaphore(max(1, limit))
        self._wait = wait_seconds

    def acquire(self) -> None:
        if not self._slots.acquire(timeout=self._wait):
            raise ServerBusy()

    def release(self) -> None:
        self._slots.release()
