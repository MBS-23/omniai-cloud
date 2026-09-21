"""Temporary storage for uploads.

Uploads are written under a per-process directory with a random name, so a
crafted filename cannot escape it and two concurrent uploads of the same file
cannot collide.
"""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path
from typing import BinaryIO

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.files")

# Anything outside this set is replaced, so the stored name can never contain
# a path separator, a drive letter or a parent reference.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

MAX_STORED_NAME_LENGTH = 80


def upload_dir() -> Path:
    directory = get_settings().upload_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_name(filename: str | None) -> str:
    """Reduce a client-supplied filename to something safe to write."""
    # Path(...).name strips any directory component the client sent.
    base = Path(filename or "").name
    cleaned = _UNSAFE.sub("_", base).strip("._")

    if not cleaned:
        raise ValueError("Invalid filename")

    stem = Path(cleaned).stem[:MAX_STORED_NAME_LENGTH]
    suffix = Path(cleaned).suffix[:16]
    return f"{stem}{suffix}"


def save_uploaded_file(filename: str | None, stream: BinaryIO) -> tuple[str, Path]:
    """Persist an upload and return ``(stored_name, path)``."""
    stored_name = f"{uuid.uuid4().hex}_{safe_name(filename)}"
    destination = upload_dir() / stored_name

    stream.seek(0)
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(stream, buffer, length=1024 * 1024)

    return stored_name, destination


def delete_file(file_path: Path | str | None) -> None:
    """Remove a temporary file. Failures are logged, never raised."""
    if file_path is None:
        return
    try:
        path = Path(file_path)
        if path.is_file():
            path.unlink()
    except OSError as error:
        logger.warning("Could not delete %s: %s", file_path, error)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def purge_stale_uploads(max_age_seconds: int = 3600) -> int:
    """Delete leftovers from crashed requests.

    The per-request ``finally`` block is the primary cleanup; this is the
    backstop for a process killed mid-request. Called once at startup.
    """
    directory = get_settings().upload_dir
    if not directory.is_dir():
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0

    for entry in directory.iterdir():
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
            removed += 1
        except OSError:
            continue

    if removed:
        logger.info("Purged %s stale upload(s)", removed)
    return removed
