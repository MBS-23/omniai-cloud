"""Logging setup.

Handlers are attached once, to the root ``OmniAI`` logger only. The previous
version attached a file and a stream handler to every named logger, and since
those also propagate to their parent, each message was emitted twice per level
of nesting.

Output goes to stdout always, and to a file only when a log directory is
writable, which keeps container logs clean.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_NAME = "OmniAI"
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"

_configured = False


def _log_dir() -> Path | None:
    # Imported lazily: config imports nothing from here, but this keeps the
    # dependency one-directional even if that changes.
    from backend.config import get_settings

    try:
        directory = get_settings().log_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    except OSError:
        return None


def _configure_root() -> logging.Logger:
    global _configured

    root = logging.getLogger(ROOT_NAME)
    if _configured:
        return root

    root.setLevel(logging.INFO)
    root.propagate = False

    # Every line is scrubbed as it is rendered, so a credential cannot reach
    # stdout or the log file even if some message embeds one.
    from backend.utils.redaction import RedactingFormatter

    formatter = RedactingFormatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    directory = _log_dir()
    if directory is not None:
        try:
            file_handler = logging.FileHandler(directory / "omniai.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # Read-only filesystem, which is normal in a hardened container.
            pass

    _configured = True
    return root


def get_logger(name: str = ROOT_NAME) -> logging.Logger:
    """Return a logger that inherits the single configured handler set."""
    _configure_root()

    if name == ROOT_NAME:
        return logging.getLogger(ROOT_NAME)
    if not name.startswith(f"{ROOT_NAME}."):
        name = f"{ROOT_NAME}.{name}"
    return logging.getLogger(name)
