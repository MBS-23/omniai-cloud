"""Scrubbing of secrets from anything that leaves the process.

The application already avoids putting credentials into messages, but "we are
careful" is not a control. Every log line and every error string shown to a
user passes through here, so a key cannot reach a terminal, a log file, a
browser or a bug report even if some upstream message embeds one.
"""

from __future__ import annotations

import logging
import re
from typing import Any

MASK = "[redacted]"

# Shapes worth catching regardless of where they came from.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bearer tokens in a header echo
    re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", re.I),
    # key=value and JSON "key": "value" forms. The optional quote after the
    # key name is what makes the JSON case match.
    re.compile(
        r"((?:api[_-]?key|apikey|secret|token|password|passwd|authorization)"
        r"[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9._\-]{6,})",
        re.I,
    ),
    # Common vendor prefixes
    re.compile(r"\b(sk-)[A-Za-z0-9]{12,}\b"),
    re.compile(r"\b(gh[pousr]_)[A-Za-z0-9]{20,}\b"),
    # A bare JWT
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),
)


def _literal_secrets() -> list[str]:
    """Exact secret values from configuration, so they are caught verbatim.

    Read lazily: settings import this module's logger, so importing settings at
    module scope would be circular.
    """
    try:
        from backend.config import get_settings

        key = get_settings().mistral.api_key
    except Exception:
        return []
    # Very short values are placeholders, and masking them would corrupt
    # ordinary prose.
    return [key] if key and len(key) >= 8 else []


def redact(value: Any) -> str:
    """Return ``value`` as a string with anything secret-shaped masked."""
    text = value if isinstance(value, str) else str(value)

    for secret in _literal_secrets():
        text = text.replace(secret, MASK)

    for pattern in _PATTERNS:
        if pattern.groups >= 2 or pattern.groups == 1:
            text = pattern.sub(lambda m: f"{m.group(1)}{MASK}", text)
        else:
            text = pattern.sub(MASK, text)

    return text


class RedactingFormatter(logging.Formatter):
    """A formatter that scrubs the rendered line.

    This deliberately works on the formatted output rather than mutating
    ``record.args``. Rewriting the arguments changes their types -- logging a
    dict with ``%s`` would turn every value into a string -- and it misses
    anything the format string itself contributes.
    """

    def format(self, record: logging.LogRecord) -> str:
        try:
            return redact(super().format(record))
        except Exception:
            # Logging must never raise. Fall back to the unformatted message,
            # still scrubbed.
            try:
                return redact(record.getMessage())
            except Exception:
                return "[log record could not be formatted]"
