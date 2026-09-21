"""Abstractive summarisation, with an extractive fallback.

This module previously carried its own hand-rolled ``.env`` parser and its own
HTTP call to Mistral, duplicating logic that now lives in
:mod:`backend.config` and :mod:`backend.providers.mistral_client`. It reads
configuration from one place and delegates transport to one client.

When no API key is available the fallback is extractive rather than absent, so
document analysis still produces something useful offline.
"""

from __future__ import annotations

import re

from backend.config import get_settings
from backend.providers.base import BackendUnavailable
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.summary")

MIN_SUMMARISABLE_CHARS = 100
MAX_SOURCE_CHARS = 12_000

_SYSTEM_PROMPT = (
    "You are an expert summariser. Write one paragraph that captures the key "
    "points, main ideas and important details. Be thorough but do not repeat "
    "yourself. Plain prose only: no markdown, no heading, no 'Summary:' prefix."
)

# Lines that are page furniture rather than content.
_BOILERPLATE = re.compile(
    r"^(ISBN|Copyright|Page \d+|Printed in|www\.|https?://|Contents|Chapter \d+)",
    re.I,
)
_CREDITS = re.compile(r"(thank you|acknowledg[e]?ment|dedication|about the author)", re.I)


def initialize_summarizer() -> None:
    """Report summarisation readiness at startup."""
    settings = get_settings()
    if settings.mistral.configured:
        logger.info("Summarisation: %s", settings.mistral.text_model)
    else:
        logger.warning(
            "No Mistral API key, so summaries will be extractive rather than abstractive."
        )


def _clean_text(text: str) -> str:
    """Strip page furniture so the model sees content, not headers."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    kept = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if len(line) < 8:
            continue
        if len(re.findall(r"[A-Za-z0-9]", line)) < 4:
            continue
        if _BOILERPLATE.match(line) or _CREDITS.search(line):
            continue
        kept.append(line)

    return " ".join(kept)


def build_extractive_summary(text: str, max_chars: int = 420) -> str:
    """Return the opening sentences, as a fallback when no model is available."""
    normalised = re.sub(r"\s+", " ", text or "").strip()
    if not normalised:
        return "No summary could be generated."

    selected: list[str] = []
    length = 0

    for sentence in re.split(r"(?<=[.!?])\s+", normalised):
        sentence = sentence.strip()
        if not sentence:
            continue
        projected = length + len(sentence) + (1 if selected else 0)
        if projected > max_chars and selected:
            break
        selected.append(sentence)
        length = projected
        if len(selected) >= 3:
            break

    if selected:
        return " ".join(selected)
    return " ".join(normalised.split()[:80])


def _sanitize(summary: str) -> str:
    """Strip markdown and lead-ins that slip through despite the prompt."""
    cleaned = summary.strip()
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:summary|brief summary|in summary)\s*[:\-–—]*\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.M)
    return cleaned.strip()


def generate_summary(text: str, max_length: int = 150) -> str:
    """Summarise a document, preferring the hosted model."""
    if not text or len(text.strip()) < MIN_SUMMARISABLE_CHARS:
        return "The document is too short to summarise."

    cleaned = _clean_text(text)
    if len(cleaned.strip()) < MIN_SUMMARISABLE_CHARS:
        fallback_source = re.sub(r"\s+", " ", text).strip()
        if len(fallback_source) < MIN_SUMMARISABLE_CHARS:
            return "No meaningful content was found to summarise."
        cleaned = fallback_source

    settings = get_settings()
    if not settings.mistral.configured:
        return build_extractive_summary(cleaned)

    from backend.providers.mistral_client import MistralClient

    try:
        summary = MistralClient(settings.mistral).chat(
            system=_SYSTEM_PROMPT,
            user=f"Summarise the following text.\n\n{cleaned[:MAX_SOURCE_CHARS]}",
            temperature=0.3,
            max_tokens=max(150, min(max_length, 500)),
        )
    except (BackendUnavailable, Exception) as error:  # noqa: B014
        logger.error("Summarisation failed, falling back to extractive: %s", error)
        return build_extractive_summary(cleaned)

    return _sanitize(summary) or build_extractive_summary(cleaned)
