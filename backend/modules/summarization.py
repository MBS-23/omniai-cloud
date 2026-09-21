"""Document summarisation with T5 (conference paper section 3C, ref [9]).

``google-t5/t5-base``, running locally. The thesis summarised through the
Mistral API; the conference paper names T5, and T5 is what keeps the paper
profile free of API keys and rate limits. Mistral remains available by setting
``OMNIAI_SUMMARIZER=mistral`` with a key.

T5 reads at most 512 tokens, and a document is usually far longer. The text is
cut into pieces at sentence boundaries, a spread of pieces from beginning,
middle and end is summarised, and if those summaries are themselves long they
are summarised once more. The number of pieces is capped so a 200-page PDF
costs the same as a ten-page one.
"""

from __future__ import annotations

import re
from typing import Any

from backend.config import get_settings
from backend.modules.hub import PAPER_MODELS
from backend.modules.lazy import Lazy
from backend.providers.base import BackendUnavailable
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.summary")

MIN_SUMMARISABLE_CHARS = 100
MAX_SOURCE_CHARS = 12_000
PIECE_TOKENS = 480
MAX_PIECES = 4

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
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _build() -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from backend.utils.cpu import quantize_for_cpu

    repo = PAPER_MODELS["summarize"].repo_id
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    return tokenizer, quantize_for_cpu(model.eval())


MODEL = Lazy("summarize", "T5 (base)", _build)


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
    """The opening sentences: the last resort when no model can run."""
    normalised = re.sub(r"\s+", " ", text or "").strip()
    if not normalised:
        return "No summary could be generated."

    selected: list[str] = []
    length = 0
    for sentence in _SENTENCES.split(normalised):
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

    return " ".join(selected) if selected else " ".join(normalised.split()[:80])


def _sanitize(summary: str) -> str:
    """Strip markdown and lead-ins a hosted model adds despite the prompt."""
    cleaned = summary.strip()
    cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\s*(?:summary|brief summary|in summary)\s*[:\-]*\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.M)
    return cleaned.strip()


def _tidy(summary: str) -> str:
    """Clean T5's output: "text . it inspects" becomes "text. It inspects"."""
    summary = re.sub(r"\s+([.,;:!?])", r"\1", summary.strip())
    sentences = [s.strip() for s in _SENTENCES.split(summary) if s.strip()]
    return " ".join(s[0].upper() + s[1:] for s in sentences)


def _pieces(text: str, tokenizer: Any) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in _SENTENCES.split(text):
        length = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        if current and size + length > PIECE_TOKENS:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += length
    if current:
        pieces.append(" ".join(current))
    return pieces


def _spread(pieces: list[str], count: int) -> list[str]:
    """``count`` pieces spaced evenly from first to last, in document order."""
    if len(pieces) <= count:
        return pieces
    step = (len(pieces) - 1) / (count - 1)
    return [pieces[round(i * step)] for i in range(count)]


def _t5(texts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    tokenizer, model = MODEL.get()
    inputs = tokenizer(
        [f"summarize: {text}" for text in texts],
        return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(40, max_new_tokens),
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return [item.strip() for item in tokenizer.batch_decode(output, skip_special_tokens=True)]


def summarize_t5(text: str, max_length: int = 200) -> str:
    tokenizer, _ = MODEL.get()
    pieces = _spread(_pieces(text, tokenizer), MAX_PIECES)
    budget = min(max(80, max_length), 250)

    partial = _t5(pieces, max_new_tokens=budget if len(pieces) == 1 else 120)
    summary = " ".join(partial)
    if len(pieces) > 1 and len(tokenizer(summary)["input_ids"]) > budget:
        summary = _t5([summary], max_new_tokens=budget)[0]
    return _tidy(summary)


def _summarize_mistral(text: str, max_length: int) -> str:
    from backend.providers.mistral_client import MistralClient

    summary = MistralClient(get_settings().mistral).chat(
        system=_SYSTEM_PROMPT,
        user=f"Summarise the following text.\n\n{text[:MAX_SOURCE_CHARS]}",
        temperature=0.3,
        max_tokens=max(150, min(max_length, 500)),
    )
    return _sanitize(summary)


def generate_summary(text: str, max_length: int = 200) -> str:
    """Summarise a document with T5, or with Mistral when configured to."""
    if not text or len(text.strip()) < MIN_SUMMARISABLE_CHARS:
        return "The document is too short to summarise."

    cleaned = _clean_text(text)
    if len(cleaned.strip()) < MIN_SUMMARISABLE_CHARS:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) < MIN_SUMMARISABLE_CHARS:
            return "No meaningful content was found to summarise."

    settings = get_settings()
    if settings.summarizer == "mistral" and settings.mistral.configured:
        try:
            return _summarize_mistral(cleaned, max_length) or summarize_t5(cleaned, max_length)
        except BackendUnavailable as error:
            logger.warning("Mistral summarisation failed, using T5 instead: %s", error)

    return summarize_t5(cleaned, max_length)
