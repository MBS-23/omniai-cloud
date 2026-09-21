"""Two-model classification with an adjudicator, shared by hosted providers.

The idea is the same one the paper applies to label refinement: rather than
fine-tuning a model, combine independent opinions. Two vision models classify
the image concurrently. When they agree the answer is corroborated; when they
disagree a stronger reasoning model looks at the image and decides.

This module owns only the orchestration. Each provider supplies how to make a
single classification call and how to adjudicate, so the Mistral and Gemini
backends cannot drift apart in how they reconcile answers.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

from backend.providers.base import BackendUnavailable, Classification
from backend.utils.logger import get_logger

logger = get_logger("OmniAI.ensemble")

# Confidence added when two independent models agree, capped so a pair that
# agrees on the wrong answer is never presented as certain.
AGREEMENT_BONUS = 4.0
AGREEMENT_CEILING = 99.0

_NOISE = frozenset({
    "a", "an", "the", "photo", "picture", "image", "of", "close", "up",
    "closeup", "shot", "view",
})


def label_tokens(label: str) -> set[str]:
    return {
        word
        for word in re.split(r"[^a-z0-9]+", (label or "").lower())
        if word and word not in _NOISE
    }


def labels_agree(first: str, second: str) -> bool:
    """True when two model labels name the same thing.

    Exact equality is too strict: "african elephant" and "elephant" plainly
    agree. One label containing all of the other's meaningful words counts as
    agreement, which also makes the more specific of the two the natural winner.
    """
    a, b = label_tokens(first), label_tokens(second)
    if not a or not b:
        return False
    return a.issubset(b) or b.issubset(a)


def run_ensemble(
    *,
    classify_once: Callable[[str], Classification],
    adjudicate: Callable[[Classification, Classification], dict[str, Any]],
    primary_model: str,
    second_model: str,
    adjudicator_model: str,
    timeout_seconds: float,
) -> Classification:
    """Classify with two models, reconciling them if they disagree.

    Args:
        classify_once: Makes one classification call with the given model.
        adjudicate: Given both answers, returns ``{"label", "confidence",
            "reason"}`` from a reasoning model. Only called on disagreement.
        primary_model / second_model / adjudicator_model: Model identifiers,
            recorded in the result so the interface can show its working.
        timeout_seconds: Upper bound on waiting for the concurrent calls.
    """
    results: dict[str, Any] = {}

    def call(slot: str, model: str) -> None:
        try:
            results[slot] = classify_once(model)
        except Exception as error:  # recorded and inspected below, not swallowed
            results[slot] = error

    threads = [
        threading.Thread(target=call, args=("primary", primary_model), daemon=True),
        threading.Thread(target=call, args=("second", second_model), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout_seconds)

    primary = results.get("primary")
    second = results.get("second")

    # Nothing to build on without the primary answer.
    if not isinstance(primary, dict):
        if isinstance(second, dict):
            second = dict(second)
            second["ensemble"] = {"agreement": "primary_failed", "models": [second_model]}
            return second  # type: ignore[return-value]
        if isinstance(primary, Exception):
            raise primary
        raise BackendUnavailable("Image classification did not return a result.")

    if not isinstance(second, dict):
        primary = dict(primary)
        primary["ensemble"] = {"agreement": "single", "models": [primary_model]}
        return primary  # type: ignore[return-value]

    if labels_agree(primary["label"], second["label"]):
        specific = max((primary, second), key=lambda r: len(r["label"]))
        agreed = dict(specific)
        agreed["confidence"] = round(
            min(AGREEMENT_CEILING, max(primary["confidence"], second["confidence"]) + AGREEMENT_BONUS),
            2,
        )
        agreed["ensemble"] = {
            "agreement": "agreed",
            "models": [primary_model, second_model],
            "labels": [primary["label"], second["label"]],
        }
        return agreed  # type: ignore[return-value]

    # Disagreement: a reasoning model decides.
    try:
        verdict = adjudicate(primary, second)
    except Exception as error:
        logger.warning("Adjudication unavailable, keeping the primary answer: %s", error)
        kept = dict(primary)
        kept["ensemble"] = {
            "agreement": "disagreed",
            "models": [primary_model, second_model],
            "labels": [primary["label"], second["label"]],
            "resolved_by": "primary (adjudicator unavailable)",
        }
        return kept  # type: ignore[return-value]

    label = str(verdict.get("label") or "").strip().lower() or primary["label"]
    return {  # type: ignore[return-value]
        "label": label,
        "confidence": verdict.get("confidence", primary["confidence"]),
        "ensemble": {
            "agreement": "disagreed",
            "models": [primary_model, second_model],
            "labels": [primary["label"], second["label"]],
            "resolved_by": adjudicator_model,
            "reasoning": (str(verdict.get("reason") or "")[:300] or None),
        },
    }
