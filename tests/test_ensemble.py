"""The two-model ensemble shared by the Mistral and Gemini backends."""

from __future__ import annotations

import pytest

from backend.providers.base import BackendUnavailable
from backend.providers.ensemble import labels_agree, run_ensemble


def _scripted(answers: dict[str, object]):
    """A classify_once that answers per model, or raises if given an exception."""
    calls: list[str] = []

    def classify_once(model: str):
        calls.append(model)
        answer = answers[model]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)

    return classify_once, calls


def _run(answers, adjudicate=None):
    classify_once, calls = _scripted(answers)
    adjudications: list[tuple] = []

    def default_adjudicate(first, second):
        adjudications.append((first["label"], second["label"]))
        return {"label": "resolved", "confidence": 77.0, "reason": "looked closely"}

    result = run_ensemble(
        classify_once=classify_once,
        adjudicate=adjudicate or default_adjudicate,
        primary_model="primary",
        second_model="second",
        adjudicator_model="judge",
        timeout_seconds=5,
    )
    return result, calls, adjudications


class TestLabelAgreement:
    @pytest.mark.parametrize("a,b", [
        ("african elephant", "elephant"),
        ("elephant", "African Elephant"),
        ("a photo of a lion", "lion"),
        ("golden retriever", "golden retriever"),
    ])
    def test_agreeing_labels(self, a, b):
        assert labels_agree(a, b)

    @pytest.mark.parametrize("a,b", [
        ("lion", "tiger"),
        ("laptop", "desk"),
        ("", "lion"),
        ("the photo", "lion"),   # only noise words on one side
    ])
    def test_disagreeing_labels(self, a, b):
        assert not labels_agree(a, b)


class TestRunEnsemble:
    def test_agreement_raises_confidence_and_keeps_specific_label(self):
        result, calls, adjudications = _run({
            "primary": {"label": "elephant", "confidence": 80.0},
            "second": {"label": "african elephant", "confidence": 70.0},
        })
        assert result["label"] == "african elephant", "the more specific label wins"
        assert result["confidence"] == 84.0
        assert result["ensemble"]["agreement"] == "agreed"
        assert not adjudications, "no adjudication when the models agree"
        assert sorted(calls) == ["primary", "second"]

    def test_agreement_never_reads_as_certain(self):
        result, _, _ = _run({
            "primary": {"label": "lion", "confidence": 98.0},
            "second": {"label": "lion", "confidence": 97.0},
        })
        assert result["confidence"] <= 99.0

    def test_disagreement_goes_to_the_adjudicator(self):
        result, _, adjudications = _run({
            "primary": {"label": "lion", "confidence": 60.0},
            "second": {"label": "tiger", "confidence": 55.0},
        })
        assert adjudications == [("lion", "tiger")]
        assert result["label"] == "resolved"
        assert result["ensemble"]["resolved_by"] == "judge"
        assert result["ensemble"]["reasoning"] == "looked closely"

    def test_adjudicator_failure_keeps_the_primary_answer(self):
        def broken(first, second):
            raise BackendUnavailable("rate limited")

        result, _, _ = _run(
            {"primary": {"label": "lion", "confidence": 60.0},
             "second": {"label": "tiger", "confidence": 55.0}},
            adjudicate=broken,
        )
        assert result["label"] == "lion"
        assert "adjudicator unavailable" in result["ensemble"]["resolved_by"]

    def test_second_model_failure_degrades_to_single(self):
        result, _, _ = _run({
            "primary": {"label": "lion", "confidence": 60.0},
            "second": BackendUnavailable("down"),
        })
        assert result["label"] == "lion"
        assert result["ensemble"]["agreement"] == "single"

    def test_primary_failure_falls_back_to_second(self):
        result, _, _ = _run({
            "primary": BackendUnavailable("down"),
            "second": {"label": "tiger", "confidence": 55.0},
        })
        assert result["label"] == "tiger"
        assert result["ensemble"]["agreement"] == "primary_failed"

    def test_both_failing_raises_the_primary_error(self):
        with pytest.raises(BackendUnavailable, match="primary down"):
            _run({
                "primary": BackendUnavailable("primary down"),
                "second": BackendUnavailable("second down"),
            })
