"""The paper profile's logic, tested without loading a single model.

The models themselves are exercised by ``python -m backend.models.verify_paper``
in the Docker build. These tests cover what sits around them: that
importing the profile stays cheap, how OCR chooses between language packs,
how long text is cut up for NLLB and T5, and what the health endpoint reports.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.models import fetch_paper
from backend.modules import ocr_paddle, summarization, translation_ai
from backend.modules.hub import PADDLE_OCR_OPTIONS, PAPER_MODELS, model_label
from backend.providers.base import BackendUnavailable
from backend.utils import cpu


def test_importing_the_paper_profile_loads_no_ml_library():
    """Heavy imports happen on first use, not at import.

    This keeps the lite profile (no PyTorch at all) importable, and keeps a
    paper-profile server answering health checks while models load.
    """
    code = (
        "import sys\n"
        "import backend.providers.paper, backend.models.fetch_paper, backend.models.verify_paper\n"
        "import backend.modules.image_classification, backend.modules.object_detection\n"
        "import backend.modules.wildlife_detection, backend.modules.ocr_paddle\n"
        "import backend.modules.blip_vision_language, backend.modules.translation_ai\n"
        "import backend.modules.text_analytics, backend.modules.summarization\n"
        "heavy = {'torch', 'transformers', 'ultralytics', 'paddleocr', 'paddle', 'spacy'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert output == "[]"


class TestModelInventory:
    def test_every_paper_model_names_its_hub_repository(self):
        assert set(PAPER_MODELS) == {
            "classify", "detect", "wildlife", "ocr", "caption",
            "translate", "sentiment", "ner", "summarize",
        }
        for model in PAPER_MODELS.values():
            assert "/" in model.repo_id, model

    def test_the_two_detectors_are_the_papers_yolov8_pair(self):
        assert PAPER_MODELS["detect"].filename == "yolov8n.pt"
        assert PAPER_MODELS["wildlife"].filename == "yolov8x.pt"

    def test_labels_name_both_the_paper_model_and_its_source(self):
        assert model_label("classify") == "ResNet50 (timm/resnet50.tv2_in1k)"


class TestOcrLanguageSearch:
    """Which language pack's reading wins, and when the search stops."""

    @staticmethod
    def _reading(lang, lines=(), confidence=0.9, regions=None):
        reading = ocr_paddle._Reading(lang=lang)
        reading.lines = list(lines)
        reading.confidences = [confidence] * len(lines)
        reading.regions = len(lines) if regions is None else regions
        return reading

    def _run(self, monkeypatch, readings, langs):
        calls = []

        def fake_read(lang, _path):
            calls.append(lang)
            return readings[lang]

        monkeypatch.setattr(ocr_paddle, "_read", fake_read)
        return ocr_paddle.extract_text_paddle("page.png", langs), calls

    def test_stops_after_a_confident_first_reading(self, monkeypatch):
        readings = {"en": self._reading("en", ["Invoice 42", "Total due"], 0.95)}
        result, calls = self._run(monkeypatch, readings, ("en", "te", "hi"))
        assert calls == ["en"]
        assert result["text"] == "Invoice 42\nTotal due"
        assert result["language_hint"] == "Latin (English)"

    def test_stops_when_no_text_regions_exist(self, monkeypatch):
        """A photo with no text: other recognisers cannot find any either."""
        readings = {"en": self._reading("en", [], regions=0)}
        result, calls = self._run(monkeypatch, readings, ("en", "te", "hi"))
        assert calls == ["en"]
        assert result["has_text"] is False

    def test_devanagari_skips_line_orientation(self):
        """The orientation model flips Devanagari lines upside down."""
        assert ocr_paddle.engine_options("hi")["use_textline_orientation"] is False
        assert ocr_paddle.engine_options("en")["use_textline_orientation"] is True
        assert ocr_paddle.engine_options("te")["use_textline_orientation"] is True

    def test_keeps_searching_when_unsure_and_picks_the_best(self, monkeypatch):
        readings = {
            "en": self._reading("en", ["x0 7l"], 0.55),
            "te": self._reading("te", ["తెలుగు పాఠం"], 0.93),
            "hi": self._reading("hi", ["?"], 0.6),
        }
        result, calls = self._run(monkeypatch, readings, ("en", "te", "hi"))
        assert calls == ["en", "te"]
        assert result["language_hint"] == "Telugu"
        assert result["languages_tried"] == ["en", "te"]


class _WordTokenizer:
    """Counts whitespace-separated words as tokens."""

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


class TestTranslationChunking:
    def test_short_text_is_one_piece(self):
        assert translation_ai._pieces("One sentence. Two.", _WordTokenizer()) == [(0, "One sentence. Two.")]

    def test_long_lines_split_at_sentence_boundaries(self, monkeypatch):
        monkeypatch.setattr(translation_ai, "CHUNK_TOKENS", 4)
        pieces = translation_ai._pieces("a b c. d e f. g h.", _WordTokenizer())
        assert pieces == [(0, "a b c."), (0, "d e f."), (0, "g h.")]

    def test_line_breaks_are_kept_as_separate_pieces(self):
        pieces = translation_ai._pieces("First line.\nSecond line.", _WordTokenizer())
        assert [line for line, _ in pieces] == [0, 1]

    def test_normalising_keeps_line_structure(self):
        assert translation_ai._normalize("a\u00a0 b\t c\n\n  d\ufeff e  ") == "a b c\nd e"


class TestSummaryCoverage:
    def test_spread_keeps_first_and_last_in_order(self):
        pieces = [f"p{i}" for i in range(10)]
        chosen = summarization._spread(pieces, 4)
        assert chosen[0] == "p0" and chosen[-1] == "p9"
        assert chosen == sorted(chosen, key=lambda p: int(p[1:]))
        assert len(chosen) == 4

    def test_spread_returns_short_documents_whole(self):
        assert summarization._spread(["a", "b"], 4) == ["a", "b"]

    def test_tidy_capitalises_sentences(self):
        assert summarization._tidy("the model works. it is fast.") == "The model works. It is fast."

    @pytest.mark.parametrize("text", ["", "too short"])
    def test_short_input_is_refused_without_loading_t5(self, text, monkeypatch):
        monkeypatch.setattr(summarization, "summarize_t5", lambda *a, **k: pytest.fail("T5 was loaded"))
        assert "too short" in summarization.generate_summary(text).lower()


class TestCpuSizing:
    """Thread pools must follow the container's CPU quota, not the host's cores."""

    def test_quota_caps_the_host_core_count(self, monkeypatch):
        monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False)
        monkeypatch.setattr(cpu, "_cgroup_quota", lambda: 2.0)
        assert cpu.effective_cpus() == 2

    def test_fractional_quota_rounds_up(self, monkeypatch):
        monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _pid: set(range(8)), raising=False)
        monkeypatch.setattr(cpu, "_cgroup_quota", lambda: 1.5)
        assert cpu.effective_cpus() == 2

    def test_no_quota_uses_every_available_core(self, monkeypatch):
        monkeypatch.setattr(cpu.os, "sched_getaffinity", lambda _pid: set(range(6)), raising=False)
        monkeypatch.setattr(cpu, "_cgroup_quota", lambda: None)
        assert cpu.effective_cpus() == 6


class TestFetchAgreesWithTheApp:
    def test_ocr_languages_follow_the_setting(self, monkeypatch):
        monkeypatch.setenv("OMNIAI_OCR_LANGS", "en, te,HI")
        assert fetch_paper.ocr_languages() == ["en", "te", "hi"]

    def test_ocr_languages_default_to_english(self, monkeypatch):
        monkeypatch.delenv("OMNIAI_OCR_LANGS", raising=False)
        assert fetch_paper.ocr_languages() == ["en"]

    def test_ocr_is_pinned_to_the_papers_engine_version(self):
        assert PADDLE_OCR_OPTIONS["ocr_version"] == "PP-OCRv5"
        assert PADDLE_OCR_OPTIONS["use_textline_orientation"] is True


def test_a_failing_model_costs_only_its_own_step(monkeypatch):
    """An unexpected model error becomes BackendUnavailable, which the
    pipeline reports as one failed step while the other models still run."""
    from backend.config import reload_settings
    from backend.modules import image_classification
    from backend.providers.paper import PaperBackend

    def explode(_path):
        raise NotImplementedError("engine bug")

    monkeypatch.setattr(image_classification, "classify_image", explode)
    backend = PaperBackend.__new__(PaperBackend)
    backend._settings = reload_settings()
    with pytest.raises(BackendUnavailable, match="Image classification could not process") as caught:
        backend.classify_image("photo.jpg")
    # The visitor learns which step failed, not how the server is built.
    assert "NotImplementedError" not in str(caught.value)
    assert "engine bug" not in str(caught.value)


def test_health_reports_loaded_models_when_the_backend_can(client, backend, monkeypatch):
    monkeypatch.setattr(
        type(backend), "status", lambda self: {"models_loaded": ["classify"], "preloading": True},
        raising=False,
    )
    body = client.get("/api/health").get_json()
    assert body["models"] == {"ready": 1, "loading": True}


@pytest.mark.parametrize("model,expected", [
    ("YOLOv8x (Ultralytics/YOLOv8)", " with YOLOv8x"),
    ("PaddleOCR (PP-OCRv5), PaddlePaddle on Hugging Face", " with PaddleOCR (PP-OCRv5)"),
    ("spaCy en_core_web_sm (spacy/en_core_web_sm) + RoBERTa sentiment (cardiffnlp/x)",
     " with spaCy en_core_web_sm + RoBERTa sentiment"),
    (None, ""),
])
def test_explanations_name_the_model_without_its_repository(model, expected):
    from backend.modules.explainability import _with

    assert _with({"model": model}) == expected


class TestMemoryBudget:
    """Under a budget, the least recently used models make way; without one, nothing is unloaded."""

    @staticmethod
    def _handles(monkeypatch, budget):
        from backend.modules import lazy

        monkeypatch.setattr(lazy, "_REGISTRY", {})
        monkeypatch.setenv("OMNIAI_MODEL_BUDGET_MB", str(budget))
        small = lazy.Lazy("small", "small", lambda: object(), resident_mb=300)
        medium = lazy.Lazy("medium", "medium", lambda: object(), resident_mb=500)
        large = lazy.Lazy("large", "large", lambda: object(), resident_mb=700, peak_mb=900)
        return small, medium, large

    def test_the_least_recently_used_model_is_unloaded_first(self, monkeypatch):
        small, medium, large = self._handles(monkeypatch, 1500)
        small.get()
        medium.get()
        small.get()  # small is now the more recently used
        large.get()  # needs 900 on top of 800: something must go
        assert large.loaded and small.loaded
        assert not medium.loaded

    def test_without_a_budget_nothing_is_unloaded(self, monkeypatch):
        small, medium, large = self._handles(monkeypatch, 0)
        for handle in (small, medium, large):
            handle.get()
        assert small.loaded and medium.loaded and large.loaded

    def test_an_unloaded_model_reloads_on_next_use(self, monkeypatch):
        small, medium, large = self._handles(monkeypatch, 1000)
        small.get()
        large.get()
        assert not small.loaded
        assert small.get() is not None and small.loaded

    def test_a_caller_keeps_its_model_after_it_is_unloaded(self, monkeypatch):
        small, _, large = self._handles(monkeypatch, 1000)
        held = small.get()
        large.get()
        assert not small.loaded
        assert held is not None
