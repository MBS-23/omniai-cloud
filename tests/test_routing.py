"""The routing decisions the paper claims as its contribution."""

from __future__ import annotations

import pytest

from backend.agent.auto_selector import analyze_input, auto_model_selector
from backend.agent.taxonomy import is_animal_image
from backend.core.labels import refine_detections
from backend.core.pipeline import run_file_analysis, run_text_analysis


class TestInputDetection:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("photo.jpg", "image"), ("photo.JPEG", "image"),
            ("shot.png", "image"), ("pic.webp", "image"),
            ("report.pdf", "document"), ("notes.DOCX", "document"),
            ("archive.zip", "unknown"), ("script.exe", "unknown"),
        ],
    )
    def test_extension_routing(self, filename, expected):
        assert analyze_input(file_path=filename)["input_type"] == expected

    def test_text_language_detection(self):
        assert analyze_input(text="This is a sentence in English.")["language"] == "en"

    def test_unreadable_text_does_not_raise(self):
        # langdetect throws on input with no letters; routing must survive it.
        assert analyze_input(text="!!! ??? 123")["language"] in {"unknown", "en"}


class TestModelSelection:
    def test_english_text_skips_translation(self):
        decision = auto_model_selector({"input_type": "text", "language": "en"})
        assert "translation_ai" not in decision["models"]
        assert "text_analytics" in decision["models"]

    def test_non_english_text_adds_translation(self):
        decision = auto_model_selector({"input_type": "text", "language": "te"})
        assert decision["models"][0] == "translation_ai"

    def test_unknown_language_does_not_trigger_translation(self):
        decision = auto_model_selector({"input_type": "text", "language": "unknown"})
        assert "translation_ai" not in decision["models"]

    def test_ocr_is_ordered_before_detection(self):
        """Detection reads the OCR result, so OCR must run first."""
        models = auto_model_selector({"input_type": "image", "language": None})["models"]
        assert models.index("ocr_paddle") < models.index("object_detection")

    def test_document_uses_one_module(self):
        decision = auto_model_selector({"input_type": "document", "language": None})
        assert decision["models"] == ["document_ai"]


class TestTaxonomy:
    @pytest.mark.parametrize(
        "label",
        # These four are the regression: the taxonomy stored them capitalised
        # while lookups lowercased the input, so they never matched.
        ["African elephant", "Indian elephant", "Arctic fox", "American black bear"],
    )
    def test_capitalised_imagenet_labels_match(self, label):
        assert is_animal_image(label) is True

    @pytest.mark.parametrize("label", ["lion", "LION", "  Lion  "])
    def test_matching_is_case_and_space_insensitive(self, label):
        assert is_animal_image(label) is True

    @pytest.mark.parametrize("label", ["laptop", "desk", "", None])
    def test_non_animals_rejected(self, label):
        assert is_animal_image(label) is False


class TestLabelRefinement:
    def test_generic_label_promoted_to_species(self):
        refined = refine_detections(
            [{"label": "elephant", "confidence": 90.0}], "african elephant", 95.0
        )
        assert refined[0]["label"] == "african elephant"
        assert refined[0]["original_yolo_label"] == "elephant"
        assert refined[0]["corrected"] is True

    def test_only_the_primary_subject_is_refined(self):
        """In a hunting scene the second animal is a different species."""
        refined = refine_detections(
            [
                {"label": "cat", "confidence": 90.0},
                {"label": "cat", "confidence": 70.0},
            ],
            "lion",
            95.0,
        )
        assert refined[0]["label"] == "lion"
        assert refined[1]["label"] == "cat"
        assert "corrected" not in refined[1]

    def test_mismatched_species_is_left_alone(self):
        refined = refine_detections([{"label": "bird", "confidence": 90.0}], "lion", 95.0)
        assert refined[0]["label"] == "bird"

    def test_empty_input_is_safe(self):
        assert refine_detections([], "lion", 95.0) == []


class TestImagePipeline:
    def test_wildlife_image_routes_to_wildlife_detector_and_skips_ocr(
        self, backend, scene_image
    ):
        backend._overrides["classify_image"] = {"label": "lion", "confidence": 94.0}
        backend._overrides["detect_objects"] = {
            "object_count": 1,
            "detections": [{"label": "cat", "confidence": 81.0, "bbox": [0, 0, 1, 1]}],
            "bbox_available": True,
        }

        result = run_file_analysis(scene_image, backend)

        assert "wildlife" in backend.calls, "should use the high-capacity detector"
        assert "extract_text" not in backend.calls, "OCR is pointless on a lion photo"
        assert "wildlife_detection" in result["selected_modules"]
        # COCO's "cat" refined to the classifier's species label.
        assert result["results"]["object_detection"]["detections"][0]["label"] == "lion"

    def test_refinement_does_not_leak_between_runs(self, backend, scene_image):
        """Two analyses of the same input must give the same answer.

        The pipeline used to refine labels in the dict the backend returned.
        A backend that reuses a response object then saw its own data rewritten,
        and the second run refined the already-refined result.
        """
        backend._overrides["classify_image"] = {"label": "lion", "confidence": 94.0}
        backend._overrides["detect_objects"] = {
            "object_count": 2,
            "bbox_available": True,
            "detections": [
                {"label": "cat", "confidence": 88.0, "bbox": [0, 0, 1, 1]},
                {"label": "cat", "confidence": 61.0, "bbox": [2, 2, 3, 3]},
            ],
        }

        first = run_file_analysis(scene_image, backend)
        second = run_file_analysis(scene_image, backend)

        def labels(result):
            return [d["label"] for d in result["results"]["object_detection"]["detections"]]

        assert labels(first) == ["lion", "cat"]
        assert labels(second) == labels(first), "the second run must not re-refine"

    def test_text_heavy_image_skips_object_detection(self, backend, scene_image):
        backend._overrides["extract_text"] = {
            "has_text": True,
            "text": " ".join(f"word{n}" for n in range(40)),
            "lines": ["a line"],
        }
        result = run_file_analysis(scene_image, backend)

        assert "objects" not in backend.calls, "a page of text is not a scene"
        assert "ocr_paddle" in result["selected_modules"]
        assert "object_detection" not in result["results"]

    def test_text_heavy_image_skips_captioning(self, backend, scene_image):
        """A page of text needs no description; OCR already has its content.

        Captioning a document also produces a garbled paraphrase of the words
        the model can partly read, so skipping it improves the output as well
        as the latency.
        """
        backend._overrides["extract_text"] = {
            "has_text": True,
            "text": " ".join(f"word{n}" for n in range(40)),
            "lines": ["a line"],
        }
        result = run_file_analysis(scene_image, backend)

        assert "caption_image" not in backend.calls
        assert "caption" not in result["results"]
        assert "blip_vision_language" not in result["selected_modules"]

    def test_scene_image_still_gets_a_caption(self, backend, scene_image):
        """The skip must be specific to text images, not general."""
        result = run_file_analysis(scene_image, backend)
        assert "caption" in result["results"]

    def test_trivial_ocr_output_is_discarded(self, backend, scene_image):
        """Three words of noise must not suppress object detection."""
        backend._overrides["extract_text"] = {
            "has_text": True, "text": "a b c", "lines": ["a b c"],
        }
        result = run_file_analysis(scene_image, backend)

        assert "ocr" not in result["results"]
        assert "object_detection" in result["results"]

    def test_unavailable_capability_degrades_instead_of_failing(
        self, backend, scene_image
    ):
        backend.unavailable = {"caption_image"}
        result = run_file_analysis(scene_image, backend)

        assert "caption" not in result["results"]
        assert any("Captioning unavailable" in w for w in result["warnings"])
        # The rest of the pipeline still produced output.
        assert "image_classification" in result["results"]

    def test_progress_events_are_monotonic_and_complete(self, backend, scene_image):
        events = []
        run_file_analysis(scene_image, backend, on_event=events.append)

        assert events, "the pipeline must report progress"
        percents = [e.percent for e in events]
        assert percents == sorted(percents), "progress must never go backwards"
        assert percents[-1] == 100
        assert {e.status for e in events} <= {"start", "done", "skipped", "error"}

    def test_unsupported_input_is_rejected(self, backend, tmp_path):
        bad = tmp_path / "payload.zip"
        bad.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match="Unsupported input type"):
            run_file_analysis(bad, backend)


class TestTextPipeline:
    def test_translation_runs_between_different_languages(self, backend):
        result = run_text_analysis(
            "కృత్రిమ మేధస్సు", backend, target_language="en"
        )
        assert "translation_ai" in result["selected_modules"]

    def test_matching_languages_skip_translation(self, backend):
        result = run_text_analysis(
            "This is English text for analysis.", backend, target_language="en"
        )
        assert "translation" not in result["results"]
        assert "translate" not in backend.calls

    def test_unapplied_translation_is_reported_not_silently_shown(self, backend):
        """A backend returning the input unchanged must not look like success."""
        backend._overrides["translate"] = {
            "translated_text": "unchanged input",
            "source_language": "te",
            "target_language": "en",
            "translation_applied": False,
            "translation_error": "Invalid API key",
        }
        result = run_text_analysis("కృత్రిమ మేధస్సు", backend, target_language="en")

        assert "translation" not in result["results"], "must not show a fake translation"
        assert any("Invalid API key" in w for w in result["warnings"])

    def test_unsupported_target_language_is_refused(self, backend):
        result = run_text_analysis(
            "Some English text here.", backend, target_language="xx"
        )
        assert any("not a supported translation target" in w for w in result["warnings"])
