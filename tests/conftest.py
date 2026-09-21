"""Shared fixtures.

Tests run against a scripted backend rather than a real one, so the suite
needs no API key, no GPU and no model weights, and it exercises the routing
logic rather than the models' accuracy.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("OMNIAI_PROFILE", "lite")
os.environ.setdefault("OMNIAI_MISTRAL_API_KEY", "test-key-not-used")

from backend.config import reload_settings  # noqa: E402
from backend.providers.base import Backend, BackendUnavailable  # noqa: E402


class FakeBackend(Backend):
    """A backend whose answers are scripted, and which records its calls."""

    name = "fake"

    def __init__(self, **overrides: Any) -> None:
        self.calls: list[str] = []
        self.unavailable: set[str] = set()
        self._overrides = overrides

    def _answer(self, capability: str, default: Any) -> Any:
        self.calls.append(capability)
        if capability in self.unavailable:
            raise BackendUnavailable(f"{capability} is switched off for this test")
        # Deep copy, so a caller that mutates a result cannot change what the
        # next call returns. A real backend builds a fresh response each time.
        return copy.deepcopy(self._overrides.get(capability, default))

    def warmup(self) -> None:
        self.calls.append("warmup")

    def capabilities(self) -> set[str]:
        return {
            "classify_image", "detect_objects", "extract_text", "caption_image",
            "summarize", "translate", "analyze_text",
        }

    def classify_image(self, image_path: str):
        return self._answer("classify_image", {"label": "laptop", "confidence": 88.0})

    def detect_objects(self, image_path: str, *, wildlife: bool = False):
        self.calls.append("wildlife" if wildlife else "objects")
        return self._answer(
            "detect_objects",
            {
                "object_count": 1,
                "detections": [{"label": "laptop", "confidence": 80.0, "bbox": [1, 2, 3, 4]}],
                "bbox_available": True,
            },
        )

    def extract_text(self, image_path: str):
        return self._answer("extract_text", {"has_text": False, "text": "", "lines": []})

    def caption_image(self, image_path: str, *, prompt: str | None = None):
        self.last_prompt = prompt
        return self._answer("caption_image", {"caption": "A laptop on a desk."})

    def summarize(self, text: str, *, max_length: int = 200) -> str:
        return self._answer("summarize", "A short summary of the document.")

    def translate(self, text: str, *, source_lang: str, target_lang: str):
        return self._answer(
            "translate",
            {
                "translated_text": f"[{target_lang}] {text}",
                "source_language": source_lang,
                "target_language": target_lang,
                "translation_applied": True,
            },
        )

    def analyze_text(self, text: str):
        return self._answer(
            "analyze_text",
            {
                "original_language": "en",
                "translated_text": None,
                "sentiment": "POSITIVE",
                "sentiment_confidence": 91.0,
                "entities": [{"text": "OmniAI", "label": "ORG"}],
                "explanation": "scripted",
            },
        )


@pytest.fixture
def settings():
    return reload_settings()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def app(settings, backend):
    from backend.app import create_app
    from backend.providers import reset_backend, set_backend

    application = create_app(settings, warm=False)
    set_backend(backend)
    application.config["TESTING"] = True
    yield application
    reset_backend()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def scene_image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    image = Image.new("RGB", (320, 240), (230, 232, 236))
    ImageDraw.Draw(image).ellipse([80, 60, 240, 180], fill=(180, 110, 60))
    image.save(path)
    return path


@pytest.fixture
def huge_image(tmp_path: Path) -> Path:
    """A small file that decodes to far more pixels than the limit allows."""
    path = tmp_path / "bomb.png"
    Image.new("RGB", (9000, 9000), "white").save(path, optimize=True)
    return path
