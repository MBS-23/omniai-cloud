"""Capability contract shared by the local and hosted backends.

The orchestration layer in :mod:`backend.core.pipeline` only ever talks to a
:class:`Backend`. That is what lets the same routing logic, the same label
refinement and the same explainability text run unchanged whether the models
are executing on this machine or behind the Mistral API.

Result dictionaries deliberately keep the exact key names the original Flask
app returned, so the existing frontend contract and the figures in the paper
remain valid.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable


class Classification(TypedDict, total=False):
    label: str
    confidence: float
    # Present when more than one model was consulted: which models ran, what
    # each said, and how a disagreement was settled.
    ensemble: dict[str, Any]


class Detection(TypedDict, total=False):
    label: str
    confidence: float
    bbox: list[float] | None
    original_yolo_label: str
    corrected: bool


class DetectionResult(TypedDict, total=False):
    object_count: int
    detections: list[Detection]
    # False when the backend can name objects but cannot localise them.
    # The UI uses this to hide the bounding-box overlay rather than draw lies.
    bbox_available: bool


class OcrResult(TypedDict, total=False):
    has_text: bool
    text: str
    lines: list[str]
    word_count: int
    language_hint: str | None
    boxes: list[dict] | None


class CaptionResult(TypedDict, total=False):
    caption: str
    original_caption: str
    enhanced_caption: str


class TranslationResult(TypedDict, total=False):
    translated_text: str
    source_language: str
    target_language: str
    translation_applied: bool
    translation_error: str


class Entity(TypedDict):
    text: str
    label: str


class TextAnalytics(TypedDict, total=False):
    original_language: str
    translated_text: str | None
    sentiment: str
    sentiment_confidence: float
    entities: list[Entity]
    explanation: str


class BackendUnavailable(RuntimeError):
    """Raised when a capability cannot run in the active profile.

    Carries a message written for the person using the app, not for a log file.
    """


@runtime_checkable
class Backend(Protocol):
    """One coherent set of model implementations."""

    name: str

    def warmup(self) -> None:
        """Load or verify whatever this backend needs before first request."""

    def capabilities(self) -> set[str]:
        """Names of the capabilities this backend can actually serve."""

    # ---- vision ----
    def classify_image(self, image_path: str) -> Classification: ...

    def detect_objects(self, image_path: str, *, wildlife: bool = False) -> DetectionResult: ...

    def extract_text(self, image_path: str) -> OcrResult: ...

    def caption_image(self, image_path: str, *, prompt: str | None = None) -> CaptionResult: ...

    # ---- language ----
    def summarize(self, text: str, *, max_length: int = 200) -> str: ...

    def translate(
        self, text: str, *, source_lang: str, target_lang: str
    ) -> TranslationResult: ...

    def analyze_text(self, text: str) -> TextAnalytics: ...


ALL_CAPABILITIES = {
    "classify_image",
    "detect_objects",
    "extract_text",
    "caption_image",
    "summarize",
    "translate",
    "analyze_text",
}
