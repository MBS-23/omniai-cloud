"""Load each model once, on first use, and never twice.

The paper manages its models through "lazy loading and the singleton pattern"
(section 3C). The original modules did that with a bare module-level
``if _model is None`` check, which two simultaneous first requests could both
pass, loading gigabytes of weights twice. This does the same job under a lock,
and records what is loaded so the health endpoint can report it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from backend.utils.logger import get_logger

logger = get_logger("OmniAI.models")

_REGISTRY: dict[str, Lazy] = {}


class Lazy:
    """A model that is built on first use, exactly once, thread-safely."""

    def __init__(self, key: str, name: str, build: Callable[[], Any]) -> None:
        self.key = key
        self.name = name
        self._build = build
        self._value: Any = None
        self._load_lock = threading.Lock()
        # For libraries whose predictors keep per-call state (Ultralytics,
        # PaddleOCR) and so must not run two inferences at once.
        self.use_lock = threading.Lock()
        self.load_seconds: float | None = None
        _REGISTRY[key] = self

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def probe(self) -> None:
        """Build the model once and discard it: proves the weights load.

        Used at image build time, where holding every model at once would
        exceed the builder's memory.
        """
        value = self._build()
        del value

    def get(self) -> Any:
        if self._value is None:
            with self._load_lock:
                if self._value is None:
                    started = time.perf_counter()
                    logger.info("Loading %s", self.name)
                    self._value = self._build()
                    self.load_seconds = round(time.perf_counter() - started, 1)
                    logger.info("Loaded %s in %.1fs", self.name, self.load_seconds)
        return self._value


def loaded_models() -> dict[str, float | None]:
    """Keys of the models in memory, with how long each took to load."""
    return {key: lazy.load_seconds for key, lazy in _REGISTRY.items() if lazy.loaded}
