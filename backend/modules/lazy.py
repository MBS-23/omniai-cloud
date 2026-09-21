"""Load each model once, on first use, and fit them into the memory available.

The paper manages its models through "lazy loading and the singleton pattern"
(section 3C). The original modules did that with a bare module-level
``if _model is None`` check, which two simultaneous first requests could both
pass, loading gigabytes of weights twice. This does the same job under a lock,
and records what is loaded so the health endpoint can report it.

On a machine that cannot hold every model at once (a laptop's Docker gets a
few gigabytes), ``OMNIAI_MODEL_BUDGET_MB`` sets how much the models may use.
Before a model loads, the least recently used ones are unloaded until it
fits: switching from images to translation then costs a reload of a few
seconds instead of an out-of-memory crash. With no budget (the default on a
server with the memory), nothing is ever unloaded.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from backend.utils.logger import get_logger

logger = get_logger("OmniAI.models")

_REGISTRY: dict[str, Lazy] = {}
# Loads are serialised when a budget is set, so two threads cannot each make
# room for their own model and then both load into the same space.
_BUDGET_LOCK = threading.Lock()


def _budget_mb() -> int:
    try:
        return max(0, int(os.getenv("OMNIAI_MODEL_BUDGET_MB", "0") or 0))
    except ValueError:
        return 0


def _release_memory() -> None:
    """Collect, then hand freed pages back to the operating system."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Lazy:
    """A model that is built on first use, exactly once, thread-safely.

    ``resident_mb`` is roughly what the loaded model occupies; ``peak_mb`` what
    loading it needs at its highest, which is more for models quantised after
    loading. Both only matter when a memory budget is set.
    """

    def __init__(
        self,
        key: str,
        name: str,
        build: Callable[[], Any],
        *,
        resident_mb: int = 0,
        peak_mb: int | None = None,
    ) -> None:
        self.key = key
        self.name = name
        self._build = build
        self._value: Any = None
        self._load_lock = threading.Lock()
        # For libraries whose predictors keep per-call state (Ultralytics,
        # PaddleOCR) and so must not run two inferences at once.
        self.use_lock = threading.Lock()
        self.load_seconds: float | None = None
        self.resident_mb = resident_mb
        self.peak_mb = peak_mb if peak_mb is not None else resident_mb
        self.last_used = 0.0
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
        # The object is returned from a local reference: if another request
        # unloads this model a moment later, the caller's copy stays valid
        # until it is done with it.
        value = self._value
        if value is None:
            budget = _budget_mb()
            if budget:
                with _BUDGET_LOCK:
                    value = self._load(budget)
            else:
                value = self._load(0)
        self.last_used = time.monotonic()
        return value

    def _load(self, budget: int) -> Any:
        with self._load_lock:
            if self._value is not None:
                return self._value
            if budget:
                _make_room(self, budget)
            started = time.perf_counter()
            logger.info("Loading %s", self.name)
            value = self._build()
            self._value = value
            self.last_used = time.monotonic()
            self.load_seconds = round(time.perf_counter() - started, 1)
            logger.info("Loaded %s in %.1fs", self.name, self.load_seconds)
            return value

    def unload(self) -> None:
        """Drop the model. A request already using it keeps it until it finishes."""
        with self._load_lock:
            self._value = None
            self.load_seconds = None


def _make_room(incoming: Lazy, budget: int) -> None:
    loaded = [lazy for lazy in _REGISTRY.values() if lazy.loaded and lazy is not incoming]
    used = sum(lazy.resident_mb for lazy in loaded)
    evicted = False
    for victim in sorted(loaded, key=lambda lazy: lazy.last_used):
        if used + incoming.peak_mb <= budget:
            break
        logger.info("Unloading %s to make room for %s", victim.name, incoming.name)
        victim.unload()
        used -= victim.resident_mb
        evicted = True
    if evicted:
        _release_memory()


def fits_in_budget(lazy: Lazy) -> bool:
    """Whether loading this model now would stay within the budget untouched."""
    budget = _budget_mb()
    if not budget:
        return True
    used = sum(item.resident_mb for item in _REGISTRY.values() if item.loaded)
    return used + lazy.peak_mb <= budget


def loaded_models() -> dict[str, float | None]:
    """Keys of the models in memory, with how long each took to load."""
    return {key: lazy.load_seconds for key, lazy in _REGISTRY.items() if lazy.loaded}
