"""How many CPUs this process may actually use.

``os.cpu_count()`` reports the host's cores, not the container's share. On a
Hugging Face Space it says a large number while the cgroup allows two, so
PyTorch and PaddlePaddle each start that many threads, fight over two CPUs,
and run at half speed or worse. Measured on PaddleOCR under a two-CPU limit:
8.2 s per page with default threading, 4.3 s with threads matched to the quota.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


def _cgroup_quota() -> float | None:
    """CPUs allowed by the cgroup, or None when unlimited or unknown."""
    try:
        # cgroup v2: "max 100000" (unlimited) or "200000 100000" (two CPUs).
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
        return None
    except (OSError, ValueError):
        pass
    try:
        # cgroup v1.
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def quantize_for_cpu(model: Any) -> Any:
    """int8 dynamic quantisation of a model's Linear layers, in place.

    Measured on two CPUs against the fp32 weights: BLIP-large captions in 4 to
    7 s instead of 48 to 61 s, NLLB-200 and T5 two to four times faster. NLLB
    and T5 produced the same text apart from punctuation; BLIP's captions were
    worded differently but described the same scene. Done in place,
    because the default copy briefly doubles memory, which is what fails on a
    small machine. Skipped when OMNIAI_QUANTIZE=none, or on a GPU.
    """
    from backend.config import get_settings

    settings = get_settings()
    if not settings.quantize or settings.torch_device == "cuda":
        return model

    import torch

    torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return model


def effective_cpus() -> int:
    """The number of threads worth running for CPU-bound inference."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    quota = _cgroup_quota()
    if quota is not None:
        available = min(available, math.ceil(quota))
    return max(1, available)
