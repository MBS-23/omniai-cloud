"""Wildlife detection with YOLOv8x (paper section 4A, "higher-capacity model").

When the classifier reports an animal from the predefined species list, the
auto-selector swaps the small detector for this one. The thesis values are
kept: a very low confidence floor so a partly hidden or distant animal is
still found, and class-specific NMS so overlapping animals (a herd, a hunt)
stay separate. Label refinement then replaces COCO's coarse class ("cow",
"bird") with the classifier's species on the primary subject.
"""

from __future__ import annotations

from typing import Any

from backend.modules.lazy import Lazy
from backend.modules.object_detection import load_yolo, run_yolo

CONFIDENCE = 0.10
IOU = 0.45
MAX_DETECTIONS = 15

MODEL = Lazy("wildlife", "YOLOv8x", lambda: load_yolo("wildlife"), resident_mb=350)


def detect_wildlife(image_path: str) -> dict[str, Any]:
    """Detect animals in an image with YOLOv8x."""
    return run_yolo(
        MODEL, image_path,
        conf=CONFIDENCE, iou=IOU, max_det=MAX_DETECTIONS, agnostic_nms=False,
    )
