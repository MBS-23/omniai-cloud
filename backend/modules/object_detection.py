"""General object detection with YOLOv8n (paper section 4A; Appendix A.2).

The fast, small detector the auto-selector uses for every image that is not
a wildlife photograph. Weights come from the Hugging Face Hub copy Ultralytics
publishes (``Ultralytics/YOLOv8``), rather than from GitHub at first use.

The thresholds are the ones in the thesis appendix: a low confidence floor so
several objects in one scene are all reported, and a looser IoU so overlapping
objects are not merged.
"""

from __future__ import annotations

from typing import Any

from backend.modules.hub import PAPER_MODELS, hub_file
from backend.modules.lazy import Lazy

CONFIDENCE = 0.20
IOU = 0.5
MAX_DETECTIONS = 10


def load_yolo(key: str) -> Any:
    """Load a YOLOv8 checkpoint from the Hub, with Ultralytics' telemetry off."""
    from ultralytics import YOLO

    try:
        from ultralytics import settings

        # Ultralytics sends anonymous usage events unless told not to. Nothing
        # about a user's upload should leave the server, even anonymously.
        settings.update({"sync": False})
    except Exception:
        pass

    spec = PAPER_MODELS[key]
    return YOLO(hub_file(spec.repo_id, spec.filename), task="detect")


def run_yolo(model: Lazy, image_path: str, **options: Any) -> dict[str, Any]:
    """One detection pass, as the list of labelled boxes the pipeline expects."""
    yolo = model.get()
    # Ultralytics predictors hold per-call state; two concurrent calls on one
    # model object corrupt each other's results.
    with model.use_lock:
        result = yolo(image_path, verbose=False, **options)[0]

    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                "label": str(yolo.names[int(box.cls)]).lower(),
                "confidence": round(float(box.conf) * 100, 2),
                "bbox": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return {"object_count": len(detections), "detections": detections}


MODEL = Lazy("detect", "YOLOv8n", lambda: load_yolo("detect"), resident_mb=80)


def detect_objects(image_path: str) -> dict[str, Any]:
    """Detect the COCO objects in an image with YOLOv8n."""
    return run_yolo(MODEL, image_path, conf=CONFIDENCE, iou=IOU, max_det=MAX_DETECTIONS)
