"""Image classification with ResNet50 (paper section 4A; He et al. 2016, ref [1]).

The weights are torchvision's IMAGENET1K_V2 set (80.9% top-1 on ImageNet),
which is what the original code loaded through ``ResNet50_Weights.DEFAULT``.
They are fetched from the Hugging Face Hub copy that timm publishes as
``timm/resnet50.tv2_in1k``: the same tensors under the same parameter names, so
the architecture, preprocessing and class list below are torchvision's own.
"""

from __future__ import annotations

from typing import Any

from backend.modules.hub import PAPER_MODELS, hub_file
from backend.modules.lazy import Lazy

TOP_K = 5


def _build() -> tuple[Any, Any, list[str]]:
    from safetensors.torch import load_file
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    spec = PAPER_MODELS["classify"]

    model = resnet50(weights=None)
    # strict=True: a checkpoint that does not match the architecture must fail
    # loudly here, not classify every image as the same class later.
    model.load_state_dict(load_file(hub_file(spec.repo_id, spec.filename)), strict=True)
    model.eval()
    return model, weights.transforms(), list(weights.meta["categories"])


MODEL = Lazy("classify", "ResNet50", _build)


def classify_image(image_path: str) -> dict[str, Any]:
    """Classify an image against the 1,000 ImageNet classes.

    Returns the top label with its confidence as a percentage, plus the next
    four, which the explainability report uses to show how close a call was.
    """
    import torch
    from PIL import Image

    model, preprocess, categories = MODEL.get()
    image = Image.open(image_path).convert("RGB")

    with torch.inference_mode():
        probabilities = model(preprocess(image).unsqueeze(0)).softmax(dim=1)[0]

    scores, indices = probabilities.topk(TOP_K)
    top = [
        {"label": categories[index].lower(), "confidence": round(score * 100, 2)}
        for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
    ]
    return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top_k": top}
