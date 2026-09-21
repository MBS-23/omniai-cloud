"""Image captioning with BLIP (paper section 4A; Li et al. 2022, ref [7]).

``Salesforce/blip-image-captioning-large``, the checkpoint the original code
used, with its generation settings: wide beam search for a considered caption,
a minimum length so it describes rather than labels, and repetition controls
because BLIP otherwise tends to loop ("a bottle supplement supplement").
"""

from __future__ import annotations

from typing import Any

from backend.modules.hub import PAPER_MODELS
from backend.modules.lazy import Lazy


def _build() -> tuple[Any, Any]:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from backend.utils.cpu import quantize_for_cpu

    repo = PAPER_MODELS["caption"].repo_id
    processor = BlipProcessor.from_pretrained(repo)
    model = BlipForConditionalGeneration.from_pretrained(repo)
    return processor, quantize_for_cpu(model.eval())


MODEL = Lazy("caption", "BLIP (large)", _build, resident_mb=1000, peak_mb=2000)


def initialize_blip_model() -> None:
    MODEL.get()


def generate_image_caption(image_path: str, prompt: str | None = None) -> dict[str, str]:
    """Describe an image in one sentence, optionally steered by a text prefix."""
    import torch
    from PIL import Image

    processor, model = MODEL.get()
    image = Image.open(image_path).convert("RGB")
    inputs = (
        processor(images=image, text=prompt, return_tensors="pt")
        if prompt else
        processor(images=image, return_tensors="pt")
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_length=80,
            min_length=20,
            num_beams=8,
            length_penalty=1.2,
            repetition_penalty=2.0,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False,
        )

    caption = _remove_repeated_words(processor.decode(output[0], skip_special_tokens=True).strip())
    # A prompted caption starts by repeating the prompt; the prompt is context,
    # not part of the description.
    if prompt and caption.lower().startswith(prompt.lower()):
        caption = caption[len(prompt):].strip(" ,.")
    if caption:
        caption = caption[0].upper() + caption[1:]
    return {"caption": caption or "No caption could be generated for this image."}


def _remove_repeated_words(text: str) -> str:
    """Collapse immediate repeats: "a bottle supplement supplement" -> "a bottle supplement"."""
    words = text.split()
    if not words:
        return text
    kept = [words[0]]
    for word in words[1:]:
        if word.lower() != kept[-1].lower():
            kept.append(word)
    return " ".join(kept)
