"""Fetch the paper profile's models from the Hugging Face Hub.

    python -m backend.models.fetch_paper

Imports nothing from the application except the model inventory in
``backend/modules/hub.py``, so the Docker image can run it in an early layer:
editing application code then reuses the downloaded weights instead of
fetching several gigabytes again. ``backend.models.verify_paper`` checks, later
and against the real application code, that every model loads and runs.
"""

from __future__ import annotations

import os
import sys

# Files per repository, so each download is exactly what the loader reads:
# several of these repositories also carry TensorFlow, Flax and Rust copies
# of the same weights, which would triple the image for nothing.
_TOKENIZER_FILES = ["*.json", "*.txt", "*.model"]
PAPER_FILES: dict[str, list[str]] = {
    "caption": [*_TOKENIZER_FILES, "model.safetensors"],
    "translate": [*_TOKENIZER_FILES, "pytorch_model.bin"],
    "sentiment": [*_TOKENIZER_FILES, "pytorch_model.bin"],
    "summarize": [*_TOKENIZER_FILES, "model.safetensors"],
}


def ocr_languages() -> list[str]:
    """The PaddleOCR language packs to fetch: OMNIAI_OCR_LANGS, default English."""
    raw = os.getenv("OMNIAI_OCR_LANGS") or "en"
    return [lang.strip().lower() for lang in raw.split(",") if lang.strip()] or ["en"]


def fetch_paper() -> int:
    """Download every model the paper profile uses."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from backend.modules.hub import PADDLE_OCR_OPTIONS, PAPER_MODELS

    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    failures: list[str] = []
    for key, model in PAPER_MODELS.items():
        print(f"  [get    ] {key:<10} {model.paper_name:<24} {model.repo_id}  ({model.size})", flush=True)
        try:
            if key == "ocr":
                # PaddleOCR fetches its PP-OCRv5 models, from the Hub, the
                # first time a pipeline is built for a language.
                from paddleocr import PaddleOCR

                for lang in ocr_languages():
                    PaddleOCR(lang=lang, **PADDLE_OCR_OPTIONS)
            elif model.filename:
                hf_hub_download(model.repo_id, model.filename)
            elif key == "ner":
                snapshot_download(model.repo_id, ignore_patterns=["*.whl"])
            else:
                snapshot_download(model.repo_id, allow_patterns=PAPER_FILES[key])
        except Exception as error:
            print(f"  [FAILED ] {key:<10} {error}", file=sys.stderr)
            failures.append(key)

    print()
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("The paper's models are cached. Set OMNIAI_PROFILE=paper to use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch_paper())
