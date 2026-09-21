"""Prove every paper model loads and runs, before anything is served.

    python -m backend.models.verify_paper

Run by the Docker build with the network switched off, so a model missing
from the cache fails the build rather than the first visitor. Loading is not
enough on its own: PaddlePaddle 3.3 loaded PP-OCRv5 without complaint and then
failed on the first page it was asked to read. So the engines that have broken
that way are also run once, on a small image made here.

Models are checked one at a time and released, since a build machine may not
hold all of them at once.
"""

from __future__ import annotations

import gc
import sys
import tempfile
from pathlib import Path

_LOADERS = ("classify", "detect", "wildlife", "caption", "translate", "sentiment", "ner", "summarize")


def _sample_image(directory: Path) -> str:
    """A white card with two lines of black text."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((40, 40), "OmniAI Cloud 2026", fill="black", font=font)
    draw.text((40, 120), "Paper model check", fill="black", font=font)
    path = directory / "verify.png"
    image.save(path)
    return str(path)


def verify_paper() -> int:
    from backend.config import get_settings
    from backend.providers.paper import PaperBackend

    failures: list[str] = []
    for key in _LOADERS:
        try:
            PaperBackend.model_handle(key).probe()
            print(f"  [loads  ] {key}", flush=True)
        except Exception as error:
            print(f"  [BROKEN ] {key:<10} {error}", file=sys.stderr)
            failures.append(key)
        gc.collect()

    with tempfile.TemporaryDirectory() as scratch:
        image = _sample_image(Path(scratch))

        try:
            from backend.modules.ocr_paddle import extract_text_paddle

            reading = extract_text_paddle(image, get_settings().ocr_langs[:1])
            # "OmniAl" for "OmniAI" is the usual l/I confusion, not a failure.
            if "omnia" not in reading.get("text", "").lower():
                raise RuntimeError(f"read {reading.get('text')!r} instead of the sample text")
            print(f"  [runs   ] ocr        read {reading['text']!r}", flush=True)
        except Exception as error:
            print(f"  [BROKEN ] ocr        {error}", file=sys.stderr)
            failures.append("ocr")

        try:
            from backend.modules.object_detection import detect_objects

            detect_objects(image)
            print("  [runs   ] detect", flush=True)
        except Exception as error:
            print(f"  [BROKEN ] detect     {error}", file=sys.stderr)
            failures.append("detect-run")

    print()
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Every paper model loads, and OCR and detection run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify_paper())
