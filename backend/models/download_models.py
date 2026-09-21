"""Download the models a local profile needs.

    python -m backend.models.download_models --profile paper            # the paper's models
    python -m backend.models.download_models --profile paper --verify   # ...and load each once
    python -m backend.models.download_models            # the offline profile's models
    python -m backend.models.download_models classify   # one offline capability
    python -m backend.models.download_models --check    # report only

Weights land in the standard Hugging Face cache, so `transformers` finds them
without any path configuration and they are shared with other projects.

Roughly 6 GB in total. Once downloaded the application runs with no API key,
no rate limits and no network access.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    approx_size: str
    purpose: str
    loader: Callable[[str], None]


def _image_classification(repo: str) -> None:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    AutoImageProcessor.from_pretrained(repo)
    AutoModelForImageClassification.from_pretrained(repo)


def _object_detection(repo: str) -> None:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    AutoImageProcessor.from_pretrained(repo)
    AutoModelForObjectDetection.from_pretrained(repo)


def _captioning(repo: str) -> None:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    BlipProcessor.from_pretrained(repo)
    BlipForConditionalGeneration.from_pretrained(repo)


def _seq2seq(repo: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    AutoTokenizer.from_pretrained(repo)
    AutoModelForSeq2SeqLM.from_pretrained(repo)


def _sequence_classification(repo: str) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(repo)
    AutoModelForSequenceClassification.from_pretrained(repo)


def _token_classification(repo: str) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(repo)
    AutoModelForTokenClassification.from_pretrained(repo)


def _easyocr(_: str) -> None:
    """EasyOCR fetches its detector and recogniser on first construction."""
    import easyocr

    easyocr.Reader(["en"], gpu=False, verbose=False)


MODELS: dict[str, ModelSpec] = {
    "classify": ModelSpec(
        "classify", "microsoft/resnet-50", "100 MB",
        "Image classification (ImageNet-1k)", _image_classification,
    ),
    "detect": ModelSpec(
        "detect", "facebook/detr-resnet-50", "167 MB",
        "Object detection (COCO classes)", _object_detection,
    ),
    "caption": ModelSpec(
        "caption", "Salesforce/blip-image-captioning-base", "990 MB",
        "Vision-language captioning", _captioning,
    ),
    "ocr": ModelSpec(
        "ocr", "easyocr/craft+crnn", "100 MB",
        "Text detection and recognition", _easyocr,
    ),
    "translate": ModelSpec(
        "translate", "facebook/nllb-200-distilled-600M", "2.4 GB",
        "Translation, 200 languages", _seq2seq,
    ),
    "sentiment": ModelSpec(
        "sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest", "500 MB",
        "Sentiment analysis", _sequence_classification,
    ),
    "ner": ModelSpec(
        "ner", "dslim/bert-base-NER", "430 MB",
        "Named entity recognition", _token_classification,
    ),
    "summarize": ModelSpec(
        "summarize", "sshleifer/distilbart-cnn-12-6", "1.2 GB",
        "Abstractive summarisation", _seq2seq,
    ),
}


def is_cached(spec: ModelSpec) -> bool:
    """True when usable weights are already on disk.

    ``snapshot_download(local_files_only=True)`` succeeds for a *partial*
    snapshot, so it cannot be used on its own: a model half-way through
    downloading would report as ready. This additionally requires an actual
    weight file of a plausible size.
    """
    from pathlib import Path

    if spec.key == "ocr":
        # EasyOCR keeps its weights outside the Hugging Face cache.
        model_dir = Path.home() / ".EasyOCR" / "model"
        return model_dir.is_dir() and any(model_dir.glob("*.pth"))

    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(spec.repo_id, local_files_only=True))
    except Exception:
        return False

    weights = [
        *path.glob("*.safetensors"),
        *path.glob("*.bin"),
        *path.glob("*.h5"),
    ]
    if not weights:
        return False

    # A stub or an interrupted transfer leaves something implausibly small.
    return max(weight.stat().st_size for weight in weights) > 1_000_000


def download(spec: ModelSpec) -> bool:
    if is_cached(spec):
        print(f"  [cached ] {spec.key:<10} {spec.repo_id}")
        return True

    print(f"  [get    ] {spec.key:<10} {spec.repo_id}  ({spec.approx_size})", flush=True)
    try:
        spec.loader(spec.repo_id)
    except Exception as error:
        print(f"  [FAILED ] {spec.key:<10} {error}", file=sys.stderr)
        return False

    print(f"  [done   ] {spec.key:<10} {spec.repo_id}", flush=True)
    return True


def report() -> None:
    width = max(len(s.key) for s in MODELS.values())
    print(f"{'':10}{'CAPABILITY':<{width + 2}}{'SIZE':>9}  MODEL")
    print("-" * 88)
    for spec in MODELS.values():
        mark = "cached " if is_cached(spec) else "missing"
        print(f"  [{mark}] {spec.key:<{width + 2}}{spec.approx_size:>9}  {spec.repo_id}")
    print()
    print("Cached models make the 'offline' profile fully self-contained:")
    print("no API key, no rate limits, no network access at request time.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=[*MODELS, []],
                        help="offline capabilities to fetch (default: all)")
    parser.add_argument("--profile", choices=["offline", "paper"], default="offline",
                        help="which profile's models to fetch")
    parser.add_argument("--verify", action="store_true",
                        help="with --profile paper: load and run every model after downloading")
    parser.add_argument("--check", action="store_true", help="report status and exit")
    args = parser.parse_args()

    if args.profile == "paper":
        from backend.models.fetch_paper import fetch_paper

        status = fetch_paper()
        if status == 0 and args.verify:
            from backend.models.verify_paper import verify_paper

            status = verify_paper()
        return status

    if args.check:
        report()
        return 0

    selected = args.models or list(MODELS)
    total = sum(1 for key in selected if not is_cached(MODELS[key]))
    print(f"Fetching {len(selected)} capability model(s); {total} not yet cached.\n")

    failures = [key for key in selected if not download(MODELS[key])]

    print()
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All models are available. Set OMNIAI_PROFILE=offline to use them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
