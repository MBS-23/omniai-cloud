"""Where the paper profile's weights come from: the Hugging Face Hub.

Every model the paper names is published on the Hub, including the ones whose
libraries normally fetch weights from elsewhere (Ultralytics' GitHub releases,
PaddleOCR's own servers, spaCy's pip index). Fetching all of them from one
place means one cache, one download step at image build time, and one switch,
``HF_HUB_OFFLINE=1``, that guarantees no request ever touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HubModel:
    """A model as the paper names it, and where its weights live."""

    key: str
    paper_name: str
    repo_id: str
    # A single file to fetch, for repos that hold several variants.
    filename: str | None = None
    size: str = ""


# The paper's model stack. Section 3C and Appendix A of the thesis, with T5 for
# summarisation as in the conference paper (reference [9]).
PAPER_MODELS: dict[str, HubModel] = {
    "classify": HubModel("classify", "ResNet50", "timm/resnet50.tv2_in1k", "model.safetensors", "103 MB"),
    "detect": HubModel("detect", "YOLOv8n", "Ultralytics/YOLOv8", "yolov8n.pt", "7 MB"),
    "wildlife": HubModel("wildlife", "YOLOv8x", "Ultralytics/YOLOv8", "yolov8x.pt", "137 MB"),
    "ocr": HubModel("ocr", "PaddleOCR (PP-OCRv5)", "PaddlePaddle/PP-OCRv5_server_det", size="110 MB"),
    "caption": HubModel("caption", "BLIP (large)", "Salesforce/blip-image-captioning-large", size="1.9 GB"),
    "translate": HubModel("translate", "NLLB-200 (600M)", "facebook/nllb-200-distilled-600M", size="2.5 GB"),
    "sentiment": HubModel(
        "sentiment", "RoBERTa sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest", size="500 MB"
    ),
    "ner": HubModel("ner", "spaCy en_core_web_sm", "spacy/en_core_web_sm", size="13 MB"),
    "summarize": HubModel("summarize", "T5 (base)", "google-t5/t5-base", size="892 MB"),
}


# The PaddleOCR options that decide which PP-OCRv5 models are used, shared by
# the download step and the OCR module so the two can never disagree.
PADDLE_OCR_OPTIONS: dict[str, object] = {
    # Pinned: PaddleOCR 3.7 would otherwise pick PP-OCRv6 for some languages.
    # The paper's engine is PP-OCRv5 (PaddleOCR 3.0).
    "ocr_version": "PP-OCRv5",
    # Page-level dewarping and orientation are for photographed documents; the
    # paper's pipeline does line-level orientation only.
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    # The original's use_angle_cls=True: reads rotated text lines.
    "use_textline_orientation": True,
    # oneDNN is what makes CPU inference fast: 3.4 s for a page against 117 s
    # and 3 GB without it. PaddlePaddle 3.3 broke it for PP-OCRv5
    # ("ConvertPirAttribute2RuntimeAttribute not support"), which is why
    # requirements/paper.txt pins 3.2.
    "enable_mkldnn": True,
    "device": "cpu",
}


def hub_file(repo_id: str, filename: str) -> str:
    """Local path to one file from a Hub repo, downloading it once if needed."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


def model_label(key: str) -> str:
    """How a model is named to people: the paper's name and its Hub repo."""
    model = PAPER_MODELS[key]
    return f"{model.paper_name} ({model.repo_id})"
