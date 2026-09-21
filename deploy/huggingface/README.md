---
title: OmniAI Cloud
emoji: 🧭
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: true
startup_duration_timeout: 1h
short_description: Paper's models - ResNet50, YOLOv8, PaddleOCR, BLIP, NLLB, T5
---

# OmniAI Cloud

**An intelligent unified multi-modal AI system with automatic model selection and explainability.**
Department of Artificial Intelligence and Machine Learning, Sasi Institute of Technology and Engineering.

Upload an image, a PDF, a Word document, or type text. OmniAI works out what it received, runs only the
models that suit it, and explains every decision.

This Space runs **the models published in the paper**, downloaded from the Hugging Face Hub and executed
inside the Space. No API key is involved and there is no rate limit.

| Capability | Model | Hub repository |
|---|---|---|
| Image classification | ResNet50 | `timm/resnet50.tv2_in1k` |
| Object detection | YOLOv8n | `Ultralytics/YOLOv8` |
| Wildlife detection | YOLOv8x | `Ultralytics/YOLOv8` |
| Text extraction | PaddleOCR (PP-OCRv5): English, Telugu, Hindi | `PaddlePaddle/*PP-OCRv5*` |
| Captioning | BLIP (large) | `Salesforce/blip-image-captioning-large` |
| Translation | NLLB-200 (distilled 600M) | `facebook/nllb-200-distilled-600M` |
| Sentiment | RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Named entities | spaCy | `spacy/en_core_web_sm` |
| Summarisation | T5 (base) | `google-t5/t5-base` |

The Space runs on shared CPU, so a full image analysis takes a few seconds and a long document takes longer.
Progress streams live while it works. `/api/health` shows which models are loaded.
