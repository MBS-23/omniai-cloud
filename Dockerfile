# ==============================================================================
# OmniAI Cloud: the paper profile
#
# The models published in the paper (ResNet50, YOLOv8n and YOLOv8x, PaddleOCR,
# BLIP, NLLB-200, spaCy, RoBERTa and T5), fetched from the Hugging Face Hub at
# build time and executed in-process. No API key, no rate limit.
#
# This is the image the Hugging Face Space runs: it listens on 7860 and runs as
# uid 1000, which is what Spaces expect. It needs about 8 GB of memory with
# every model loaded; set OMNIAI_PRELOAD=none on a smaller machine and models
# load on first use instead.
# ==============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# poppler-utils: pdftoppm, which rasterises scanned PDFs for OCR.
# libgl1, libglib2.0-0: OpenCV, which Ultralytics and PaddleOCR both import.
# libgomp1: OpenMP, which PaddlePaddle's CPU kernels use.
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils libgl1 libglib2.0-0 libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OMNIAI_OCR_LANGS=en,te,hi

WORKDIR /app

# The CPU build of PyTorch first: the default wheel brings ~2.5 GB of CUDA
# libraries that a CPU host never loads. Only the two requirement files this
# image reads are copied, so editing another profile's list does not throw
# away this layer and every one after it.
COPY requirements/base.txt requirements/paper.txt requirements/
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/paper.txt \
 && pip install --no-deps --force-reinstall opencv-contrib-python==4.10.0.84

# Weights, in a layer of their own. The fetch step reads only the model
# inventory, so only that is copied first: editing any other part of the
# application reuses this layer instead of downloading 6 GB again.
COPY --chown=user backend/__init__.py backend/
COPY --chown=user backend/modules/__init__.py backend/modules/hub.py backend/modules/
COPY --chown=user backend/models/fetch_paper.py backend/models/
USER user
RUN python -m backend.models.fetch_paper

# The application.
USER root
COPY --chown=user gunicorn.conf.py ./
COPY --chown=user backend/ backend/
RUN mkdir -p /app/temp_uploads /app/logs && chown -R user:user /app/temp_uploads /app/logs
USER user

# Prove every model loads from the cache alone, with the network switched
# off, and that OCR and detection actually run, before the image may exist.
RUN HF_HUB_OFFLINE=1 python -m backend.models.verify_paper

ENV OMNIAI_PROFILE=paper \
    OMNIAI_PRELOAD=all \
    OMNIAI_RATE_LIMIT_PER_MINUTE=12 \
    OMNIAI_HOST=0.0.0.0 \
    OMNIAI_PORT=7860 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://127.0.0.1:7860/api/health || exit 1

CMD ["gunicorn", "--config", "gunicorn.conf.py", "backend.main:app"]
