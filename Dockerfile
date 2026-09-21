# ==============================================================================
# OmniAI Cloud — lite profile image
#
# Builds the API-backed profile: no PyTorch, no CUDA, no model weights.
# The result is a small image that starts in about a second, which is what
# makes public deployment practical. The full profile is a local/GPU
# configuration and is not containerised here; see docs/deployment.md.
# ==============================================================================

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# poppler-utils provides pdftoppm, which pdf2image needs to rasterise scanned
# PDFs. Installing it here is what replaces the hardcoded Windows path the
# original code carried.
RUN apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so edits to application code do not invalidate this layer.
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/lite.txt

COPY backend/ backend/

# Run as a non-root user. The writable directories are created and handed over
# explicitly; everything else stays read-only to the process.
RUN useradd --create-home --uid 10001 omniai \
 && mkdir -p /app/temp_uploads /app/logs \
 && chown -R omniai:omniai /app/temp_uploads /app/logs
USER omniai

ENV OMNIAI_PROFILE=lite \
    OMNIAI_HOST=0.0.0.0 \
    OMNIAI_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# One worker, several threads: requests are I/O-bound on the upstream API.
# The in-memory rate limiter is per-process, so scaling workers weakens it
# proportionally -- see docs/deployment.md before raising this.
CMD ["gunicorn", "backend.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "8", \
     "--worker-class", "gthread", \
     "--timeout", "180", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
