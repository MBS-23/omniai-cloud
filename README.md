# OmniAI Cloud

**An intelligent unified multi-modal AI system with automatic model selection and explainability.**

Give it an image, a PDF, a Word document or a block of text. It works out what
it received, routes the input to only the models that suit it, and explains
every decision it made along the way.

Department of Artificial Intelligence and Machine Learning,
Sasi Institute of Technology and Engineering.

**Live:** <https://omniai-cloud.vercel.app> (the `lite` profile on Gemini).
Check what it is running with [`/api/health`](https://omniai-cloud.vercel.app/api/health).

---

## Quick start

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere). Then pick a profile.

**Everything on this machine, no API key** (the `offline` profile, the default):

```bash
pip install -r requirements/offline.txt
```

```bash
python -m backend.models.download_models
```

**Or a light install that calls a hosted model** (the `lite` profile, what the
live site runs):

```bash
pip install -r requirements/lite.txt
```

Copy `.env.example` to `.env`, set `OMNIAI_PROFILE=lite`, and set
`OMNIAI_GEMINI_API_KEY` to a key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Either way, run:

```bash
python -m backend.main
```

Open <http://127.0.0.1:8000>. `GET /api/health` reports the active profile and
capabilities; it answers `503` with a plain-language `detail` when something is
missing.

---

## Runtime profiles

The same pipeline, the same routing logic and the same explainability output
run under every profile. Only model execution differs.

| | `offline` (default) | `lite` | `full` |
|---|---|---|---|
| Where models run | **This machine** | Gemini API (or Mistral) | This machine |
| API key | **not required** | required | not required |
| Rate limits | **none** | yes, per key | none |
| Python | 3.11 to 3.14 | 3.11 to 3.14 | 3.11 only |
| Install | ~2.5 GB + 10 GB weights | ~40 MB | ~6 GB weights |
| Cold start | 20-30 s warmup | under a second | 60-90 s |
| Bounding boxes | **yes, from DETR** | not available | yes, from YOLO |
| Works without internet | **yes** | no | yes |
| Deployable to ordinary hosting | no | **yes, incl. Vercel** | no |

Switch with `OMNIAI_PROFILE` in `.env`.

### offline — recommended

Everything runs locally. No key, no quota, no network at request time.

```bash
pip install -r requirements/offline.txt
```

```bash
python -m backend.models.download_models
```

Roughly 10 GB of weights, downloaded once. Check what is cached with
`python -m backend.models.download_models --check`.

Inference is CPU-bound by default; a document image takes about 30 s on six
cores. For a CUDA GPU, install torch from the PyTorch index and set
`OMNIAI_TORCH_DEVICE=cuda`.

### full — the original PaddleOCR stack

Requires Python 3.11; PaddlePaddle publishes no wheels for 3.12 or later.
Scanned PDFs additionally need [Poppler](https://poppler.freedesktop.org/) on
`PATH`, or `OMNIAI_POPPLER_PATH` pointing at its `bin` directory.

---

## Models

| Capability | `offline` | `lite` (Gemini) | `full` |
|---|---|---|---|
| Image classification | `microsoft/resnet-50` | `gemini-3.6-flash` + `gemini-3.5-flash-lite` ensemble | ResNet50 |
| Object detection | `facebook/detr-resnet-50` | `gemini-3.6-flash` (no boxes) | YOLOv8n |
| Wildlife detection | DETR at a lower threshold | `gemini-3.6-flash`, species-focused | YOLOv8x |
| Text extraction | EasyOCR (CRAFT + CRNN) | `gemini-3.1-flash-lite`, whole PDFs natively | PaddleOCR |
| Captioning | `blip-image-captioning-base` | `gemini-3.6-flash` | BLIP large |
| Translation | `nllb-200-distilled-600M` | `gemini-3.6-flash` | NLLB-200 |
| Sentiment | `twitter-roberta-base-sentiment` | `gemini-3.6-flash` | RoBERTa |
| Entities | `dslim/bert-base-NER` | `gemini-3.6-flash` | spaCy |
| Summarisation | `distilbart-cnn-12-6` | `gemini-3.6-flash` | Mistral |

`lite` falls back to Mistral (`mistral-medium-latest`, `mistral-ocr-latest`)
when only a Mistral key is set, or when `OMNIAI_HOSTED_PROVIDER=mistral`.

DETR emits COCO class names, so the label-refinement step described in the
paper applies unchanged in the `offline` profile — and unlike the hosted
backend, it returns real bounding boxes.

`mistral-large` is deliberately not used: it is absent from standard API tiers,
and selecting a model a key cannot reach fails every request.

### Classification ensemble

In `lite`, two vision models classify each image concurrently. When they agree
the answer is corroborated and confidence is raised; when they disagree a
reasoning model (`gemini-3.8-flash`) looks at the image and settles it. The interface shows which
models ran, what each said, and how a disagreement was resolved.

This is the same principle as label refinement: combine independent opinions
rather than fine-tune. It costs one or two extra API calls per image. Set
`OMNIAI_ENSEMBLE=false` for a single call. It disables itself automatically
while the key is being rate limited.

---

## How routing works

```
input
  |
  +-- inspect  ----->  image | document | text
  |
  +-- route    ----->  the auto-selector picks a model sequence
  |
  +-- execute
  |     images:     classify -> OCR -> detect -> caption
  |     documents:  extract -> OCR if scanned -> summarise
  |     text:       translate if needed -> sentiment + entities
  |
  +-- explain  ----->  plain-language account of every decision
```

Three decisions are worth calling out, because they are why the system is
faster and more specific than running every model on every input:

- **OCR runs before detection.** If an image turns out to be a screenshot or a
  scanned page, both object detection *and* captioning are skipped: OCR already
  has the content, and a caption model given a page of text produces a garbled
  paraphrase of the words it can partly read. On a document image this cuts
  end-to-end time from about 50 s to 30 s.
- **Animals re-route.** When the classifier reports a species, detection
  switches to its wildlife setting (YOLOv8x in `full`, DETR at a lower
  threshold in `offline`, a species-focused prompt in `lite`) and OCR is
  skipped, since wildlife photographs carry no readable text.
- **Labels are refined.** COCO calls every big cat `cat` and every antelope
  `cow`. The classifier's species label replaces the coarse one on the primary
  subject only, and the interface shows what changed.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The interface |
| `GET` | `/api/health` | Profile, capabilities, readiness |
| `POST` | `/api/analyze/file` | Analyse an upload, one JSON response |
| `POST` | `/api/analyze/text` | Analyse text, one JSON response |
| `POST` | `/api/analyze/file/stream` | Same, streaming progress over SSE |
| `POST` | `/api/analyze/text/stream` | Same, streaming progress over SSE |

```bash
curl -F "file=@photo.jpg" http://127.0.0.1:8000/api/analyze/file
```

The streaming endpoints emit a `progress` event as each stage starts and
finishes, then a final `result` or `failed` event. The interface uses these, so
the progress bar and the flow diagram reflect the pipeline rather than a timer.

---

## Tests

```bash
pip install -r requirements/dev.txt
```

```bash
python -m pytest
```

127 tests against a scripted backend: no API key, no GPU, no weights. Covers
routing, label refinement, the classification ensemble, profile and key
resolution, upload validation, rate limiting, security headers, secret
redaction, the SSE contract, and static checks on the stylesheet and template.

---

## Benchmark

```bash
python -m benchmarks.run
```

Measures the routing and label-refinement logic — the system-level
contribution — deterministically. Currently 120/120. Use `--markdown` for a
table to paste into the paper.

It does **not** measure model accuracy. See
[benchmarks/README.md](benchmarks/README.md) for what it establishes, and how
to reconcile it with the paper's published figures.

---

## Deployment

**Vercel** (how the live site runs): every push to `main` redeploys. The one
setting it needs is `OMNIAI_GEMINI_API_KEY`, added under Project, Settings,
Environment Variables with **Production** ticked, followed by a redeploy.

**Docker**, anywhere else:

```bash
docker compose up --build
```

See [docs/deployment.md](docs/deployment.md) for hosting options, a
troubleshooting table for a site stuck on "degraded", the single-worker
rate-limit caveat, and what to change before going public.

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — layers, request lifecycle, extension points
- [docs/design-system.md](docs/design-system.md) — tokens, typography, motion, the 3D field
- [docs/deployment.md](docs/deployment.md) — containers, hosting, scaling, hardening
- [docs/roadmap.md](docs/roadmap.md) — what comes next and why
- [benchmarks/README.md](benchmarks/README.md) — what the benchmark proves

---

## Security

- `.env` is git-ignored. Never commit an API key.
- Uploads are checked for extension, magic bytes, size and pixel count before
  any model sees them.
- Requests are rate limited per client, in memory, per process.
- A strict Content Security Policy is sent; no third-party scripts are loaded.
- Every log line, error body and UI warning passes through a redactor, so a
  credential cannot reach a terminal, a log file or the browser.
- All model output is written to the DOM with `textContent`, never as markup.

---

## Licence and attribution

Academic project. The pretrained models carry their own licences: ResNet50 and
YOLOv8 (AGPL-3.0 for Ultralytics), BLIP (BSD-3-Clause), NLLB-200
(CC-BY-NC-4.0, **non-commercial**), PaddleOCR (Apache-2.0). NLLB's
non-commercial clause applies to the `offline` and `full` profiles; check it
before any commercial deployment.
