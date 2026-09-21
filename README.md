# OmniAI Cloud

**An intelligent unified multi-modal AI system with automatic model selection and explainability.**

Give it an image, a PDF, a Word document or a block of text. It works out what
it received, routes the input to only the models that suit it, and explains
every decision it made along the way.

Department of Artificial Intelligence and Machine Learning,
Sasi Institute of Technology and Engineering.

---

## Quick start

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere), then:

```bash
pip install -r requirements/lite.txt
```

Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Set `OMNIAI_MISTRAL_API_KEY` in `.env` to a key from
[console.mistral.ai](https://console.mistral.ai/api-keys), then run:

```bash
python -m backend.main
```

Open <http://127.0.0.1:8000>.

> **Check your key can actually do inference.** A key that lists models is not
> necessarily allowed to call them. If chat requests return `429` while
> `/v1/ocr` returns `200`, the key is valid but its chat quota is exhausted or
> not yet enabled — check limits and billing in the Mistral console.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $OMNIAI_MISTRAL_API_KEY" -H "Content-Type: application/json" -d '{"model":"mistral-small-latest","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

---

## Runtime profiles

The same pipeline, the same routing logic and the same explainability output
run under every profile. Only model execution differs.

| | `offline` (default) | `lite` | `full` |
|---|---|---|---|
| Where models run | **This machine** | Mistral API | This machine |
| API key | **not required** | required | not required |
| Rate limits | **none** | yes, per key | none |
| Python | 3.11 to 3.14 | 3.11 to 3.14 | 3.11 only |
| Install | ~2.5 GB + 10 GB weights | ~40 MB | ~6 GB weights |
| Cold start | 20-30 s warmup | under a second | 60-90 s |
| Bounding boxes | **yes, from DETR** | not available | yes, from YOLO |
| Works without internet | **yes** | no | yes |
| Deployable to ordinary hosting | no | yes | no |

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

| Capability | `offline` | `lite` | `full` |
|---|---|---|---|
| Image classification | `microsoft/resnet-50` | Mistral vision, 2-model ensemble | ResNet50 |
| Object detection | `facebook/detr-resnet-50` | Mistral vision (no boxes) | YOLOv8n |
| Wildlife detection | DETR at a lower threshold | Mistral vision | YOLOv8x |
| Text extraction | EasyOCR (CRAFT + CRNN) | `mistral-ocr-latest` | PaddleOCR |
| Captioning | `blip-image-captioning-base` | Mistral vision | BLIP large |
| Translation | `nllb-200-distilled-600M` | Mistral | NLLB-200 |
| Sentiment | `twitter-roberta-base-sentiment` | Mistral | RoBERTa |
| Entities | `dslim/bert-base-NER` | Mistral | spaCy |
| Summarisation | `distilbart-cnn-12-6` | Mistral | Mistral |

DETR emits COCO class names, so the label-refinement step described in the
paper applies unchanged in the `offline` profile — and unlike the hosted
backend, it returns real bounding boxes.

`mistral-large` is deliberately not used: it is absent from standard API tiers,
and selecting a model a key cannot reach fails every request.

### Classification ensemble

In `lite`, two vision models classify each image concurrently. When they agree
the answer is corroborated and confidence is raised; when they disagree a
reasoning model looks at the image and settles it. The interface shows which
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
- **Animals re-route.** When the classifier reports a species, detection moves
  from YOLOv8n to YOLOv8x and OCR is skipped, since wildlife photographs carry
  no readable text.
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

88 tests against a scripted backend: no API key, no GPU, no weights. Covers
routing, label refinement, upload validation, rate limiting, security headers,
secret redaction, the SSE contract, and static checks on the stylesheet and
template.

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

```bash
docker compose up --build
```

See [docs/deployment.md](docs/deployment.md) for hosting options, the
single-worker rate-limit caveat, and what to change before going public.

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
non-commercial clause applies to the `full` profile; check it before any
commercial deployment.
