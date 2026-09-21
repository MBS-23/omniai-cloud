# OmniAI Cloud

**An intelligent unified multi-modal AI system with automatic model selection and explainability.**

Give it an image, a PDF, a Word document or a block of text. It works out what
it received, routes the input to only the models that suit it, and explains
every decision it made along the way.

Department of Artificial Intelligence and Machine Learning,
Sasi Institute of Technology and Engineering.

**Runs the models published in the paper**: ResNet50, YOLOv8n and YOLOv8x,
PaddleOCR, BLIP, NLLB-200, spaCy, RoBERTa and T5, downloaded from the Hugging
Face Hub and executed on the server itself. No API key, no rate limit.

---

## Quick start

With Docker (any OS; this is exactly what the hosted Space runs):

```bash
docker compose up --build
```

Open <http://localhost:7860>. The first build downloads about 6 GB of model
weights into the image; later builds reuse them. `GET /api/health` lists the
models in memory and whether startup pre-loading is still running.

Without Docker, on Python 3.10 to 3.13:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements/paper.txt
```

```bash
python -m backend.models.fetch_paper
```

```bash
python -m backend.main
```

---

## Runtime profiles

The same pipeline, the same routing logic and the same explainability output
run under every profile. Only model execution differs.

| | `paper` (recommended) | `offline` | `lite` |
|---|---|---|---|
| Models | **exactly the paper's** | paper's where they install on Python 3.14, substitutes elsewhere | a hosted API |
| Where models run | **this machine / the Space** | this machine | Gemini or Mistral |
| API key | **not required** | not required | required |
| Rate limits | **none** | none | yes, per key |
| Python | 3.10 to 3.13 (or Docker) | 3.10 to 3.14 | 3.10 to 3.14 |
| Memory with every model loaded | about 8 GB | about 7 GB | under 200 MB |
| Bounding boxes | **yes, from YOLOv8** | yes, from DETR | no |
| Deployable to | Docker hosts, Hugging Face Spaces | a workstation | serverless (Vercel) |

Choose with `OMNIAI_PROFILE`. Left unset, the app picks the most faithful
profile whose packages are installed. `full`, the profile's older name, still
selects `paper`.

`offline` exists because PaddlePaddle publishes no wheels for Python 3.14: it
runs natively on a 3.14 machine, with EasyOCR, DETR, BERT-NER and DistilBART
standing in for the four paper models that do not install there.

---

## Models

| Capability | Paper model (`paper` profile) | Hugging Face repository |
|---|---|---|
| Image classification | ResNet50, torchvision's ImageNet weights | `timm/resnet50.tv2_in1k` |
| Object detection | YOLOv8n | `Ultralytics/YOLOv8` |
| Wildlife detection | YOLOv8x | `Ultralytics/YOLOv8` |
| Text extraction | PaddleOCR 3, PP-OCRv5 (English, Telugu, Hindi) | `PaddlePaddle/*PP-OCRv5*` |
| Captioning | BLIP (large) | `Salesforce/blip-image-captioning-large` |
| Translation | NLLB-200 (distilled 600M) | `facebook/nllb-200-distilled-600M` |
| Sentiment | RoBERTa | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Named entities | spaCy `en_core_web_sm` | `spacy/en_core_web_sm` |
| Summarisation | T5 (base) | `google-t5/t5-base` |
| Language detection | langdetect | (library) |

Settings follow the thesis appendix: YOLOv8n at confidence 0.20, YOLOv8x at
0.10 for wildlife, BLIP with an 8-beam search, RoBERTa with the thesis's
sentiment calibration. Summarisation uses T5, as the conference paper
describes (its reference [9]). The thesis summarised through the Mistral API,
which is still available with `OMNIAI_SUMMARIZER=mistral` and a key.

The `paper` profile adds three things the original lacked: translation splits
long text at sentence boundaries instead of silently truncating at 512 tokens;
OCR stops trying language packs once one reads the image confidently; and
every model loads exactly once, under a lock, with the slow ones pre-loaded in
the background so the server answers from its first second.

### Hosted API (`lite`)

`lite` runs the same pipeline against Gemini (`gemini-3.6-flash` with a
two-model classification ensemble) or Mistral. It is the only profile small
enough for a serverless function, but it is bound by the provider's rate
limits and is not the paper's model stack.

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
- **Animals re-route.** When ResNet50 reports a species from the wildlife
  list, detection switches from YOLOv8n to the larger YOLOv8x, and OCR is
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

167 tests against a scripted backend: no API key, no GPU, no weights. Covers
routing, label refinement, the paper profile's OCR language search, text
chunking for NLLB and T5, CPU sizing and failure isolation, the classification
ensemble, profile and key resolution, upload validation, rate limiting,
security headers, secret redaction, the SSE contract, and static checks on the
stylesheet and template.

The models themselves are checked where they run: the Docker build loads every
one with the network switched off and runs OCR and detection on a sample image
(`python -m backend.models.verify_paper`), and fails if any of them cannot.

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

Academic project. The pretrained models carry their own licences, as published
on their Hugging Face model cards:

| Model | Licence |
|---|---|
| ResNet50 (`timm/resnet50.tv2_in1k`) | BSD-3-Clause |
| YOLOv8n, YOLOv8x (`Ultralytics/YOLOv8`) | AGPL-3.0 |
| PaddleOCR PP-OCRv5 | Apache-2.0 |
| BLIP (`Salesforce/blip-image-captioning-large`) | BSD-3-Clause |
| NLLB-200 (`facebook/nllb-200-distilled-600M`) | CC-BY-NC-4.0, **non-commercial** |
| RoBERTa sentiment (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | CC-BY-4.0 |
| spaCy `en_core_web_sm` | MIT |
| T5 (`google-t5/t5-base`) | Apache-2.0 |

NLLB's non-commercial clause and Ultralytics' AGPL (which requires the source
of a network service using it to be available, as it is here) both matter
before any commercial deployment.
