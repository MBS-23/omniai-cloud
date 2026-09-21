# Architecture

## The three layers

The paper describes OmniAI Cloud as a three-layer system. The code is organised
the same way, so the mapping between the two is direct.

```
┌──────────────────────────────────────────────────────────┐
│  Interface layer                                         │
│  backend/templates/  backend/static/  backend/routes/    │
│  Server-rendered page, JSON API, SSE progress stream     │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│  Intelligence layer                                      │
│  backend/agent/      input analysis, model selection     │
│  backend/core/       orchestration, label refinement     │
│  Decides WHAT runs. Knows nothing about HOW.             │
└──────────────────────────┬───────────────────────────────┘
                           │  Backend protocol
┌──────────────────────────▼───────────────────────────────┐
│  Execution layer                                         │
│  backend/providers/paper.py    the paper's models, local │
│  backend/providers/offline.py  local, Python 3.14 subs   │
│  backend/providers/gemini.py   hosted: Gemini API        │
│  backend/providers/remote.py   hosted: Mistral API       │
│  backend/modules/              the model code itself     │
└──────────────────────────────────────────────────────────┘
```

The boundary between the middle and bottom layer is
`backend/providers/base.py`. The intelligence layer only ever calls the seven
methods on the `Backend` protocol. That is what allows the same routing rules,
the same refinement step and the same explanation text to run whether the models
are on this machine or behind an API.

## Request lifecycle

```
POST /api/analyze/file/stream
  │
  ├─ rate limit          sliding window, per client
  ├─ validate            extension → magic bytes → size → pixel count
  ├─ persist             uuid-prefixed name in a private temp dir
  │
  ├─ worker thread ─────────────────────────────┐
  │    run_file_analysis()                      │  emits ProgressEvent
  │      inspect → route → execute → explain    │  onto a queue
  │                                             │
  ├─ generator ◄────────────────────────────────┘
  │    formats each event as SSE, yields it
  │
  ├─ final event         result, or failed
  └─ finally             delete the upload
```

The pipeline is synchronous and blocking, so it runs on a worker thread while
the response generator drains an event queue. This is why progress is real: each
event is emitted at the moment the corresponding stage actually changes state.

## Module map

| Path | Responsibility |
|---|---|
| `config.py` | All environment reading. Nothing else calls `os.getenv`. |
| `app.py` | Application factory, error handlers, security headers |
| `main.py` | Entry point and WSGI target |
| `agent/auto_selector.py` | Input classification, model sequence selection |
| `agent/taxonomy.py` | 791 species names, five groups |
| `core/pipeline.py` | Stage orchestration and progress events |
| `core/labels.py` | COCO → ImageNet label refinement |
| `providers/base.py` | The `Backend` protocol and result shapes |
| `providers/paper.py` | The paper profile: adapters over `modules/`, background pre-loading |
| `providers/offline.py` | Local substitutes for Python 3.14 |
| `providers/gemini.py` | Gemini implementations (lite) |
| `providers/remote.py` | Mistral implementations |
| `providers/mistral_client.py` | HTTP, retries, JSON extraction |
| `routes/api.py` | JSON and SSE endpoints |
| `routes/views.py` | The one page |
| `modules/` | The paper's model code, one file per model, as in the thesis appendix |
| `modules/hub.py` | Which Hugging Face repository each paper model comes from |
| `modules/lazy.py` | Load-once model handles (the paper's singleton pattern), under a lock |
| `models/fetch_paper.py` | Download, and optionally verify, every paper model |
| `utils/` | Files, PDFs, DOCX, logging, languages, security |

## Design decisions worth knowing

**Lazy imports in the local backend.** Every PyTorch import sits inside a
method, not at module scope. Without that, the lite profile could not run on a
machine with no PyTorch installed, because Python would import the module graph
regardless of which backend was selected.

**Load once, under a lock, in the background.** The original code guarded model
loading with a bare module-level boolean, read and written without
synchronisation, inside a route handler. Two concurrent first requests could
both see `False` and both begin loading several gigabytes. Each model is now a
`Lazy` handle (`modules/lazy.py`) that builds exactly once behind a lock. The
paper profile pre-loads them on a background thread, so the server answers
health checks from its first second while the weights load. Warmup happens once, at application
construction, behind a lock.

**Progress as data, not decoration.** `ProgressEvent` carries a stage name, a
status, a percentage and a human detail string. The same events feed the SSE
stream and the test assertions, so a change to the pipeline that breaks the
displayed progress fails a test.

**Failures degrade, they do not cascade.** Each stage catches
`BackendUnavailable` and records a warning rather than aborting. A captioning
outage still returns classification and detection, and the response says
explicitly what did not run.

## Extending it

**Adding a model** means implementing the relevant `Backend` method in one or
both providers, then adding a branch in `core/pipeline.py` and a name in
`auto_selector.auto_model_selector`. The interface renders whatever keys appear
under `results`, so a new panel needs a case in `renderPanels` in
`static/js/omniai.js`.

**Adding a third backend** — a self-hosted inference server, say — means one new
class implementing `Backend` and one branch in `providers/build_backend`.
Nothing in the intelligence layer changes.

## Known limitations

- The rate limiter holds state in process memory, so it weakens in proportion to
  worker count. See `docs/deployment.md`.
- `lite` cannot return bounding boxes. The response sets
  `bbox_available: false` and the interface says so rather than drawing boxes it
  cannot justify.
- Scanned PDFs are rasterised at 200 DPI and capped at 30 pages.
- Translation truncates at 512 tokens, an NLLB constraint.
