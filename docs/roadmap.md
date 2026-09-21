# Roadmap

Ordered by value per unit of effort. Items marked **paper** come from the future
work section of the OmniAI Cloud paper; the rest came out of building and
testing the current version.

---

## Near term

### Restore bounding boxes in the `lite` profile
`mistral-ocr-latest` already returns paragraph-level bounding boxes, and the OCR
panel could draw a text-region overlay today. Object localisation is the harder
half: a hosted vision model will not give reliable coordinates. The honest
options are a small detection model invoked as a separate hosted call, or
leaving `bbox_available: false` and saying so, which is the current behaviour.

### Shared state for several containers
Per-visitor limits are kept in each process. Serving from several containers
at once would move them to a shared store such as Redis.

### Result caching
Hash the upload and the routing decision, cache the result. Re-analysing the
same file is currently a full set of billed API calls. This is the single
largest cost reduction available.

### Batch and multi-file analysis
The pipeline handles one input per request. Accepting several and reporting them
as one comparative result would suit the wildlife-survey use case the paper
motivates, where images arrive in sets from one camera trap.

### Automatic document cleanup **paper**
The paper names scan quality as a limitation. Deskew, denoise and binarise
before OCR. OpenCV is already a `paper`-profile dependency, so this costs one
preprocessing function and measurably improves OCR on photographed pages.

---

## Medium term

### Learned routing **paper**
The auto-selector is a rule table. The paper proposes letting the system learn
its own routing decisions. The prerequisite is instrumentation: log every
routing decision with its outcome and latency. That log is also what would let
the routing claims be evaluated rather than asserted.

### Benchmark-driven evaluation **paper**
The reported figures — 95% wildlife accuracy, 132 species, the ablation table —
are not reproducible from this repository. A `benchmarks/` directory with a
fixed test set, a runner and a results table would make them so. This matters
more than any feature here: it converts the central claim from a statement into
something a reader can check.

Note for that work: the taxonomy now holds **791 unique species names**, and 127
of ImageNet's 1000 classes route to the wildlife path. The paper's "132 species"
figure should be reconciled against whichever of these it meant.

### Lighter model variants **paper**
The paper commits to variants for constrained environments. Concretely:
YOLOv8n instead of YOLOv8x, `blip-image-captioning-base` instead of `large`,
and `nllb-200-distilled-600M` quantised to int8. That is roughly 6 GB down to
under 2 GB, at a measurable accuracy cost worth publishing.

### Deeper explainability **paper**
Explanations today describe *what* ran. LIME or SHAP over the classifier, and
attention visualisation over the captioner, would explain *why* a prediction was
made — which is what the paper's citation of LIME actually promises.

### Accounts and per-user quotas
Sign-in and a per-account quota, for a deployment that needs to know who is
using it.

---

## Longer term

### Video and audio **paper**
The paper names both as the next modalities. Video is the natural fit: the
routing and refinement logic applies per keyframe, and wildlife footage is the
motivating case. Audio needs a genuinely new pipeline (transcription, then the
existing text path) and a new provider method.

### Asynchronous execution **paper**
Independent stages — classification and OCR do not depend on each other — could
run concurrently. On the `lite` profile, where each is a network round trip,
this is close to a straight halving of image-analysis latency. The
`ProgressEvent` contract already supports out-of-order completion; the pipeline
would move to `asyncio.gather` over a step group.

### Edge and sensor streams **paper**
Camera traps that classify on-device and upload only positives. This is a
different product shape, but the auto-selector is the right component to decide
what is worth uploading.

### Self-hosted inference backend
A third `Backend` implementation targeting a local vLLM or Triton server would
give the privacy of `paper` with the deployability of `lite`. The protocol
already supports it: one new class, one branch in `build_backend`.

---

## Explicitly not planned

**A second accent colour, dark mode, or a component library.** The interface has
one visual system, documented in `docs/design-system.md`. Additions should
justify themselves against it rather than accumulate.

**A frontend framework.** The page is server-rendered with about 700 lines of
vanilla JavaScript and no build step. React would add a toolchain, a bundle and
a deployment step to a single-page form. If the interface ever grows several
distinct views, revisit; not before.
