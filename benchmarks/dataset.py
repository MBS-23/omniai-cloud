"""Test cases for the routing and refinement benchmark.

What this measures, and what it does not
----------------------------------------
OmniAI Cloud's stated contribution is *system level*: automatic routing,
modular orchestration and label refinement over pretrained models that are
used as-is. This benchmark measures exactly that, and it is fully
deterministic -- no weights, no API key, no GPU.

It does **not** measure how accurately ResNet50 classifies or YOLOv8 detects.
Those are properties of the pretrained models, and measuring them needs a
labelled image corpus and the `full` profile. See `benchmarks/README.md`.

Cases are generated from the project's own data where possible, so the suite
cannot drift away from the shipped taxonomy, and hand-written where a specific
behaviour needs pinning down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RefinementCase:
    """One (classifier label, detector labels) pair and its expected outcome."""

    classifier_label: str
    detector_labels: tuple[str, ...]
    expected_first_label: str
    should_refine: bool
    note: str = ""


@dataclass(frozen=True)
class RoutingCase:
    kind: str                      # "image" | "document" | "text"
    value: str                     # filename or text
    expected_models: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class WildlifeCase:
    label: str
    is_animal: bool
    note: str = ""


# ----------------------------------------------------------------------
# 1. Label refinement
# ----------------------------------------------------------------------
# COCO's 80 classes are coarse. Each case pairs the coarse detector label with
# the ImageNet species the classifier would report, and states what the
# refined output must be.

REFINEMENT_POSITIVE: tuple[RefinementCase, ...] = (
    # Big cats: COCO has no lion, tiger or cheetah.
    RefinementCase("lion", ("cat",), "lion", True, "COCO has no lion class"),
    RefinementCase("tiger", ("cat",), "tiger", True),
    RefinementCase("cheetah", ("cat",), "cheetah", True),
    RefinementCase("leopard", ("cat",), "leopard", True),
    RefinementCase("snow leopard", ("cat",), "snow leopard", True),
    RefinementCase("jaguar", ("cat",), "jaguar", True),
    RefinementCase("lynx", ("cat",), "lynx", True),
    # Canids detected as dog.
    RefinementCase("timber wolf", ("dog",), "timber wolf", True),
    RefinementCase("red fox", ("dog",), "red fox", True),
    RefinementCase("arctic fox", ("dog",), "arctic fox", True),
    RefinementCase("coyote", ("dog",), "coyote", True),
    RefinementCase("hyena", ("dog",), "hyena", True),
    # Elephants: the regression that motivated the taxonomy fix.
    RefinementCase("african elephant", ("elephant",), "african elephant", True,
                   "capitalisation regression"),
    RefinementCase("indian elephant", ("elephant",), "indian elephant", True),
    # Bears.
    RefinementCase("polar bear", ("bear",), "polar bear", True),
    RefinementCase("brown bear", ("bear",), "brown bear", True),
    RefinementCase("american black bear", ("bear",), "american black bear", True),
    RefinementCase("giant panda", ("bear",), "giant panda", True),
    # Bovids and antelope, which COCO calls cow or sheep.
    RefinementCase("gazelle", ("cow",), "gazelle", True),
    RefinementCase("impala", ("cow",), "impala", True),
    RefinementCase("bison", ("cow",), "bison", True),
    RefinementCase("water buffalo", ("cow",), "water buffalo", True),
    RefinementCase("wildebeest", ("cow",), "wildebeest", True),
    RefinementCase("bighorn", ("sheep",), "bighorn", True),
    RefinementCase("ibex", ("sheep",), "ibex", True),
    # Equids.
    RefinementCase("zebra", ("horse",), "zebra", True),
    # Birds: COCO has exactly one bird class.
    RefinementCase("bald eagle", ("bird",), "bald eagle", True),
    RefinementCase("great grey owl", ("bird",), "great grey owl", True),
    RefinementCase("peacock", ("bird",), "peacock", True),
    RefinementCase("flamingo", ("bird",), "flamingo", True),
    RefinementCase("toucan", ("bird",), "toucan", True),
    RefinementCase("pelican", ("bird",), "pelican", True),
    RefinementCase("king penguin", ("bird",), "bird", False,
                   "not in the refinement map; stays 'bird' rather than guessing a species"),
    # Primates and marine mammals.
    RefinementCase("gorilla", ("monkey",), "gorilla", True),
    RefinementCase("orangutan", ("monkey",), "orangutan", True),
    RefinementCase("killer whale", ("whale",), "killer whale", True),
    RefinementCase("sea lion", ("seal",), "sea lion", True),
)

REFINEMENT_NEGATIVE: tuple[RefinementCase, ...] = (
    # The classifier and detector disagree about the kind of animal: refining
    # would replace a correct coarse label with a wrong specific one.
    RefinementCase("lion", ("bird",), "bird", False, "category mismatch"),
    RefinementCase("bald eagle", ("dog",), "dog", False, "category mismatch"),
    RefinementCase("zebra", ("bird",), "bird", False, "category mismatch"),
    # Not an animal at all.
    RefinementCase("laptop", ("cat",), "cat", False, "classifier says non-animal"),
    RefinementCase("coffee mug", ("dog",), "dog", False),
    # Already specific: COCO's own label is correct and needs no change.
    RefinementCase("zebra", ("zebra",), "zebra", False, "already species-level"),
    RefinementCase("giraffe", ("giraffe",), "giraffe", False),
    # Non-animal COCO classes must never be touched.
    RefinementCase("lion", ("car",), "car", False),
    RefinementCase("lion", ("person",), "person", False),
    RefinementCase("african elephant", ("truck",), "truck", False),
)

# Multi-subject scenes. Only the primary subject may be refined: in a hunting
# scene the predator and the prey are different species, and rewriting both
# with one label would be worse than leaving COCO's coarse names.
REFINEMENT_MULTI: tuple[RefinementCase, ...] = (
    RefinementCase("lion", ("cat", "cat"), "lion", True, "second cat must stay 'cat'"),
    RefinementCase("lion", ("cat", "zebra"), "lion", True, "prey must stay 'zebra'"),
    RefinementCase("african elephant", ("elephant", "elephant", "bird"),
                   "african elephant", True, "only the first elephant refines"),
    RefinementCase("cheetah", ("cat", "cow"), "cheetah", True, "gazelle stays 'cow'"),
)


# ----------------------------------------------------------------------
# 2. Routing
# ----------------------------------------------------------------------
IMAGE_MODELS = ("image_classification", "ocr_paddle", "object_detection", "blip_vision_language")

ROUTING_CASES: tuple[RoutingCase, ...] = (
    RoutingCase("image", "photo.jpg", IMAGE_MODELS),
    RoutingCase("image", "screenshot.PNG", IMAGE_MODELS, "extension case-insensitive"),
    RoutingCase("image", "render.webp", IMAGE_MODELS),
    RoutingCase("document", "report.pdf", ("document_ai",)),
    RoutingCase("document", "notes.DOCX", ("document_ai",)),
    RoutingCase("text", "This is an English sentence for analysis.", ("text_analytics",),
                "English needs no translation"),
    RoutingCase("text", "కృత్రిమ మేధస్సు ఈరోజు అనేక రంగాలలో ఉపయోగించబడుతోంది.",
                ("translation_ai", "text_analytics"), "Telugu adds translation"),
    RoutingCase("text", "L'intelligence artificielle transforme de nombreux secteurs.",
                ("translation_ai", "text_analytics"), "French adds translation"),
    RoutingCase("text", "künstliche Intelligenz verändert viele Branchen heute.",
                ("translation_ai", "text_analytics"), "German adds translation"),
)

UNSUPPORTED_INPUTS: tuple[str, ...] = (
    "archive.zip", "payload.exe", "video.mp4", "sheet.xlsx", "song.mp3", "page.html",
)


# ----------------------------------------------------------------------
# 3. Wildlife gate
# ----------------------------------------------------------------------
# Hand-picked cases that pin down behaviour the generated set cannot.
WILDLIFE_CASES: tuple[WildlifeCase, ...] = (
    # The four capitalised ImageNet labels that silently failed before the fix.
    WildlifeCase("African elephant", True, "regression"),
    WildlifeCase("Indian elephant", True, "regression"),
    WildlifeCase("Arctic fox", True, "regression"),
    WildlifeCase("American black bear", True, "regression"),
    # Case and whitespace handling.
    WildlifeCase("lion", True),
    WildlifeCase("LION", True),
    WildlifeCase("  Lion  ", True),
    # Non-animals.
    WildlifeCase("laptop", False),
    WildlifeCase("sports car", False),
    WildlifeCase("coffee mug", False),
    WildlifeCase("", False),
)


def imagenet_labels() -> list[str]:
    """The 1000 ImageNet class names the classifier can emit."""
    path = PROJECT_ROOT / "backend" / "models" / "imagenet_classes.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
