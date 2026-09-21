"""Smart label refinement.

YOLO is trained on COCO, whose 80 classes are deliberately coarse: every big
cat is ``cat`` or ``dog``, every raptor is ``bird``, every antelope is ``cow``
or ``sheep``. ResNet50 is trained on ImageNet, which has species-level classes
but no localisation. Combining them -- taking YOLO's boxes and ResNet's label
-- is the refinement step described in section 4.A of the paper.

Previously this lived in ``backend/main.py``, mixed in with Flask routing.
"""

from __future__ import annotations

from typing import Any

from backend.utils.logger import get_logger

logger = get_logger("OmniAI.labels")

# Coarse COCO label -> the ImageNet species that legitimately refine it.
# All keys and values are lowercase; lookups normalise before comparing.
GENERIC_LABELS: dict[str, frozenset[str]] = {
    "dog": frozenset({
        "lion", "tiger", "cheetah", "leopard", "jaguar", "cougar", "puma",
        "wolf", "timber wolf", "red wolf", "white wolf", "coyote", "dingo",
        "fox", "red fox", "arctic fox", "kit fox", "grey fox", "fennec fox",
        "hyena", "wild dog", "african wild dog", "painted dog",
    }),
    "cat": frozenset({
        "lion", "tiger", "cheetah", "leopard", "snow leopard", "jaguar",
        "cougar", "puma", "lynx", "bobcat", "serval", "caracal", "ocelot",
        "panther", "black panther", "clouded leopard",
    }),
    "bird": frozenset({
        "eagle", "bald eagle", "golden eagle", "hawk", "falcon", "owl",
        "great grey owl", "great gray owl", "barn owl", "snowy owl",
        "peacock", "parrot", "macaw", "flamingo", "ostrich", "penguin",
        "pelican", "swan", "crane", "heron", "stork", "vulture", "toucan",
        "cockatoo", "woodpecker", "hummingbird", "kingfisher", "albatross",
        "jay", "magpie", "goose", "quail", "partridge", "ptarmigan",
    }),
    "horse": frozenset({"zebra", "donkey", "mule", "pony", "sorrel"}),
    "cow": frozenset({
        "buffalo", "bison", "ox", "yak", "water buffalo", "wildebeest", "gnu",
        "antelope", "gazelle", "impala", "springbok", "kudu", "eland",
        "hartebeest", "oryx", "gemsbok",
    }),
    "sheep": frozenset({
        "ram", "bighorn", "goat", "ibex", "mountain goat", "gazelle",
        "impala", "antelope", "springbok",
    }),
    "elephant": frozenset({
        "african elephant", "indian elephant", "asian elephant", "tusker",
    }),
    "bear": frozenset({
        "brown bear", "polar bear", "ice bear", "black bear",
        "american black bear", "panda", "giant panda", "grizzly",
        "sloth bear", "sun bear",
    }),
    "giraffe": frozenset({"giraffe", "okapi"}),
    "zebra": frozenset({"zebra"}),
    "deer": frozenset({
        "elk", "moose", "reindeer", "caribou", "antelope", "gazelle",
        "impala", "springbok", "gemsbok", "oryx", "red deer", "fallow deer",
    }),
    "monkey": frozenset({
        "gorilla", "orangutan", "chimpanzee", "baboon", "macaque", "lemur",
        "gibbon", "marmoset", "capuchin", "spider monkey", "siamang",
    }),
    "whale": frozenset({
        "killer whale", "orca", "blue whale", "humpback whale", "grey whale",
        "sperm whale",
    }),
    "seal": frozenset({"sea lion", "walrus", "fur seal", "dugong"}),
}


def refine_detections(
    detections: list[dict[str, Any]],
    classification_label: str,
    confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Promote the first coarse detection to the classifier's species label.

    Only the *primary* subject is refined. Secondary detections are left alone
    on purpose: in a hunting scene the predator and the prey are different
    species, and overwriting both with one label would be worse than leaving
    the coarse COCO names in place.

    Refined entries gain ``original_yolo_label`` and ``corrected: True`` so the
    interface can show what changed and why.
    """
    if not detections or not classification_label:
        return detections

    species = classification_label.strip().lower()
    refined: list[dict[str, Any]] = []
    already_refined = False

    for detection in detections:
        coarse = str(detection.get("label", "")).strip().lower()
        candidates = GENERIC_LABELS.get(coarse)

        # When the detector already names the species the classifier reports,
        # there is nothing to refine. Marking it corrected would make the
        # interface claim a change it did not make ("refined from 'zebra'").
        if coarse == species:
            refined.append(detection)
            already_refined = True
            continue

        if candidates and not already_refined and species in candidates:
            promoted = dict(detection)
            promoted["label"] = classification_label
            promoted["original_yolo_label"] = detection.get("label")
            promoted["corrected"] = True
            if confidence:
                promoted["classifier_confidence"] = round(float(confidence), 2)
            refined.append(promoted)
            already_refined = True
            logger.info("Refined primary subject: %s -> %s", coarse, classification_label)
        else:
            refined.append(detection)

    return refined


def refinement_summary(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe what refinement did, for the explainability panel."""
    corrected = [d for d in detections if d.get("corrected")]
    if not corrected:
        return {"applied": False, "count": 0, "changes": []}
    return {
        "applied": True,
        "count": len(corrected),
        "changes": [
            {"from": d.get("original_yolo_label"), "to": d.get("label")}
            for d in corrected
        ],
    }
