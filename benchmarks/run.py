"""Reproducible benchmark for OmniAI Cloud's routing and label refinement.

    python -m benchmarks.run              # table to stdout
    python -m benchmarks.run --markdown   # markdown, for the paper
    python -m benchmarks.run --json       # machine-readable

Needs no API key, no GPU and no model weights: every metric here is a property
of the routing code, which is what the paper claims as its contribution.

Read `benchmarks/README.md` for what this does and does not establish.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from backend.agent.auto_selector import analyze_input, auto_model_selector
from backend.agent.taxonomy import SPECIES_COUNT, is_animal_image
from backend.core.labels import GENERIC_LABELS, refine_detections
from benchmarks.dataset import (
    REFINEMENT_MULTI,
    REFINEMENT_NEGATIVE,
    REFINEMENT_POSITIVE,
    ROUTING_CASES,
    UNSUPPORTED_INPUTS,
    WILDLIFE_CASES,
    imagenet_labels,
)


@dataclass
class Result:
    name: str
    passed: int = 0
    total: int = 0
    failures: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def record(self, ok: bool, description: str = "") -> None:
        self.total += 1
        if ok:
            self.passed += 1
        elif description:
            self.failures.append(description)

    @property
    def rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0


# ======================================================================
def bench_wildlife_gate() -> Result:
    """Does the animal gate fire for the labels the classifier emits?"""
    result = Result("Wildlife gate")

    for case in WILDLIFE_CASES:
        got = is_animal_image(case.label)
        result.record(got == case.is_animal, f"{case.label!r}: expected {case.is_animal}, got {got}")

    # Coverage across the real ImageNet label set the classifier can return.
    labels = imagenet_labels()
    matched = [label for label in labels if is_animal_image(label)]
    result.extra = {
        "taxonomy_entries": SPECIES_COUNT,
        "imagenet_classes": len(labels),
        "imagenet_routed_to_wildlife": len(matched),
        "coverage_pct": round(len(matched) / len(labels) * 100, 1),
    }
    return result


def bench_refinement() -> Result:
    """Does refinement promote coarse labels correctly, and leave the rest alone?"""
    result = Result("Label refinement")

    for case in (*REFINEMENT_POSITIVE, *REFINEMENT_NEGATIVE, *REFINEMENT_MULTI):
        detections = [{"label": lab, "confidence": 90.0} for lab in case.detector_labels]
        refined = refine_detections(detections, case.classifier_label, 95.0)

        first = refined[0]["label"]
        was_refined = bool(refined[0].get("corrected"))

        ok = first == case.expected_first_label and was_refined == case.should_refine
        result.record(
            ok,
            f"{case.classifier_label!r} + {case.detector_labels} -> {first!r} "
            f"(refined={was_refined}); expected {case.expected_first_label!r} "
            f"(refined={case.should_refine}){' :: ' + case.note if case.note else ''}",
        )

    # Only the primary subject may change, never a secondary one.
    secondary_preserved = Result("secondary")
    for case in REFINEMENT_MULTI:
        detections = [{"label": lab, "confidence": 90.0} for lab in case.detector_labels]
        refined = refine_detections(detections, case.classifier_label, 95.0)
        untouched = all(not d.get("corrected") for d in refined[1:])
        secondary_preserved.record(untouched)

    result.extra = {
        "positive_cases": len(REFINEMENT_POSITIVE),
        "negative_cases": len(REFINEMENT_NEGATIVE),
        "multi_subject_cases": len(REFINEMENT_MULTI),
        "coarse_labels_mapped": len(GENERIC_LABELS),
        "secondary_subjects_preserved_pct": round(secondary_preserved.rate, 1),
    }
    return result


def bench_routing() -> Result:
    """Does the agent select the right model sequence for each input?"""
    result = Result("Model routing")

    for case in ROUTING_CASES:
        if case.kind == "text":
            metadata = analyze_input(text=case.value)
        else:
            metadata = analyze_input(file_path=case.value)

        got = tuple(auto_model_selector(metadata)["models"])
        ok = got == case.expected_models
        result.record(ok, f"{case.value[:40]!r} -> {got}; expected {case.expected_models}")

    for filename in UNSUPPORTED_INPUTS:
        metadata = analyze_input(file_path=filename)
        ok = metadata["input_type"] == "unknown"
        result.record(ok, f"{filename!r} should be unknown, got {metadata['input_type']!r}")

    return result


def bench_selectivity() -> Result:
    """How many model invocations does routing avoid versus running everything?

    This is the mechanism behind the paper's latency claim: the auto-selector
    does not make models faster, it stops them being called.
    """
    result = Result("Routing selectivity")

    every_model = 7  # classification, detection, wildlife, OCR, caption, translation, analytics
    selected_total = 0
    naive_total = 0
    per_type: dict[str, int] = {}

    for case in ROUTING_CASES:
        metadata = (
            analyze_input(text=case.value) if case.kind == "text"
            else analyze_input(file_path=case.value)
        )
        count = len(auto_model_selector(metadata)["models"])
        selected_total += count
        naive_total += every_model
        per_type.setdefault(case.kind, count)
        result.record(count < every_model, f"{case.kind} selected {count} of {every_model}")

    saved = (1 - selected_total / naive_total) * 100 if naive_total else 0.0
    result.extra = {
        "models_available": every_model,
        "invocations_with_routing": selected_total,
        "invocations_without_routing": naive_total,
        "reduction_pct": round(saved, 1),
        "models_per_input_type": per_type,
    }
    return result


def bench_determinism() -> Result:
    """The same input must produce the same routing decision every time."""
    result = Result("Determinism")

    for case in ROUTING_CASES:
        metadata = (
            analyze_input(text=case.value) if case.kind == "text"
            else analyze_input(file_path=case.value)
        )
        decisions = {tuple(auto_model_selector(metadata)["models"]) for _ in range(25)}
        result.record(len(decisions) == 1, f"{case.value[:40]!r} produced {len(decisions)} decisions")

    # Refinement must not depend on call order or mutate its input.
    for case in REFINEMENT_POSITIVE[:12]:
        detections = [{"label": lab, "confidence": 90.0} for lab in case.detector_labels]
        outputs = {refine_detections(detections, case.classifier_label, 95.0)[0]["label"]
                   for _ in range(25)}
        result.record(len(outputs) == 1, f"{case.classifier_label!r} produced {len(outputs)} outputs")
        result.record(
            all(not d.get("corrected") for d in detections),
            f"{case.classifier_label!r}: refine_detections mutated its input",
        )

    return result


def bench_throughput() -> Result:
    """Routing overhead per input, so it can be excluded from model latency."""
    result = Result("Routing overhead")

    samples = []
    for case in ROUTING_CASES:
        start = time.perf_counter()
        for _ in range(200):
            metadata = (
                analyze_input(text=case.value) if case.kind == "text"
                else analyze_input(file_path=case.value)
            )
            auto_model_selector(metadata)
        samples.append((time.perf_counter() - start) / 200 * 1000)

    average = sum(samples) / len(samples)
    result.record(average < 50.0, f"routing averaged {average:.2f} ms")
    result.extra = {
        "mean_ms": round(average, 3),
        "max_ms": round(max(samples), 3),
        "note": "language detection dominates; file routing is sub-millisecond",
    }
    return result


BENCHMARKS = (
    bench_wildlife_gate,
    bench_refinement,
    bench_routing,
    bench_selectivity,
    bench_determinism,
    bench_throughput,
)


# ======================================================================
def render_text(results: list[Result]) -> str:
    lines = ["", "=" * 74, "OmniAI Cloud — routing and refinement benchmark", "=" * 74, ""]
    lines.append(f"{'Benchmark':<24}{'Passed':>10}{'Total':>8}{'Rate':>9}")
    lines.append("-" * 74)
    for r in results:
        lines.append(f"{r.name:<24}{r.passed:>10}{r.total:>8}{r.rate:>8.1f}%")
    lines.append("-" * 74)

    total_passed = sum(r.passed for r in results)
    total = sum(r.total for r in results)
    lines.append(f"{'OVERALL':<24}{total_passed:>10}{total:>8}{total_passed / total * 100:>8.1f}%")
    lines.append("")

    for r in results:
        if r.extra:
            lines.append(f"{r.name}:")
            for key, value in r.extra.items():
                lines.append(f"    {key.replace('_', ' ')}: {value}")
            lines.append("")

    failures = [(r.name, f) for r in results for f in r.failures]
    if failures:
        lines.append(f"FAILURES ({len(failures)}):")
        for name, failure in failures:
            lines.append(f"  [{name}] {failure}")
        lines.append("")

    return "\n".join(lines)


def render_markdown(results: list[Result]) -> str:
    lines = [
        "## Routing and refinement benchmark",
        "",
        "Deterministic; no API key, GPU or model weights required.",
        "Reproduce with `python -m benchmarks.run`.",
        "",
        "| Benchmark | Cases | Passed | Rate |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(f"| {r.name} | {r.total} | {r.passed} | {r.rate:.1f}% |")

    total_passed = sum(r.passed for r in results)
    total = sum(r.total for r in results)
    lines.append(f"| **Overall** | **{total}** | **{total_passed}** | **{total_passed / total * 100:.1f}%** |")
    lines.append("")

    for r in results:
        if not r.extra:
            continue
        lines.append(f"**{r.name}**")
        lines.append("")
        for key, value in r.extra.items():
            lines.append(f"- {key.replace('_', ' ').capitalize()}: `{value}`")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit markdown")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    results = [benchmark() for benchmark in BENCHMARKS]

    if args.json:
        print(json.dumps(
            {
                r.name: {
                    "passed": r.passed, "total": r.total,
                    "rate": round(r.rate, 2), "failures": r.failures, "extra": r.extra,
                }
                for r in results
            },
            indent=2,
        ))
    elif args.markdown:
        print(render_markdown(results))
    else:
        print(render_text(results))

    # Non-zero exit when anything failed, so CI can gate on this.
    return 0 if all(r.passed == r.total for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
