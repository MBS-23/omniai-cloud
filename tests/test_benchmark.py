"""Gate CI on the benchmark.

The benchmark is deterministic, so a regression in routing or refinement is a
test failure rather than something noticed later by reading a report.
"""

from __future__ import annotations

from benchmarks.run import BENCHMARKS


def test_every_benchmark_passes_completely():
    results = [benchmark() for benchmark in BENCHMARKS]

    failing = [
        f"{r.name}: {r.passed}/{r.total}\n    " + "\n    ".join(r.failures)
        for r in results
        if r.passed != r.total
    ]
    assert not failing, "benchmark regressions:\n" + "\n".join(failing)


def test_wildlife_coverage_does_not_regress():
    """The elephant regression cost four ImageNet classes; guard the count."""
    from benchmarks.run import bench_wildlife_gate

    extra = bench_wildlife_gate().extra
    assert extra["imagenet_routed_to_wildlife"] >= 127
    assert extra["taxonomy_entries"] >= 791


def test_routing_still_avoids_most_model_invocations():
    """The latency claim rests on this; a regression would be silent."""
    from benchmarks.run import bench_selectivity

    assert bench_selectivity().extra["reduction_pct"] >= 60.0
