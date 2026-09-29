"""The offline evaluation benchmark must pass as part of the standard suite.

This wires the Phase-4 evaluation framework (evaluation/) into the normal
offline test run so a regression in routing, geography, semantics, source
handling, safety, or citation grounding fails CI — without any network.
"""

from evaluation.cases import BENCHMARK
from evaluation.evaluators import Outcome
from evaluation.runner import run


def test_benchmark_is_non_trivial():
    # Guard against an accidentally empty benchmark silently "passing".
    assert len(BENCHMARK) >= 40
    kinds = {c.kind for c in BENCHMARK}
    for required in (
        "routing", "geography", "status", "transmission", "alert",
        "source_quality", "safety", "citation",
    ):
        assert required in kinds, f"benchmark missing kind {required!r}"


def test_all_case_ids_unique():
    ids = [c.id for c in BENCHMARK]
    assert len(ids) == len(set(ids)), "duplicate case ids in benchmark"


def test_benchmark_no_failures_and_no_unverified():
    report = run()
    # No case may FAIL.
    assert report.failed == 0, "; ".join(
        f"{r.case_id}: {r.detail}" for r in report.failures()
    )
    # Every case should be evaluable offline — nothing left UNVERIFIED.
    unverified = [r for r in report.results if r.outcome is Outcome.UNVERIFIED]
    assert not unverified, "; ".join(f"{r.case_id}: {r.detail}" for r in unverified)
    # Sanity: the vast majority are applicable (pass/fail), not NA.
    assert report.passed >= 40


def test_key_metric_accuracies_are_perfect_on_benchmark():
    report = run()
    for kind in (
        "routing", "geography", "status", "transmission", "alert",
        "source_quality", "safety", "citation",
    ):
        passed, applicable = report.accuracy(kind)
        assert applicable > 0, f"no applicable cases for {kind}"
        assert passed == applicable, f"{kind}: {passed}/{applicable} passed"
