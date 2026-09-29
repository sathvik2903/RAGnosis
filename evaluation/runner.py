"""Deterministic, offline evaluation runner (Phase-4 §8).

Run with::

    python evaluation/run_evaluation.py

It evaluates every case in :data:`evaluation.cases.BENCHMARK`, prints totals,
per-category results, and any failures, then exits non-zero if any case FAILED.
NOT_APPLICABLE and UNVERIFIED never count as passes and never fail the run on
their own, but they are reported so nothing is hidden.

No API keys. No live network. Fully reproducible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .cases import BENCHMARK, EvalCase
from .evaluators import CaseResult, Outcome, evaluate_case


@dataclass(frozen=True)
class EvaluationReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def count(self, outcome: Outcome) -> int:
        return sum(1 for r in self.results if r.outcome is outcome)

    @property
    def passed(self) -> int:
        return self.count(Outcome.PASS)

    @property
    def failed(self) -> int:
        return self.count(Outcome.FAIL)

    @property
    def not_applicable(self) -> int:
        return self.count(Outcome.NOT_APPLICABLE)

    @property
    def unverified(self) -> int:
        return self.count(Outcome.UNVERIFIED)

    @property
    def ok(self) -> bool:
        """True only when nothing FAILED and nothing is UNVERIFIED.

        NOT_APPLICABLE is a legitimate, decided outcome (a case that does not
        apply to a dimension) and does not fail the run. UNVERIFIED means the
        contract could NOT be decided (an evaluator raised, or a required check
        could not run); reporting overall success while any case is unverified
        would silently turn "we don't know" into "it passed", so it is not ok.
        """
        return self.failed == 0 and self.unverified == 0

    def by_category(self) -> dict[str, dict[str, int]]:
        cats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "fail": 0, "not_applicable": 0, "unverified": 0}
        )
        for r in self.results:
            cats[r.category][r.outcome.value] += 1
        return dict(cats)

    def by_kind(self) -> dict[str, dict[str, int]]:
        kinds: dict[str, dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "fail": 0, "not_applicable": 0, "unverified": 0}
        )
        for r in self.results:
            kinds[r.kind][r.outcome.value] += 1
        return dict(kinds)

    def accuracy(self, kind: str) -> tuple[int, int]:
        """(passed, applicable) for a kind, where applicable = pass + fail."""
        passed = sum(1 for r in self.results if r.kind == kind and r.outcome is Outcome.PASS)
        applicable = sum(
            1 for r in self.results
            if r.kind == kind and r.outcome in (Outcome.PASS, Outcome.FAIL)
        )
        return passed, applicable

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if r.outcome is Outcome.FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "not_applicable": self.not_applicable,
            "unverified": self.unverified,
            "by_category": self.by_category(),
            "by_kind": self.by_kind(),
            "results": [r.to_dict() for r in self.results],
        }


def run(cases: list[EvalCase] | None = None) -> EvaluationReport:
    """Evaluate ``cases`` (default: the full benchmark) fully offline.

    Deliberate source-failure cases would otherwise emit WARNING logs from the
    health layer; we quiet that logger for the duration so the report is clean
    (the failures are still asserted by the evaluators themselves).
    """
    cases = cases if cases is not None else BENCHMARK
    health_logger = logging.getLogger("ragnosis.health")
    prev_level = health_logger.level
    health_logger.setLevel(logging.ERROR)
    try:
        return EvaluationReport(results=[evaluate_case(c) for c in cases])
    finally:
        health_logger.setLevel(prev_level)


def _fmt_pct(passed: int, applicable: int) -> str:
    if applicable == 0:
        return "n/a"
    return f"{100.0 * passed / applicable:.1f}% ({passed}/{applicable})"


def format_report(report: EvaluationReport) -> str:
    lines: list[str] = []
    lines.append("RAGnosis offline evaluation")
    lines.append("=" * 72)
    lines.append(f"Total cases:     {report.total}")
    lines.append(f"  passed:        {report.passed}")
    lines.append(f"  failed:        {report.failed}")
    lines.append(f"  not applicable:{report.not_applicable}")
    lines.append(f"  unverified:    {report.unverified}")
    lines.append("-" * 72)
    lines.append("Per-category (pass/fail/na/unverified):")
    for cat, c in sorted(report.by_category().items()):
        lines.append(
            f"  {cat:<16} {c['pass']:>3} / {c['fail']:>3} / "
            f"{c['not_applicable']:>3} / {c['unverified']:>3}"
        )
    lines.append("-" * 72)
    lines.append("Key metrics (accuracy over applicable cases):")
    metric_kinds = [
        ("routing", "routing accuracy"),
        ("geography", "geographic relevance"),
        ("status", "status classification"),
        ("transmission", "transmission classification"),
        ("alert", "alert classification"),
        ("source_quality", "source precedence/conflict/freshness"),
        ("safety", "personal-medical safety"),
        ("citation", "citation grounding"),
    ]
    for kind, label in metric_kinds:
        passed, applicable = report.accuracy(kind)
        lines.append(f"  {label:<38} {_fmt_pct(passed, applicable)}")
    if report.failures():
        lines.append("-" * 72)
        lines.append("FAILURES:")
        for r in report.failures():
            lines.append(f"  [{r.category}] {r.case_id}: {r.detail}")
    if report.unverified:
        lines.append("-" * 72)
        lines.append("UNVERIFIED:")
        for r in report.results:
            if r.outcome is Outcome.UNVERIFIED:
                lines.append(f"  {r.case_id}: {r.detail}")
    lines.append("-" * 72)
    if report.ok:
        result = "OK (no failures, no unverified)"
    elif report.failed:
        result = f"FAILURES PRESENT ({report.failed} failed, {report.unverified} unverified)"
    else:
        result = f"UNVERIFIED ({report.unverified} unverified, 0 failed) -> not OK"
    lines.append("RESULT: " + result)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    report = run()
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
