#!/usr/bin/env python3
"""Final submission verification report (Phase-6 §17).

Runs the offline component and end-to-end contract benchmarks and emits ONE
report in both machine-readable (JSON) and human-readable (Markdown) form. Every
metric carries a numerator, a denominator, and a status. Live checks (live model
contract validation, live health validation) are reported as ``UNVERIFIED`` when
credentials/network are absent — never fabricated.

Usage:
    python evaluation/final_report.py                 # print Markdown
    python evaluation/final_report.py --json out.json # also write JSON

This is an end-to-end CONTRACT pass rate over deterministic fixtures. It is NOT
a measure of LLM factual accuracy, medical accuracy, or diagnostic accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.agent_runner import DIMENSION_ORDER  # noqa: E402
from evaluation.agent_runner import run as run_agent  # noqa: E402
from evaluation.runner import run as run_component  # noqa: E402


def _status(num: int, den: int) -> str:
    if den == 0:
        return "NOT_APPLICABLE"
    return "PASS" if num == den else "FAIL"


def _metric(num: int, den: int) -> dict:
    return {"numerator": num, "denominator": den, "status": _status(num, den)}


def build_report() -> dict:
    comp = run_component()
    e2e = run_agent()

    dims = {}
    for d in DIMENSION_ORDER:
        p, a = e2e.dimension_accuracy(d)
        dims[d] = _metric(p, a)

    # Live checks: only verified if the environment provides them. We do not
    # attempt network here; absence of credentials => UNVERIFIED, not fabricated.
    live_model = {
        "status": "UNVERIFIED",
        "reason": "no generation credentials in this environment"
        if not os.getenv("COHERE_API_KEY")
        else "run `python evaluation/run_agent_evaluation.py --live` to verify",
    }
    live_health = {
        "status": "UNVERIFIED",
        "reason": "outbound HTTPS blocked / not attempted here; run "
        "`python evaluation/live_smoke.py --live` where egress is available",
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": "end_to_end_contract_pass_rate",
        "disclaimer": (
            "Contract pass rate over deterministic fixtures. This measures "
            "whether the system honours its declared behavioural contracts "
            "(routing, grounding, provenance, uncertainty, safety, geographic "
            "honesty). It is NOT a measure of LLM factual accuracy, medical or "
            "diagnostic accuracy, and NOT evidence of real-time or exhaustive "
            "real-world disease surveillance. Absence of a finding is never "
            "proof of absence of disease."
        ),
        "verification_scope": {
            "offline_contract_verification": (
                "Deterministic component and end-to-end benchmarks run against "
                "injected fakes (no network, no credentials). Proves the code "
                "honours its contracts; every metric below is of this kind."
            ),
            "live_dependency_verification": (
                "Whether real external dependencies (the generation model, live "
                "health feeds) were actually contacted. Reported separately as "
                "LIVE VERIFIED / UNVERIFIED and never inferred from offline "
                "results. UNVERIFIED is not a success."
            ),
            "semantic_truth_limitation": (
                "No metric here evaluates whether an answer is factually or "
                "medically correct, nor whether current outbreak status in the "
                "real world matches any claim. Grounding checks verify fidelity "
                "to the provided fixtures/evidence only, not truth in the world."
            ),
        },
        "component_benchmark": {
            "total": comp.total, "passed": comp.passed, "failed": comp.failed,
            "unverified": comp.unverified, "ok": comp.ok,
            "metric": _metric(comp.passed, comp.total),
        },
        "end_to_end_benchmark": {
            "total": e2e.total, "passed": e2e.passed, "failed": e2e.failed,
            "unverified": e2e.unverified, "ok": e2e.ok,
            "metric": _metric(e2e.passed, e2e.total),
        },
        "dimensions": dims,
        "live_model_contract_validation": live_model,
        "live_health_validation": live_health,
    }


_DIM_LABELS = {
    "tool_selection": "Tool selection / routing",
    "location_handling": "Location handling",
    "live_data_state": "Live-data state honesty",
    "grounding": "Answer grounding (fixture fidelity)",
    "citation": "Citation grounding",
    "provenance": "Provenance completeness",
    "uncertainty": "Uncertainty preservation",
    "safety": "Medical safety enforcement",
    "conflict": "Conflict preservation",
    "trace": "Execution-trace correctness",
    "geo_honesty": "Geographic honesty",
}


def format_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append("# RAGnosis final verification report")
    L.append("")
    L.append(f"_Generated: {rep['generated_at']}_")
    L.append("")
    L.append(f"> {rep['disclaimer']}")
    L.append("")
    scope = rep["verification_scope"]
    L.append("## What this report does and does not verify")
    L.append("")
    L.append(f"- **Offline contract verification** — {scope['offline_contract_verification']}")
    L.append(f"- **Live dependency verification** — {scope['live_dependency_verification']}")
    L.append(f"- **Semantic-truth limitation** — {scope['semantic_truth_limitation']}")
    L.append("")
    c = rep["component_benchmark"]
    e = rep["end_to_end_benchmark"]
    L.append("## Benchmarks")
    L.append("")
    L.append("| Benchmark | numerator | denominator | status |")
    L.append("| --- | ---: | ---: | --- |")
    L.append(f"| Component contract pass rate | {c['metric']['numerator']} "
             f"| {c['metric']['denominator']} | {c['metric']['status']} |")
    L.append(f"| End-to-end contract pass rate | {e['metric']['numerator']} "
             f"| {e['metric']['denominator']} | {e['metric']['status']} |")
    L.append("")
    L.append(f"Component: {c['passed']}/{c['total']} passed, {c['failed']} failed, "
             f"{c['unverified']} unverified (ok={c['ok']}).")
    L.append(f"End-to-end: {e['passed']}/{e['total']} passed, {e['failed']} failed, "
             f"{e['unverified']} unverified (ok={e['ok']}).")
    L.append("")
    L.append("## End-to-end dimensions")
    L.append("")
    L.append("| Dimension | numerator | denominator | status |")
    L.append("| --- | ---: | ---: | --- |")
    for d in DIMENSION_ORDER:
        m = rep["dimensions"][d]
        L.append(f"| {_DIM_LABELS[d]} | {m['numerator']} | {m['denominator']} "
                 f"| {m['status']} |")
    L.append("")
    L.append("## Live validation (optional, environment-dependent)")
    L.append("")
    L.append(f"- Live model contract validation: **{rep['live_model_contract_validation']['status']}** "
             f"— {rep['live_model_contract_validation']['reason']}")
    L.append(f"- Live health validation: **{rep['live_health_validation']['status']}** "
             f"— {rep['live_health_validation']['reason']}")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", default=None,
                        help="Also write the machine-readable JSON report to PATH.")
    args = parser.parse_args(argv)

    rep = build_report()
    print(format_markdown(rep))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\nJSON report written to {args.json}")

    # Exit non-zero only if a REQUIRED offline benchmark failed.
    ok = rep["component_benchmark"]["ok"] and rep["end_to_end_benchmark"]["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
