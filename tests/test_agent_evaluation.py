"""Wire the end-to-end agent benchmark into the offline suite (Phase-5 §15).

These tests run the REAL ``AgentService.run()`` through deterministic injected
fakes (no network, no API keys) and assert the end-to-end contract holds. They
also carry focused regressions for the two agent bugs the benchmark surfaced:

  * a pure vision/literature question must NOT be blocked on a location just
    because the text contained an incidental geographic cue; and
  * a generated prediction that the user "will probably become infected" must be
    treated as a personal-medical claim and withheld.
"""

from __future__ import annotations

import pytest

from evaluation.agent_cases import CASES
from evaluation.agent_evaluators import DIMENSIONS
from evaluation.agent_runner import (
    DIMENSION_ORDER,
    AgentCaseResult,
    AgentEvaluationReport,
    DimensionResult,
    format_report,
    run,
)
from evaluation.outcome import Outcome
from multimodal.safety import detect_personal_medical_claim


def _case_result(case_id: str, *outcomes: Outcome, error: str | None = None):
    dims = [DimensionResult(f"dim{i}", o, "") for i, o in enumerate(outcomes)]
    return AgentCaseResult(case_id, "synthetic", dims, error=error)


# --- Report semantics: ok must require zero failed AND zero unverified -------
def test_report_ok_when_all_pass():
    report = AgentEvaluationReport(results=[
        _case_result("a", Outcome.PASS, Outcome.NOT_APPLICABLE),
        _case_result("b", Outcome.PASS, Outcome.PASS),
    ])
    assert report.failed == 0
    assert report.unverified == 0
    assert report.ok is True
    assert "OK" in format_report(report)


def test_report_not_ok_when_a_case_is_unverified():
    report = AgentEvaluationReport(results=[
        _case_result("a", Outcome.PASS, Outcome.PASS),
        _case_result("b", Outcome.PASS, Outcome.UNVERIFIED),
    ])
    assert report.failed == 0
    assert report.unverified == 1
    # The critical regression: no failures, but an unverified case must NOT be ok.
    assert report.ok is False
    text = format_report(report)
    assert "UNVERIFIED" in text
    assert "not OK" in text


def test_report_not_ok_when_a_case_errors():
    report = AgentEvaluationReport(results=[
        _case_result("a", Outcome.PASS),
        _case_result("b", error="RuntimeError: boom"),
    ])
    # An errored case is both failed and unverified per AgentCaseResult; either
    # way the report must not be ok.
    assert report.ok is False


def test_report_not_ok_when_a_case_fails():
    report = AgentEvaluationReport(results=[
        _case_result("a", Outcome.PASS),
        _case_result("b", Outcome.FAIL),
    ])
    assert report.failed == 1
    assert report.ok is False
    assert "FAILURES PRESENT" in format_report(report)


def test_component_runner_ok_requires_no_unverified():
    # The component runner shares the same corrected semantics.
    from evaluation.evaluators import CaseResult, Outcome as CO
    from evaluation.runner import EvaluationReport

    passing = CaseResult("x", "routing", "routing", CO.PASS, "")
    unver = CaseResult("y", "routing", "routing", CO.UNVERIFIED, "could not decide")
    # (case_id, kind, category, outcome, detail)
    assert EvaluationReport(results=[passing]).ok is True
    assert EvaluationReport(results=[passing, unver]).ok is False


def test_benchmark_has_full_dimension_coverage():
    # Every declared dimension must have an evaluator and run slot.
    assert set(DIMENSION_ORDER) == set(DIMENSIONS)


def test_end_to_end_benchmark_has_no_failures():
    report = run()
    assert report.total == len(CASES)
    assert report.total >= 20  # realistic coverage (§3/§12)
    failures = report.failures()
    assert not failures, f"E2E contract failures: {failures}"
    # Every case is decisively pass/fail — nothing silently unverified.
    assert report.unverified == 0, report.unverifieds()


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_each_case_passes_all_applicable_dimensions(case):
    report = run([case])
    (result,) = report.results
    assert result.error is None, result.error
    bad = [(d.dimension, d.detail) for d in result.dimensions
           if d.outcome in (Outcome.FAIL, Outcome.UNVERIFIED)]
    assert not bad, f"{case.id}: {bad}"


def test_grounding_detector_selftests_are_present():
    # At least one case must exercise the grounding detector in "fail" mode so
    # the detector cannot silently pass everything.
    fail_mode = [c for c in CASES if c.expected.get("grounding") == "fail"]
    assert fail_mode, "no grounding detector self-test case present"


# --- Regression: incidental geo cue must not force a location prompt ---------
def test_image_question_with_incidental_geo_cue_does_not_ask_location():
    case = next(c for c in CASES if c.id == "e2e-image-only")
    (result,) = run([case]).results
    tool = next(d for d in result.dimensions if d.dimension == "tool_selection")
    assert tool.outcome is Outcome.PASS, tool.detail


# --- Regression: "will probably become infected" is a personal claim --------
@pytest.mark.parametrize(
    "text",
    [
        "You will probably become infected with the flu.",
        "You will likely be infected with dengue.",
        "You are probably going to get infected.",
    ],
)
def test_infection_prediction_is_flagged(text):
    assert detect_personal_medical_claim(text)


@pytest.mark.parametrize(
    "text",
    [
        "Cases are rising in the region.",
        "You will receive your results next week.",
        "People in the area should take standard precautions.",
    ],
)
def test_benign_statements_not_flagged(text):
    assert not detect_personal_medical_claim(text)


# --- Final report generator (Phase-6 §17) -----------------------------------
def test_final_report_builds_and_is_consistent():
    from evaluation.final_report import build_report, format_markdown

    rep = build_report()
    # Structure present.
    for key in ("component_benchmark", "end_to_end_benchmark", "dimensions",
                "live_model_contract_validation", "live_health_validation",
                "verification_scope"):
        assert key in rep, key
    # The three verification layers must be separated explicitly (§4).
    assert set(rep["verification_scope"]) == {
        "offline_contract_verification",
        "live_dependency_verification",
        "semantic_truth_limitation",
    }
    # Every metric has numerator/denominator/status.
    for d, m in rep["dimensions"].items():
        assert set(m) == {"numerator", "denominator", "status"}, d
        assert m["numerator"] <= m["denominator"]
    # Offline benchmarks must be green and decided.
    assert rep["component_benchmark"]["ok"] is True
    assert rep["end_to_end_benchmark"]["ok"] is True
    # Live checks are UNVERIFIED unless the environment provides them; they must
    # never be fabricated as PASS here.
    assert rep["live_model_contract_validation"]["status"] in ("UNVERIFIED", "PASS")
    assert rep["live_health_validation"]["status"] in ("UNVERIFIED", "PASS")
    # Markdown renders without error and frames results as a contract pass rate.
    md = format_markdown(rep).lower()
    assert "contract pass rate" in md
    # The three verification layers must be rendered.
    assert "offline contract verification" in md
    assert "live dependency verification" in md
    assert "semantic-truth limitation" in md
    # Accuracy / real-world-surveillance terms may appear ONLY as negated
    # disclaimers; they must never be asserted as a property of the system.
    import re as _re
    for forbidden in ("factual accuracy", "medical accuracy",
                      "diagnostic accuracy", "diagnostic",
                      "real-time", "exhaustive"):
        for m in _re.finditer(_re.escape(forbidden), md):
            window = md[max(0, m.start() - 120):m.start()]
            assert "not " in window or "never" in window, (
                f"{forbidden!r} appears without negation")
