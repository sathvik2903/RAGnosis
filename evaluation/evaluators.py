"""Deterministic evaluators for the benchmark (Phase-4 §7).

Each evaluator maps an :class:`~evaluation.cases.EvalCase` to a
:class:`CaseResult` with one of four outcomes:

* ``PASS``       — the system behaved as the case expects.
* ``FAIL``       — the system behaved differently (a real defect or regression).
* ``NOT_APPLICABLE`` — the case cannot apply in this configuration.
* ``UNVERIFIED`` — the check could not be evaluated (e.g. a dependency missing);
  this is never silently counted as a pass.

Everything runs offline against the shipped public interfaces and injected fake
providers. No API keys, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import FeedItem, HealthIntelligence
from multimodal.health_semantics import (
    analyze_alert,
    analyze_status,
    analyze_transmission,
)
from multimodal.location import parse_locations
from multimodal.routing import classify
from multimodal.safety import validate_response
from multimodal.schemas import Evidence

from .cases import EvalCase
from .outcome import Outcome

# A fixed clock so freshness cases are deterministic.
EVAL_NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _clock() -> datetime:
    return EVAL_NOW


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    kind: str
    category: str
    outcome: Outcome
    detail: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "category": self.category,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "expected": self.expected,
            "actual": self.actual,
        }


def _pass(case: EvalCase, detail: str = "", actual: Any = None) -> CaseResult:
    return CaseResult(case.id, case.kind, case.category, Outcome.PASS, detail,
                      case.expected, actual)


def _fail(case: EvalCase, detail: str, actual: Any = None) -> CaseResult:
    return CaseResult(case.id, case.kind, case.category, Outcome.FAIL, detail,
                      case.expected, actual)


def _na(case: EvalCase, detail: str) -> CaseResult:
    return CaseResult(case.id, case.kind, case.category, Outcome.NOT_APPLICABLE, detail,
                      case.expected)


def _unverified(case: EvalCase, detail: str) -> CaseResult:
    return CaseResult(case.id, case.kind, case.category, Outcome.UNVERIFIED, detail,
                      case.expected)


# ---------------------------------------------------------------------------
# Fake provider used by health-pipeline cases (offline, deterministic).
# ---------------------------------------------------------------------------

def _feed_item(spec: dict[str, Any], org: str, tier: str) -> FeedItem:
    d = dict(
        organization=org,
        tier=tier,
        title="",
        summary="",
        uri=None,
        published_at="2026-09-15T00:00:00+00:00",
        updated_at="2026-09-15T00:00:00+00:00",
        retrieved_at=EVAL_NOW.isoformat(),
        geo_scope="national",
    )
    d.update(spec)
    return FeedItem(**d)


class _FakeProvider:
    def __init__(self, name: str, tier: str, items: list[FeedItem], fail: bool = False):
        self.name = name
        self.tier = tier
        self._items = items
        self._fail = fail

    def fetch(self, locations):
        if self._fail:
            raise RuntimeError("simulated source failure")
        return list(self._items)


def _build_health(providers_spec: list[dict[str, Any]]) -> HealthIntelligence:
    providers = []
    for p in providers_spec:
        items = [_feed_item(s, p["name"], p["tier"]) for s in p.get("items", [])]
        providers.append(_FakeProvider(p["name"], p["tier"], items, p.get("fail", False)))
    return HealthIntelligence(
        MultimodalConfig(health_cache_ttl=0), providers=providers, clock=_clock
    )


# ---------------------------------------------------------------------------
# Evaluators by kind
# ---------------------------------------------------------------------------

def _eval_routing(case: EvalCase) -> CaseResult:
    i = case.inputs
    decision = classify(i["question"], i.get("has_image", False), i.get("has_location", False))
    caps = decision.capabilities
    actual = {
        "capabilities": caps,
        "geographic_intent": decision.geographic_intent,
        "live_health_intent": decision.live_health_intent,
    }
    exp = case.expected

    if "capabilities" in exp and set(caps) != set(exp["capabilities"]):
        return _fail(case, f"capabilities {caps} != expected {exp['capabilities']}", actual)
    if "must_use" in exp:
        missing = [c for c in exp["must_use"] if c not in caps]
        if missing:
            return _fail(case, f"missing capabilities {missing}", actual)
    for key in ("geographic_intent", "live_health_intent"):
        if key in exp and bool(getattr(decision, key)) != bool(exp[key]):
            return _fail(case, f"{key}={getattr(decision, key)} != {exp[key]}", actual)
    return _pass(case, "routing matched", actual)


def _eval_geography(case: EvalCase) -> CaseResult:
    from multimodal.health_intelligence import assess_relevance

    i = case.inputs
    locs = parse_locations(i["location"])
    if not locs:
        return _unverified(case, f"could not parse location {i['location']!r}")
    item = _feed_item(
        {"title": i["title"], "summary": i.get("summary", ""),
         "geo_scope": i.get("geo_scope", "national")},
        "WHO", "primary_official",
    )
    relevance, reason, _areas = assess_relevance(item, locs[0])
    if relevance != case.expected["relevance"]:
        return _fail(case, f"relevance {relevance!r} != {case.expected['relevance']!r} "
                           f"({reason})", relevance)
    return _pass(case, f"relevance {relevance} ({reason})", relevance)


def _eval_status(case: EvalCase) -> CaseResult:
    status, _term, reason = analyze_status(case.inputs.get("term"), case.inputs["text"])
    if status != case.expected["status"]:
        return _fail(case, f"status {status!r} != {case.expected['status']!r} ({reason})",
                     status)
    return _pass(case, f"status {status} ({reason})", status)


def _eval_transmission(case: EvalCase) -> CaseResult:
    cls, reason = analyze_transmission(case.inputs.get("term"), case.inputs["text"])
    if cls != case.expected["transmission"]:
        return _fail(case, f"transmission {cls!r} != {case.expected['transmission']!r} "
                           f"({reason})", cls)
    return _pass(case, f"transmission {cls} ({reason})", cls)


def _eval_alert(case: EvalCase) -> CaseResult:
    level, reason = analyze_alert(case.inputs["text"])
    if level != case.expected["alert"]:
        return _fail(case, f"alert {level!r} != {case.expected['alert']!r} ({reason})",
                     level)
    return _pass(case, f"alert {level} ({reason})", level)


def _find_disease(findings, name_contains: str):
    for f in findings:
        if name_contains.lower() in f.disease_name.lower():
            return f
    return None


def _eval_source_quality(case: EvalCase) -> CaseResult:
    i = case.inputs
    hi = _build_health(i["providers"])
    result = hi.gather(parse_locations(i["location"]))
    exp = case.expected
    actual: dict[str, Any] = {"live_data_status": result.live_data_status,
                              "findings": [f.disease_name for f in result.findings]}

    if "live_data_status" in exp and result.live_data_status != exp["live_data_status"]:
        return _fail(case, f"live_data_status {result.live_data_status!r} != "
                           f"{exp['live_data_status']!r}", actual)
    if "findings_count" in exp and len(result.findings) != exp["findings_count"]:
        return _fail(case, f"findings_count {len(result.findings)} != "
                           f"{exp['findings_count']}", actual)

    finding = None
    if "disease" in exp:
        finding = _find_disease(result.findings, exp["disease"])
        if finding is None:
            return _fail(case, f"expected disease {exp['disease']!r} not found", actual)

    if "status" in exp:
        if finding is None:
            return _fail(case, "no finding to check status", actual)
        if finding.status != exp["status"]:
            return _fail(case, f"status {finding.status!r} != {exp['status']!r}",
                         finding.status)
    if "classification_source" in exp:
        if finding.classification_source != exp["classification_source"]:
            return _fail(case, f"lead {finding.classification_source!r} != "
                               f"{exp['classification_source']!r}",
                         finding.classification_source)
    if exp.get("conflict_present"):
        if not (finding.status == "conflicting" or finding.conflict_summary):
            return _fail(case, "expected a conflict but none recorded", actual)
    if "source_orgs" in exp:
        orgs = {s.organization for s in finding.sources}
        missing = [o for o in exp["source_orgs"] if o not in orgs]
        if missing:
            return _fail(case, f"conflict lost sources {missing}", list(orgs))
    if "data_gap_state" in exp:
        if finding.data_gap_state != exp["data_gap_state"]:
            return _fail(case, f"data_gap {finding.data_gap_state!r} != "
                               f"{exp['data_gap_state']!r}", finding.data_gap_state)
    if exp.get("no_concealment_language"):
        banned = ("conceal", "suppress", "cover-up", "cover up", "downplay", "hiding")
        blob = f"{finding.uncertainty or ''} {finding.conflict_summary or ''}".lower()
        hit = [b for b in banned if b in blob]
        if hit:
            return _fail(case, f"concealment language present: {hit}", blob)
    if "freshness_state" in exp:
        if finding.freshness_state != exp["freshness_state"]:
            return _fail(case, f"freshness {finding.freshness_state!r} != "
                               f"{exp['freshness_state']!r}", finding.freshness_state)
    if "disease_contains" in exp:
        f2 = _find_disease(result.findings, exp["disease_contains"])
        if f2 is None:
            return _fail(case, f"disease containing {exp['disease_contains']!r} not found",
                         actual)

    return _pass(case, "source-quality expectations met", actual)


def _eval_safety(case: EvalCase) -> CaseResult:
    result = validate_response(case.inputs["text"], [], [])
    expected = case.expected["action"]
    actual = result.action.value
    if actual != expected:
        return _fail(case, f"safety action {actual!r} != {expected!r} "
                           f"(warnings: {result.warnings})", actual)
    return _pass(case, f"safety action {actual}", actual)


def _eval_citation(case: EvalCase) -> CaseResult:
    i = case.inputs
    evidence = [
        Evidence("PubMed", f"study {p}", "excerpt", metadata={"pmid": p})
        for p in i.get("evidence_pmids", [])
    ]
    result = validate_response(i["text"], evidence, [])
    exp = case.expected
    actual = result.action.value
    if actual != exp["action"]:
        return _fail(case, f"action {actual!r} != {exp['action']!r} "
                           f"(warnings: {result.warnings})", actual)
    if "retains_pmid" in exp and exp["retains_pmid"] not in result.text:
        return _fail(case, f"expected PMID {exp['retains_pmid']} retained", result.text)
    if "removes_pmid" in exp and exp["removes_pmid"] in result.text:
        return _fail(case, f"fabricated PMID {exp['removes_pmid']} not removed", result.text)
    return _pass(case, f"citation handling: {actual}", actual)


_DISPATCH = {
    "routing": _eval_routing,
    "geography": _eval_geography,
    "status": _eval_status,
    "transmission": _eval_transmission,
    "alert": _eval_alert,
    "source_quality": _eval_source_quality,
    "safety": _eval_safety,
    "citation": _eval_citation,
}


def evaluate_case(case: EvalCase) -> CaseResult:
    """Evaluate a single case, dispatching by ``kind``. Never raises for a
    normal defect; unexpected exceptions become UNVERIFIED so they are visible
    without masking as a pass."""
    fn = _DISPATCH.get(case.kind)
    if fn is None:
        return _unverified(case, f"no evaluator for kind {case.kind!r}")
    try:
        return fn(case)
    except Exception as exc:  # noqa: BLE001 - surface as UNVERIFIED, not a silent pass
        return _unverified(case, f"evaluator raised {type(exc).__name__}: {exc}")
