"""Focused unit tests for the strengthened grounding evaluator (Phase-6 §2/§3).

These construct minimal AgentResponse-like objects and drive ``eval_grounding``
directly so each violation category is covered in isolation, plus explicit
false-positive guards (faithful answers must stay PASS). This is deterministic
structured comparison against controlled fixtures — NOT general factual
verification; paraphrase/entailment are intentionally out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.agent_evaluators import (
    _answer_disease_mentions,
    _significant_tokens,
    _supported_disease_names,
    eval_grounding,
)
from evaluation.outcome import Outcome


def _evidence(title="", excerpt="", pmid=None, kind="pubmed_evidence", eid="ev-1", uri=None):
    return SimpleNamespace(
        title=title, excerpt=excerpt, uri=uri, kind=kind, evidence_id=eid,
        metadata={"pmid": pmid} if pmid else {},
    )


def _finding(disease_name, status="outbreak", transmissible=None,
             transmission_class="unknown", relevance="direct", **extra):
    f = {
        "disease_name": disease_name,
        "status": status,
        "transmissible": transmissible,
        "transmission_class": transmission_class,
        "relevance_to_location": relevance,
        "sources": [{
            "organization": "WHO", "tier": "primary_official",
            "title": f"{disease_name} report", "excerpt": f"{disease_name} details.",
            "uri": "https://who.int/x", "published_at": "2026-09-15T00:00:00+00:00",
            "updated_at": "2026-09-15T00:00:00+00:00",
            "retrieved_at": "2026-09-20T00:00:00+00:00",
        }],
    }
    f.update(extra)
    return f


def _response(answer, findings=None, evidence=None, live_status="ok"):
    health = None
    if findings is not None:
        health = {
            "findings": findings,
            "retrieved_at": "2026-09-20T00:00:00+00:00",
            "notes": [],
        }
    return SimpleNamespace(
        answer=answer,
        health=health,
        evidence=evidence or [],
        live_data_status=live_status,
    )


# --- helper-level sanity ----------------------------------------------------
def test_generic_disease_phrases_have_no_significant_tokens():
    assert _significant_tokens("infectious disease") == set()
    assert _significant_tokens("the disease") == set()
    # "who"/"disease"/"outbreak"/"news" are generic or event words; the
    # provenance phrase yields no disease-identifying token.
    assert _significant_tokens("WHO Disease Outbreak News") == set()


def test_out_of_lexicon_name_is_extracted():
    mentions = _answer_disease_mentions(
        "An oropouche virus disease outbreak was reported."
    )
    # The distinctive token of the extracted phrase must be the disease itself.
    assert any("oropouche" in _significant_tokens(m) for m in mentions), mentions


def test_supported_names_include_source_supplied_ool():
    r = _response("x", findings=[_finding("oropouche disease")])
    names = _supported_disease_names(r)
    assert "oropouche disease" in names


# --- faithful answers must PASS (false-positive guards) ---------------------
def test_faithful_lexicon_answer_passes():
    r = _response(
        "A dengue outbreak is reported in India per the source.",
        findings=[_finding("dengue")],
    )
    out, detail = eval_grounding(r, {"grounding": "pass"})
    assert out is Outcome.PASS, detail


def test_faithful_out_of_lexicon_answer_passes():
    r = _response(
        "An oropouche virus disease outbreak is reported.",
        findings=[_finding("oropouche disease")],
    )
    out, detail = eval_grounding(r, {"grounding": "pass"})
    assert out is Outcome.PASS, detail


def test_uncertainty_language_is_not_over_flagged():
    r = _response(
        "The evidence does not establish an outbreak; the situation is uncertain.",
        findings=[_finding("cholera", status="no_current_outbreak_status_found")],
    )
    out, detail = eval_grounding(r, {"grounding": "pass"})
    assert out is Outcome.PASS, detail


# --- each violation category must FAIL under pass-mode ----------------------
def _fails(response, exp_detail_substr):
    out, detail = eval_grounding(response, {"grounding": "pass"})
    assert out is Outcome.FAIL, detail
    assert exp_detail_substr in detail, detail


def test_unsupported_lexicon_disease_flagged():
    _fails(
        _response("A measles outbreak is spreading.", findings=[_finding("dengue")]),
        "unsupported disease/condition claim",
    )


def test_unsupported_out_of_lexicon_disease_flagged():
    _fails(
        _response("A marburg virus disease outbreak is reported.",
                  findings=[_finding("oropouche disease")]),
        "marburg",
    )


def test_fabricated_pmid_flagged():
    _fails(
        _response("Dengue is viral (PMID: 99999999).",
                  findings=[_finding("dengue")],
                  evidence=[_evidence(pmid="12345678")]),
        "fabricated PMID",
    )


def test_invented_year_flagged():
    _fails(
        _response("A dengue outbreak began in 1999.", findings=[_finding("dengue")]),
        "invented year",
    )


def test_absence_as_fact_under_unavailable_flagged():
    _fails(
        _response("There is no outbreak in India.", findings=[], live_status="unavailable"),
        "absence stated as fact",
    )


def test_active_spread_without_active_finding_flagged():
    # dengue present but only as endemic (not active) -> "is spreading" is unsupported
    _fails(
        _response("Dengue is currently spreading rapidly.",
                  findings=[_finding("dengue", status="endemic")]),
        "active spread",
    )


def test_unsupported_transmission_claim_flagged():
    _fails(
        _response("Dengue is contagious person-to-person.",
                  findings=[_finding("dengue", status="endemic",
                                     transmissible=False, transmission_class="vector")]),
        "transmissibility of 'dengue' not established",
    )


def test_local_presence_for_imported_only_flagged():
    _fails(
        _response("Mpox has a confirmed local outbreak here.",
                  findings=[_finding("mpox", status="outbreak", relevance="imported_risk")]),
        "imported-only",
    )


def test_uncertainty_to_certainty_flagged():
    _fails(
        _response("There is an outbreak, confirmed outbreak in the area.",
                  findings=[_finding("cholera", status="under_investigation")]),
        "certainty asserted despite hedged findings",
    )


# --- detector self-test mode ------------------------------------------------
def test_fail_mode_requires_a_detected_problem():
    # A faithful answer under fail-mode means the detector found nothing to flag
    # -> the self-test itself FAILS (the detector must catch planted violations).
    r = _response("A dengue outbreak is reported.", findings=[_finding("dengue")])
    out, detail = eval_grounding(r, {"grounding": "fail"})
    assert out is Outcome.FAIL, detail


def test_fail_mode_passes_when_problem_detected():
    r = _response("A measles outbreak is spreading.", findings=[_finding("dengue")])
    out, detail = eval_grounding(r, {"grounding": "fail"})
    assert out is Outcome.PASS, detail


# --- Phase-7: disease-SPECIFIC active-spread & transmission validation -------
# An active/transmissible finding for disease A must never justify a claim about
# disease B. These are the two release-candidate hardening fixes.

def test_cross_disease_active_spread_flagged():
    # dengue is an active outbreak, but the answer claims MEASLES is spreading.
    _fails(
        _response(
            "Measles is currently spreading rapidly in the area.",
            findings=[_finding("dengue", status="outbreak"),
                      _finding("measles", status="endemic")],
        ),
        "active spread of 'measles' with no active finding for that disease",
    )


def test_cross_disease_active_spread_does_not_hide_behind_other_active():
    # Sanity: the presence of ANY active finding used to short-circuit the
    # per-disease check. Ensure the false claim is still caught.
    out, detail = eval_grounding(
        _response(
            "Cholera is spreading. Dengue is spreading.",
            findings=[_finding("dengue", status="outbreak"),
                      _finding("cholera", status="endemic")],
        ),
        {"grounding": "pass"},
    )
    assert out is Outcome.FAIL
    assert "cholera" in detail
    # the true dengue claim must NOT be flagged
    assert "active spread of 'dengue'" not in detail


def test_same_disease_active_spread_passes():
    r = _response(
        "Measles is currently spreading rapidly in the area.",
        findings=[_finding("measles", status="outbreak")],
    )
    out, detail = eval_grounding(r, {"grounding": "pass"})
    assert out is Outcome.PASS, detail


def test_cross_disease_transmission_flagged():
    # measles is transmissible, but the answer claims DENGUE is contagious.
    _fails(
        _response(
            "Dengue is contagious person-to-person.",
            findings=[_finding("measles", status="outbreak",
                               transmissible=True, transmission_class="airborne"),
                      _finding("dengue", status="endemic",
                               transmissible=False, transmission_class="vector")],
        ),
        "transmissibility of 'dengue' not established by a finding for that disease",
    )


def test_same_disease_transmission_passes():
    r = _response(
        "Measles is contagious person-to-person.",
        findings=[_finding("measles", status="outbreak",
                           transmissible=True, transmission_class="airborne")],
    )
    out, detail = eval_grounding(r, {"grounding": "pass"})
    assert out is Outcome.PASS, detail


def test_transmission_claim_without_named_disease_falls_back():
    # No disease named in the sentence -> require SOME transmissible finding.
    _fails(
        _response(
            "This is contagious person-to-person.",
            findings=[_finding("dengue", status="outbreak",
                               transmissible=False, transmission_class="vector")],
        ),
        "transmissibility not established by any finding",
    )


# --- Phase-7: no false passes from uncertainty / imported / conflict / multi -
def test_imported_only_active_spread_flagged():
    _fails(
        _response(
            "Mpox is currently spreading locally with a local outbreak here.",
            findings=[_finding("mpox", status="outbreak", relevance="imported_risk")],
        ),
        "imported-only",
    )


def test_conflicting_claimed_as_confirmed_flagged():
    _fails(
        _response(
            "There is a confirmed outbreak of cholera.",
            findings=[_finding("cholera", status="conflicting",
                               conflict_summary="sources disagree")],
        ),
        "confirmed outbreak",
    )


def test_multi_disease_mixed_truth_flags_only_false_claim():
    out, detail = eval_grounding(
        _response(
            "Dengue is contagious person-to-person. Cholera is spreading rapidly.",
            findings=[_finding("dengue", status="outbreak",
                               transmissible=True, transmission_class="person_to_person"),
                      _finding("cholera", status="endemic")],
        ),
        {"grounding": "pass"},
    )
    assert out is Outcome.FAIL
    # cholera spread claim flagged; dengue transmission claim NOT flagged
    assert "cholera" in detail
    assert "transmissibility of 'dengue'" not in detail
