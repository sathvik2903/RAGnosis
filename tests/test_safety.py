"""Deterministic safety detection AND enforcement tests (independent of any model)."""

from multimodal.safety import (
    SafetyAction,
    WITHHELD_DIAGNOSIS_MESSAGE,
    detect_fabricated_citations,
    detect_overconfident_diagnosis,
    validate_response,
)
from multimodal.schemas import Evidence


def _evidence(pmid: str) -> Evidence:
    return Evidence(
        source="PubMed",
        title="t",
        excerpt="e",
        uri=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


# --------------------------------------------------------------------------- #
# Detection: definitive-diagnosis language
# --------------------------------------------------------------------------- #
def test_detects_definitive_diagnosis_language():
    assert detect_overconfident_diagnosis("This scan confirms malignancy.")
    assert detect_overconfident_diagnosis("The image shows cancer.")
    assert detect_overconfident_diagnosis("You have a tumor.")
    assert detect_overconfident_diagnosis("This is definitely malignant.")
    assert detect_overconfident_diagnosis("The MRI reveals a carcinoma.")
    assert detect_overconfident_diagnosis("The patient was diagnosed with pneumonia.")


# --------------------------------------------------------------------------- #
# Detection: false positives that MUST NOT be flagged
# --------------------------------------------------------------------------- #
def test_allows_conservative_language():
    conservative = (
        "The image shows an area of increased density that may warrant further "
        "review. This is not a diagnosis and cannot confirm any condition."
    )
    assert detect_overconfident_diagnosis(conservative) == []


def test_allows_negated_confirmation():
    assert detect_overconfident_diagnosis("This scan cannot confirm cancer.") == []
    assert (
        detect_overconfident_diagnosis(
            "These findings are consistent with, but do not confirm, malignancy."
        )
        == []
    )
    assert (
        detect_overconfident_diagnosis(
            "It is not possible to confirm a diagnosis from this image alone."
        )
        == []
    )


def test_allows_discussing_cancer_as_possibility():
    assert (
        detect_overconfident_diagnosis(
            "Cancer is one possibility that a clinician should evaluate."
        )
        == []
    )


def test_allows_differential_diagnosis_discussion():
    assert (
        detect_overconfident_diagnosis(
            "The differential diagnosis of a lung nodule includes infection and malignancy."
        )
        == []
    )


def test_allows_quoting_evidence_about_diagnosis():
    assert (
        detect_overconfident_diagnosis(
            "The evidence discusses the diagnosis of asthma in children."
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Detection: fabricated citations
# --------------------------------------------------------------------------- #
def test_detects_fabricated_pmid():
    text = "See PMID: 99999999 for details."
    fabricated = detect_fabricated_citations(text, [_evidence("12345678")])
    assert fabricated == ["99999999"]


def test_accepts_cited_pmid_present_in_evidence():
    text = "As shown in PMID 12345678."
    fabricated = detect_fabricated_citations(text, [_evidence("12345678")])
    assert fabricated == []


def test_detects_fabricated_pubmed_url():
    text = "Reference: https://pubmed.ncbi.nlm.nih.gov/55555555/"
    fabricated = detect_fabricated_citations(text, [_evidence("12345678")])
    assert "55555555" in fabricated


# --------------------------------------------------------------------------- #
# Enforcement: the core of the safety requirement
# --------------------------------------------------------------------------- #
def test_enforcement_withholds_definitive_diagnosis():
    unsafe = "Based on the scan, you have cancer. This is confirmed."
    result = validate_response(unsafe, [])
    assert result.action is SafetyAction.WITHHELD
    assert result.enforced
    # The unsafe substantive claim must NOT be returned to the user.
    assert "you have cancer" not in result.text.lower()
    assert result.text.startswith(WITHHELD_DIAGNOSIS_MESSAGE[:40])
    assert not result.ok  # warnings recorded


def test_enforcement_redacts_fabricated_citations_preserving_safe_text():
    text = "This area may warrant clinician review. See PMID 99999999 and PMID 12345678."
    result = validate_response(text, [_evidence("12345678")])
    assert result.action is SafetyAction.REDACTED
    assert "99999999" not in result.text
    assert "12345678" in result.text  # legitimate citation preserved
    assert "This area may warrant clinician review" in result.text


def test_enforcement_withhold_takes_priority_over_redaction():
    # Both problems present -> the answer is withheld (stronger action) and both
    # warnings are recorded.
    text = "This scan confirms cancer. See PMID 99999999."
    result = validate_response(text, [_evidence("11111111")])
    assert result.action is SafetyAction.WITHHELD
    assert "99999999" not in result.text
    assert len(result.warnings) == 2


def test_safe_output_passes_through_unchanged():
    text = (
        "The image shows nonspecific findings that may warrant clinician review. "
        "Evidence in PMID 12345678 discusses similar patterns but is not "
        "patient-specific."
    )
    result = validate_response(text, [_evidence("12345678")])
    assert result.action is SafetyAction.PASS
    assert result.ok
    assert not result.enforced
    assert text in result.text  # original content preserved verbatim


def test_pass_appends_safety_notice_when_missing():
    result = validate_response("Some informational content.", [])
    assert "safety note:" in result.text.lower()
    assert result.action is SafetyAction.PASS


def test_does_not_duplicate_safety_notice():
    text = "Content. Safety note: consult a clinician."
    result = validate_response(text, [])
    assert result.text.lower().count("safety note:") == 1
