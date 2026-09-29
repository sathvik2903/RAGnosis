"""Adversarial semantic-parsing tests for status/transmission/alert (§6, §7, §16).

These prove the Phase-3 context-aware parser does NOT classify every occurrence
of a keyword as an active/affirmative claim. Every case here is a regression
guard for a specific naive-substring bug.
"""

import pytest

from multimodal.health_semantics import (
    analyze_alert,
    analyze_status,
    analyze_transmission,
)


# --- STATUS (§6) -----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("No outbreak has been detected.", "no_current_outbreak_status_found"),
        ("There is currently no evidence of an outbreak.", "no_current_outbreak_status_found"),
        ("An outbreak was ruled out.", "no_current_outbreak_status_found"),
        ("Previous outbreak, now contained.", "no_current_outbreak_status_found"),
        ("The outbreak has ended and been declared over.", "no_current_outbreak_status_found"),
        ("Outbreak risk remains low.", "no_current_outbreak_status_found"),
        ("Outbreak risk is minimal at this time.", "no_current_outbreak_status_found"),
    ],
)
def test_status_not_active(text, expected):
    status, _term, _reason = analyze_status(None, text)
    assert status == expected, f"{text!r} -> {status}"


@pytest.mark.parametrize(
    "text",
    [
        "Reports are investigating a possible outbreak.",
        "Health officials are assessing a suspected outbreak.",
        "Authorities probe a potential cluster of cases.",
    ],
)
def test_status_hypothetical_is_unknown(text):
    status, _term, _reason = analyze_status(None, text)
    assert status == "unknown", f"{text!r} -> {status}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ongoing cholera outbreak confirmed by the ministry.", "outbreak"),
        ("WHO declares a measles epidemic.", "epidemic"),
        ("A pandemic is underway according to WHO.", "pandemic"),
        ("An active cluster of Legionnaires cases has been reported.", "cluster"),
        ("Dengue is endemic in this country.", "endemic"),
    ],
)
def test_status_affirmative(text, expected):
    status, _term, _reason = analyze_status(None, text)
    assert status == expected, f"{text!r} -> {status}"


def test_status_preserves_source_term():
    _status, term, _reason = analyze_status("Outbreak", "Ongoing cholera outbreak.")
    assert term == "Outbreak"


def test_status_no_keyword_is_unknown():
    status, _term, _reason = analyze_status(None, "Routine vaccination coverage report.")
    assert status == "unknown"


def test_status_stronger_wins_when_multiple_affirmative():
    status, _term, _reason = analyze_status(
        None, "A confirmed outbreak has escalated into a declared epidemic."
    )
    assert status == "epidemic"


# --- TRANSMISSION (§7) -----------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("The disease does not spread person to person.", "not_person_to_person"),
        ("There is no evidence of human-to-human transmission.", "not_person_to_person"),
        ("The disease is not known to be contagious.", "not_person_to_person"),
    ],
)
def test_transmission_negated_contagion(text, expected):
    cls, _reason = analyze_transmission(None, text)
    assert cls == expected, f"{text!r} -> {cls}"


@pytest.mark.parametrize(
    "text",
    [
        "There is possible human-to-human transmission under investigation.",
        "The report discusses airborne transmission elsewhere.",
        "Guidance on airborne precautions is provided.",
    ],
)
def test_transmission_uncertain_is_unknown(text):
    cls, _reason = analyze_transmission(None, text)
    assert cls == "unknown", f"{text!r} -> {cls}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Transmission is primarily mosquito-borne.", "vector_borne"),
        ("Confirmed person-to-person spread is ongoing.", "contagious_person_to_person"),
        ("The outbreak is waterborne, linked to contaminated water.", "food_or_water_borne"),
        ("This is a zoonotic spillover from animals.", "zoonotic"),
    ],
)
def test_transmission_affirmative(text, expected):
    cls, _reason = analyze_transmission(None, text)
    assert cls == expected, f"{text!r} -> {cls}"


def test_transmission_no_statement_is_unknown():
    cls, _reason = analyze_transmission(None, "Cases continue to be reported.")
    assert cls == "unknown"


# --- ALERT (§16) -----------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("No official alert has been issued.", "none"),
        ("Alert lifted.", "none"),
        ("The previous emergency has ended.", "none"),
        ("The health emergency was declared over last month.", "none"),
    ],
)
def test_alert_negated_or_past_is_none(text, expected):
    level, _reason = analyze_alert(text)
    assert level == expected, f"{text!r} -> {level}"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Monitoring only.", "watch"),
        ("Countries were advised to remain vigilant.", "watch"),
        ("A health advisory has been issued.", "watch"),
        ("Elevated risk reported by the agency.", "elevated"),
        ("WHO declared a public health emergency of international concern.", "official_alert"),
        ("A public health emergency has been declared.", "official_alert"),
    ],
)
def test_alert_affirmative_levels(text, expected):
    level, _reason = analyze_alert(text)
    assert level == expected, f"{text!r} -> {level}"


def test_alert_highest_current_wins():
    level, _reason = analyze_alert(
        "Initial advisory issued; WHO has now declared a public health emergency."
    )
    assert level == "official_alert"


def test_alert_none_when_no_keyword():
    level, reason = analyze_alert("Weekly surveillance summary published.")
    # "surveillance" alone should not raise an alert.
    assert level == "none"
    assert reason is None
