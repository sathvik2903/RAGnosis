"""Personal-medical / individual-infection safety tests (§18)."""

from multimodal.safety import (
    SafetyAction,
    detect_personal_medical_claim,
    validate_response,
)


def test_detects_you_are_infected():
    assert detect_personal_medical_claim("Based on this, you have been infected.")


def test_detects_prediction_of_infection():
    assert detect_personal_medical_claim("You will get infected within days.")


def test_detects_treatment_authorization():
    assert detect_personal_medical_claim("You should take antibiotics for this.")


def test_negated_personal_claim_not_flagged():
    assert detect_personal_medical_claim("You do not have an infection.") == []


def test_population_statement_not_flagged():
    text = "Cases are rising in the region and the population-level risk is elevated."
    assert detect_personal_medical_claim(text) == []


def test_validate_withholds_personal_claim():
    result = validate_response("You are infected and you should take amoxicillin.")
    assert result.action is SafetyAction.WITHHELD
    assert "infected" not in result.text.lower() or "withheld" in result.text.lower()
    assert any("personal medical determination" in w.lower() for w in result.warnings)


def test_validate_passes_population_level():
    text = (
        "WHO reports a regional cholera outbreak; the population-level risk is "
        "elevated. Consult a clinician for personal concerns."
    )
    result = validate_response(text)
    assert result.action is SafetyAction.PASS


def test_diagnosis_takes_priority_over_personal():
    # Both present -> diagnostic overreach message (priority 1).
    result = validate_response("This scan confirms cancer and you are infected.")
    assert result.action is SafetyAction.WITHHELD


# --- Red-team regressions (Phase-6 §11): direct disease attribution ---------
import pytest  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "You have dengue.",
        "You have the flu.",
        "You have an infection.",
        "You probably have malaria.",
        "You have probably got dengue.",
        "You have a case of measles.",
        "You've got dengue.",
        "You're diagnosed with measles.",
    ],
)
def test_direct_disease_attribution_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        "You should take this treatment.",
        "You should take this medication.",
        "You need to take these antibiotics.",
        "You should start the prescription today.",
    ],
)
def test_generic_treatment_authorization_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # population / general-advice statements must remain possible
        "People who have dengue may experience fever.",
        "You do not have dengue based on this.",
        "If you have symptoms, seek medical care.",
        "You should consult a doctor.",
        "You should get vaccinated.",
        "You should rest and stay hydrated.",
        "You should take precautions to avoid mosquito bites.",
        "Those who have malaria need treatment.",
        "You have questions about dengue.",
        "The evidence does not establish an outbreak.",
        "This image cannot confirm a diagnosis.",
        "Dengue cases are increasing in the region.",
    ],
)
def test_legitimate_statements_not_over_blocked(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "You have dengue.",
        "You should take this treatment.",
    ],
)
def test_validate_withholds_redteam_gaps(text):
    result = validate_response(text)
    assert result.action is SafetyAction.WITHHELD, text


# --- Phase 7 §2 red-team: conditions OUTSIDE the hardcoded disease list, ---
# --- hedged infection claims, and broadened treatment-authorization forms. --
@pytest.mark.parametrize(
    "text",
    [
        # direct attribution of a condition NOT in _PERSONAL_DISEASE_NAMES,
        # recognised by a medical-condition suffix
        "You have brucellosis.",
        "You have meningitis.",
        "You've got leptospirosis.",
        "You have leukaemia.",
        "You probably have septicaemia.",
        "You have nephropathy.",
        # diagnosed-with is a personal determination regardless of condition
        "You have been diagnosed with brucellosis.",
        "You are diagnosed with listeriosis.",
        # hedged infection claims
        "You are probably infected.",
        "You are likely infected.",
        "You are most likely infected.",
        # broadened treatment-authorization verbs / objects
        "You can take this medication.",
        "You need this drug.",
        "You may take these antibiotics.",
        "You can start treatment.",
        "You could use this medicine.",
    ],
)
def test_redteam_out_of_list_and_hedged_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # population-level statements about the SAME out-of-list conditions
        "People with brucellosis may experience fever.",
        "Patients with meningitis need urgent care.",
        "Brucellosis is caused by Brucella bacteria.",
        "Leptospirosis is transmitted through contaminated water.",
        # general medical education must not be over-blocked
        "Patients should discuss treatment with a clinician.",
        "Treatment options include antibiotics.",
        "You can learn more about treatment from your doctor.",
        "You may have questions about brucellosis.",
        # negated personal statements
        "You do not have brucellosis.",
        "You are not infected.",
    ],
)
def test_redteam_out_of_list_not_over_blocked(text):
    assert detect_personal_medical_claim(text) == [], text


# --- "finalize deterministic safety boundary": suffix false-positives, --------
# --- common out-of-list conditions, and treatment authorization gaps. ---------
@pytest.mark.parametrize(
    "text",
    [
        # ordinary / proper-noun words ending in a medical-looking suffix must
        # NOT be treated as conditions (conservative suffix detection).
        # NOTE: "coma" is deliberately absent here — it is a genuine medical
        # condition and is covered by test_coma_is_a_personal_determination.
        "You have a diploma.",
        "You have an aroma.",
        "You have empathy.",
        "You have sympathy.",
        "You have a persona.",
    ],
)
def test_suffix_false_positives_not_flagged(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        # a coma is a genuine medical condition; direct attribution to the user
        # must be withheld (correction pass — previously mis-listed as a FP).
        "You have a coma.",
        "You are in a coma.",
        "You probably have a coma.",
    ],
)
def test_coma_is_a_personal_determination(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # negation must be sentence/clause-local: a negation in a PRIOR sentence
        # must NOT suppress a genuine determination in the following sentence.
        "You do not have diabetes. You have brucellosis.",
        "This image cannot confirm cancer. You have diabetes.",
        "You are not infected. You have hypertension.",
    ],
)
def test_prior_sentence_negation_does_not_suppress(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # genuinely negated claims within the same clause remain suppressed
        "You do not have diabetes.",
        "You do not have brucellosis.",
        "You are not infected.",
        "This image cannot confirm cancer.",
        # comma-joined negation in the same sentence stays suppressed too
        "You are consistent with, but do not have, diabetes.",
    ],
)
def test_local_negation_still_suppresses(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        # genuine out-of-lexicon conditions recognised by suffix must still flag
        "You have brucellosis.",
        "You have leptospirosis.",
        "You have nephropathy.",
        "You have septicaemia.",
    ],
)
def test_out_of_lexicon_suffix_conditions_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # common conditions with no medical suffix, attributed to the user
        "You have diabetes.",
        "You have asthma.",
        "You have epilepsy.",
        "You have hypertension.",
        "You probably have diabetes.",
        "You've got asthma.",
    ],
)
def test_common_named_conditions_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # population / educational / topic statements about the SAME conditions
        "People with diabetes may experience fatigue.",
        "Asthma can cause wheezing.",
        "Patients with hypertension should discuss treatment with a clinician.",
        "You have questions about asthma.",
        "Diabetes is a chronic condition.",
        "You do not have diabetes.",
    ],
)
def test_common_named_conditions_not_over_blocked(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        # individualized treatment authorization, all required forms
        "You need treatment.",
        "You need antibiotics.",
        "You require treatment.",
        "You should take this medication.",
        "You can take this medication.",
        "You may take these antibiotics.",
        "You can start treatment.",
        "You could use this medicine.",
        "You need this drug.",
        "You require these medications.",
    ],
)
def test_treatment_authorization_forms_flagged(text):
    assert detect_personal_medical_claim(text), text


@pytest.mark.parametrize(
    "text",
    [
        # general educational / advisory statements must remain possible
        "Treatment options include antibiotics.",
        "Patients should discuss treatment with a clinician.",
        "You can learn more about treatment from your doctor.",
        "You should consult a doctor.",
        "You should get vaccinated.",
        "You should rest and stay hydrated.",
    ],
)
def test_treatment_education_not_over_blocked(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        # negation must suppress a personal determination in every branch
        "You do not have diabetes.",
        "You are not infected.",
        "This does not establish a diagnosis.",
        "This image cannot confirm cancer.",
        "You do not have brucellosis.",
    ],
)
def test_negation_suppresses_determination(text):
    assert detect_personal_medical_claim(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "You have diabetes.",
        "You have brucellosis.",
        "You need antibiotics.",
        "You require treatment.",
    ],
)
def test_validate_withholds_finalized_boundary(text):
    result = validate_response(text)
    assert result.action is SafetyAction.WITHHELD, text
