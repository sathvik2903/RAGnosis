from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .schemas import Evidence, ImageObservation

MEDICAL_BOUNDARY = (
    "RAGnosis provides informational, evidence-grounded biomedical analysis. "
    "It is not a diagnostic or treatment system and must not replace qualified clinical judgment."
)

IMAGE_BOUNDARY = (
    "Image observations are model-generated observations, not confirmed findings. "
    "The system must not present an image observation as a diagnosis, malignancy determination, "
    "risk score, or treatment recommendation."
)

# Short, explicit rules that both the vision stage and the generation stage share.
# The vision prompt and tests reference these by content, so they live here as the
# single source of truth for the safety contract.
IMAGE_RULES = (
    "Do not diagnose disease.",
    "Do not give a definitive diagnosis from an image.",
    "Do not claim a tumor, cancer, infection, fracture, or other condition is confirmed.",
    "Describe visible patterns conservatively and always include uncertainty.",
    "If the image is not a medical image or is inadequate, report that explicitly.",
    "Confidence must be null unless it is a justified observation confidence.",
    "Do not infer patient identity or demographics that are not visible in the image.",
)


def build_system_instruction() -> str:
    rules = "\n".join(f"- {rule}" for rule in IMAGE_RULES)
    return f"""You are RAGnosis, an evidence-grounded biomedical research assistant.

SAFETY BOUNDARY:
- {MEDICAL_BOUNDARY}
- {IMAGE_BOUNDARY}
- Distinguish what is directly observed, what is retrieved from evidence, and what is uncertain.
- Never invent a source, citation, imaging finding, patient fact, or confidence value.
- Do not infer a person's identity from an image.
{rules}
- If the supplied image is inadequate, say that explicitly.
- Encourage qualified clinical review for personal medical decisions.

OUTPUT DISCIPLINE:
1. Observations, if any.
2. Relevant evidence and why it was retrieved.
3. Uncertainty and limitations.
4. A concise answer to the user's question.
"""


# ---------------------------------------------------------------------------
# Deterministic output validation and ENFORCEMENT
# ---------------------------------------------------------------------------
# The prompt above *asks* the model to behave. The checks below *verify* the
# output after generation, independent of the model. Crucially, when a check
# fails the pipeline does not present the model's substantive claim as an
# authoritative answer: diagnostic overreach is withheld and fabricated
# citations are removed, so unsafe text is never returned verbatim as the answer.
#
# This is a conservative heuristic layer, not a proof of medical safety. It
# reduces, but cannot mathematically guarantee the absence of, overconfident or
# fabricated output.

SAFETY_NOTICE = (
    "\n\n---\nSafety note: This response is informational and derived from model "
    "observations and retrieved literature. It is not a diagnosis. Please consult "
    "a qualified clinician for any personal medical decision."
)

# Response substituted when diagnostic overreach is detected. The unsafe answer
# is withheld; observations and evidence remain available in the structured
# response so nothing is silently lost.
WITHHELD_DIAGNOSIS_MESSAGE = (
    "The generated answer was withheld because it asserted a definitive medical "
    "diagnosis, which RAGnosis does not provide. RAGnosis is an informational "
    "research assistant, not a diagnostic system. The image observations and any "
    "retrieved literature are available in the structured fields of this response; "
    "please review them with a qualified clinician, who can interpret them in the "
    "full clinical context."
)

# Response substituted when a personal medical determination is detected (§18).
WITHHELD_PERSONAL_MESSAGE = (
    "The generated answer was withheld because it made a personal medical "
    "determination about you (such as asserting you are or will be infected, or "
    "presenting a treatment as authorized for you). RAGnosis provides population- "
    "and region-level public health information and biomedical evidence; it does "
    "not assess an individual's infection status, predict personal infection, or "
    "authorize treatment. If you have symptoms or a personal health concern, "
    "please consult a qualified clinician. Any retrieved public-health findings "
    "and literature remain available in the structured fields of this response."
)


class SafetyAction(str, Enum):
    """What the enforcement layer did to the model output."""

    PASS = "pass"  # safe output returned unchanged
    REDACTED = "redacted"  # fabricated citations stripped, rest preserved
    WITHHELD = "withheld"  # diagnostic overreach; substantive answer replaced


# Negation cues that, when they appear shortly before a matched phrase, indicate
# the sentence is *denying* a definitive claim ("cannot confirm cancer") rather
# than asserting one. Used to suppress false positives.
_NEGATION_CUES = (
    "not",
    "no",
    "cannot",
    "can't",
    "cant",
    "unable",
    "without",
    "never",
    "isn't",
    "aren't",
    "does not",
    "doesn't",
    "rule out",
    "ruling out",
    "unlikely",
)

# Phrases that assert a *confirmed* diagnosis. Kept deliberately narrow: they
# require definitive framing ("this scan shows cancer", "you have a tumor",
# "confirms malignancy"), not mere mention of a condition. "diagnosis of X" is
# intentionally NOT matched here because it appears in legitimate differential
# discussion; only "diagnosed with X" (a statement about the patient) is.
_DEFINITIVE_PATTERNS = (
    r"(?:this|the)\s+(?:image|scan|x-?ray|mri|ct|ultrasound|radiograph)\s+"
    r"(?:shows?|confirms?|proves?|demonstrates?|reveals?)\s+"
    r"(?:a\s+|an\s+)?(?:malignan\w+|cancer|tumou?rs?|carcinoma|metastas\w+|fractures?|infections?)",
    r"confirm(?:s|ed)?\s+(?:a\s+|the\s+)?(?:diagnosis|malignan\w+|cancer|tumou?r|carcinoma)",
    r"(?:you|the patient)\s+(?:have|has|are|is)\s+(?:definitely\s+|certainly\s+|clearly\s+)?"
    r"(?:cancer|a\s+tumou?r|a\s+malignan\w+|carcinoma|metastatic\s+\w+)",
    r"(?:is|are)\s+(?:definitely|certainly|clearly)\s+(?:malignant|cancerous|benign)",
    r"diagnos(?:ed)\s+(?:with|as)\s+\w+",
    r"100%\s+(?:certain|sure|confident)",
    r"there\s+is\s+no\s+doubt\s+(?:that\s+)?(?:this|it|the\s+\w+)\s+is\s+"
    r"(?:cancer|malignan\w+|a\s+tumou?r)",
)

# Disease/condition names that, when attributed directly to "you", constitute a
# personal medical determination ("you have dengue"). Kept small and explicit;
# generic nouns (disease/infection/virus/illness) are handled by a separate
# pattern above. "the flu"/"a cold" colloquialisms are included.
_PERSONAL_DISEASE_NAMES = (
    "dengue", "malaria", "cholera", "measles", "mpox", "monkeypox", "ebola",
    "marburg", "influenza", "the flu", "flu", "covid-19", "covid", "zika",
    "chikungunya", "typhoid", "tuberculosis", "tb", "hepatitis", "nipah",
    "polio", "diphtheria", "meningitis", "rabies", "plague", "cancer",
    "a tumour", "a tumor", "pneumonia", "a cold", "leptospirosis",
    "yellow fever", "lassa fever", "scrub typhus", "japanese encephalitis",
)

# Common conditions that are personal medical determinations when attributed to
# the user but do NOT end in a recognised medical suffix, so the suffix rule
# below cannot catch them. Kept as an explicit, conservative allow-list of
# widely-known conditions (not an attempt at exhaustive medical coverage).
_PERSONAL_CONDITION_NAMES = (
    "diabetes", "asthma", "epilepsy", "hypertension", "anaemia", "anemia",
    "arthritis", "hiv", "aids", "covid", "covid-19", "influenza", "the flu",
    # "coma" is a genuine medical condition; it is too short for the suffix
    # detector's minimum-stem rule, so it is listed explicitly here.
    "coma",
)

# Ordinary English / proper-noun words that end in a medical-looking suffix but
# are NOT medical conditions. The suffix-based detector below excludes these so
# it does not flag sentences like "You have a diploma." A short, curated list is
# used deliberately: the detector stays conservative rather than guessing.
_NON_MEDICAL_SUFFIX_WORDS = (
    "diploma", "aroma", "sympathy", "empathy", "apathy", "antipathy",
    "telepathy", "homeopathy", "osteopathy", "naturopathy", "allopathy",
    "oklahoma", "sonoma", "tacoma", "paloma", "roma", "gnosis", "prognosis",
    "diagnosis",
)

# §18 — personal medical determination patterns. These assert something about
# the *individual user's* health status, infection, or treatment authorization,
# which RAGnosis must never do. Population/regional statements are NOT matched
# here (e.g. "cases are rising in the region" is allowed); only claims directed
# at "you"/"your" as a personal determination are flagged.
_PERSONAL_MEDICAL_PATTERNS = (
    # "you have/are infected/have caught ..."
    r"\byou\s+(?:have|are|might have|probably have|likely have|may have)\s+"
    r"(?:been\s+)?(?:infected|contracted|caught|got|developed)\b",
    r"\byou\s+(?:are|'re)\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+"
    r"|clearly\s+|most likely\s+)?(?:infected|contagious|sick with|ill with)\b",
    # "you are/'ve been diagnosed with <anything>" is a personal determination
    # regardless of the condition named (covers conditions outside the disease
    # list, e.g. brucellosis).
    r"\byou\s+(?:are|'ve been|have been|were|'re)\s+diagnosed\s+with\s+\w+",
    r"\byou\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+|clearly\s+"
    r"|most likely\s+)?(?:have|'ve got)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:disease|infection|virus|illness)\b",
    # Direct disease attribution to the user: "you have dengue", "you have the
    # flu", "you've got measles", "you probably have malaria". An optional hedge
    # (probably/likely/definitely) — before OR after the verb — is still a
    # personal determination. A leading negation is excluded by the
    # negation-window check in detect_personal_medical_claim.
    r"\byou\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+|clearly\s+"
    r"|most likely\s+)?"
    r"(?:have|'ve|are\s+diagnosed\s+with)\s*"
    r"(?:probably\s+|likely\s+|definitely\s+|certainly\s+|clearly\s+|"
    r"most likely\s+)?"
    r"(?:got\s+|gotten\s+)?"
    r"(?:a\s+case\s+of\s+|an?\s+|the\s+)?"
    r"(?:" + "|".join(re.escape(_d) for _d in _PERSONAL_DISEASE_NAMES) + r")\b",
    # Contraction form: "you've got dengue" / "you're diagnosed with measles".
    r"\byou'(?:ve|re)\s+(?:got\s+|gotten\s+|been\s+diagnosed\s+with\s+|"
    r"diagnosed\s+with\s+)?"
    r"(?:probably\s+|likely\s+)?(?:a\s+case\s+of\s+|an?\s+|the\s+)?"
    r"(?:" + "|".join(re.escape(_d) for _d in _PERSONAL_DISEASE_NAMES) + r")\b",
    # Direct attribution of a common named condition OUTSIDE the disease list
    # that carries no recognisable medical suffix (diabetes, asthma, epilepsy,
    # hypertension, ...). Population statements ("people with diabetes ...") and
    # topic statements ("you have questions about asthma") are not matched
    # because this requires "you have/'ve got <condition>" directed at the user.
    r"\byou(?:\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+|clearly\s+"
    r"|most likely\s+)?(?:have|have got)|'ve\s+got)\s+"
    r"(?:probably\s+|likely\s+)?(?:a\s+case\s+of\s+|an?\s+|the\s+)?"
    r"(?:" + "|".join(re.escape(_c) for _c in _PERSONAL_CONDITION_NAMES) + r")\b",
    # "you are in a coma" — state-of-being attribution of a condition to the user
    # (the "you have <condition>" pattern above does not cover "are in a ...").
    r"\byou\s+(?:are|'re|are\s+(?:probably|likely|definitely)|"
    r"were|have\s+been)\s+in\s+(?:a\s+|an\s+|the\s+)?coma\b",
    # predicting the user will get infected. An optional adverb
    # (probably/likely/definitely/certainly/soon) may sit between "will" and the
    # verb, and "become infected" is included alongside get/catch/contract.
    r"\byou\s+will\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+|soon\s+|"
    r"most likely\s+)?(?:get|catch|contract|develop|become|be)\s+"
    r"(?:infected|ill|sick|the\s+\w+)\b",
    r"\byou\s+will\s+(?:probably\s+|likely\s+)?be infected with\b",
    r"\byou\s+are\s+(?:definitely|certainly|likely|going to be)\s+infected\b",
    r"\byou\s+(?:are|'re)\s+(?:probably|likely|going to be)\s+"
    r"(?:going to\s+)?(?:get|catch|become)\s+(?:infected|ill|sick)\b",
    # treatment as authorized for the user. Generic determiners
    # (this/that/the/a) are included so "you should take this treatment" is
    # caught, not only named drugs. General wellbeing advice ("you should
    # consult a doctor", "you should get vaccinated", "you should rest") is NOT
    # matched because it does not name a medication/drug/treatment object.
    r"\byou\s+(?:should|can|may|could|must|need to|have to|ought to|"
    r"will\s+need to)\s+(?:take|start|begin|use|be prescribed|"
    r"be given|get on)\s+"
    r"(?:this|that|the|these|those|a|an|some\s+)?\s*"
    r"(?:antibiotics?|antivirals?|medications?|medicines?|drugs?|treatments?|"
    r"pills?|doses?|prescriptions?|therapy|"
    r"amoxicillin|azithromycin|doxycycline|oseltamivir|tamiflu)\b",
    # Object stated directly without an explicit "take/use" verb, with or without
    # a determiner: "you need this drug", "you require these medications",
    # "you need antibiotics", "you require treatment". A determiner is optional
    # here because "you need <medication object>" is itself an individualized
    # authorization. General education ("treatment options include antibiotics",
    # "patients should discuss treatment") is not matched: it is not "you
    # need/require ...". "you can learn more about treatment" is not matched
    # because the verb is learn, not need/require/take.
    r"\byou\s+(?:need|require|will\s+need)\s+"
    r"(?:this|that|the|these|those|a|an|some\s+)?\s*"
    r"(?:antibiotics?|antivirals?|medications?|medicines?|drugs?|treatments?|"
    r"treatment|pills?|doses?|prescriptions?|therapy)\b",
    # "you should/can need this drug" style with mandatory determiner (keeps
    # "you should rest" etc. unmatched, which lack a medication object).
    r"\byou\s+(?:should|can|may|could|must)\s+"
    r"(?:this|that|the|these|those|a|an|some\s+)\s*"
    r"(?:antibiotics?|antivirals?|medications?|medicines?|drugs?|treatments?|"
    r"pills?|doses?|prescriptions?|therapy)\b",
    # "you can start/need/take treatment" — bare treatment noun with an
    # authorization verb (take/start/begin/get/receive).
    r"\byou\s+(?:should|can|may|could|must|need to|have to)\s+"
    r"(?:take|start|begin|get|receive)\s+treatment\b",
    r"\bi\s+(?:diagnose|prescribe)\s+you\b",
    # individualized infection risk stated as fact
    r"\byour\s+(?:personal\s+)?(?:risk of infection|infection risk)\s+is\s+"
    r"(?:high|low|elevated|certain|guaranteed)\b",
)

# Suffix-based detector for out-of-lexicon conditions ("you have brucellosis").
# The condition token is captured so it can be validated against a non-medical
# stoplist and a minimum stem length before it is treated as a determination —
# this is what keeps "You have a diploma."/"You have an aroma." unflagged
# without deleting suffix support entirely.
_PERSONAL_SUFFIX_PATTERN = re.compile(
    r"\byou(?:\s+(?:probably\s+|likely\s+|definitely\s+|certainly\s+|clearly\s+"
    r"|most likely\s+)?(?:have|have got)|'ve\s+got)\s+"
    r"(?:probably\s+|likely\s+)?(?:a\s+case\s+of\s+|an?\s+|the\s+)?"
    r"(?P<cond>[a-z]+(?:osis|itis|aemia|emia|caemia|opathy|pathy|oma|iasis|"
    r"coccus|ococci))\b"
)
# A genuine condition stem is reasonably long; very short tokens ending in a
# suffix string (e.g. "oma", "roma") are almost always ordinary words.
_MIN_CONDITION_LENGTH = 6

_PMID_PATTERN = re.compile(r"\bPMID[:\s]*([0-9]{4,9})\b", re.IGNORECASE)
_PUBMED_URL_PATTERN = re.compile(
    r"pubmed\.ncbi\.nlm\.nih\.gov/([0-9]{4,9})", re.IGNORECASE
)
# Preceding window (characters) scanned for a negation cue before a match.
_NEGATION_WINDOW = 45


@dataclass
class ValidationResult:
    """Outcome of deterministic post-generation validation and enforcement."""

    text: str
    warnings: list[str] = field(default_factory=list)
    action: SafetyAction = SafetyAction.PASS

    @property
    def ok(self) -> bool:
        """True when no safety warnings were raised."""
        return not self.warnings

    @property
    def enforced(self) -> bool:
        """True when the output was altered (redacted or withheld)."""
        return self.action is not SafetyAction.PASS


def _known_pmids(evidence: list[Evidence]) -> set[str]:
    pmids: set[str] = set()
    for item in evidence:
        pmid = str(item.metadata.get("pmid", "")).strip()
        if pmid:
            pmids.add(pmid)
        if item.uri:
            for match in _PUBMED_URL_PATTERN.findall(item.uri):
                pmids.add(match)
    return pmids


def _is_negated(text_lower: str, start: int) -> bool:
    """Return True if a negation cue applies to the phrase beginning at ``start``.

    The scan is sentence/clause-local: it only looks back to the start of the
    current sentence (the most recent ``.``/``!``/``?`` before ``start``), so a
    negation in a *previous* sentence cannot suppress a positive determination in
    this one. For example, in "You do not have diabetes. You have brucellosis."
    the second clause is NOT considered negated. A fixed character cap is still
    applied as an upper bound within very long sentences. Commas are intentionally
    not treated as boundaries so "consistent with, but do not confirm, malignancy"
    remains correctly suppressed.
    """
    boundary = 0
    for terminator in (".", "!", "?"):
        idx = text_lower.rfind(terminator, 0, start)
        if idx + 1 > boundary:
            boundary = idx + 1
    window_start = max(boundary, start - _NEGATION_WINDOW)
    window = text_lower[window_start:start]
    return any(cue in window for cue in _NEGATION_CUES)


def detect_overconfident_diagnosis(text: str) -> list[str]:
    """Return definitive-diagnosis phrases found in ``text`` (may be empty).

    Negated statements (e.g. "cannot confirm cancer", "this does not show a
    tumor") are excluded so conservative, correctly-hedged language is not
    flagged.
    """
    found: list[str] = []
    lowered = text.lower()
    for pattern in _DEFINITIVE_PATTERNS:
        for match in re.finditer(pattern, lowered):
            if _is_negated(lowered, match.start()):
                continue
            found.append(match.group(0).strip())
    return found


def detect_personal_medical_claim(text: str) -> list[str]:
    """Return phrases that make a personal medical determination about the user.

    These are population/individual boundary violations (§18): telling the user
    they are infected, predicting they will be infected, presenting treatment as
    authorized for them, or stating an individualized infection risk as fact.
    Negated statements ("you do not have ...") are excluded.
    """
    found: list[str] = []
    lowered = text.lower()
    for pattern in _PERSONAL_MEDICAL_PATTERNS:
        for match in re.finditer(pattern, lowered):
            if _is_negated(lowered, match.start()):
                continue
            found.append(match.group(0).strip())

    # Conservative suffix-based detection for out-of-lexicon conditions. The
    # candidate token must (a) not be a known non-medical word, and (b) be long
    # enough to plausibly be a condition, so ordinary words such as "diploma",
    # "aroma", "empathy" or "coma" are not treated as personal determinations.
    for match in _PERSONAL_SUFFIX_PATTERN.finditer(lowered):
        if _is_negated(lowered, match.start()):
            continue
        cond = match.group("cond")
        if cond in _NON_MEDICAL_SUFFIX_WORDS:
            continue
        if len(cond) < _MIN_CONDITION_LENGTH:
            continue
        found.append(match.group(0).strip())

    return found


def detect_fabricated_citations(text: str, evidence: list[Evidence]) -> list[str]:
    """Return PMIDs cited in ``text`` that are absent from retrieved evidence."""
    known = _known_pmids(evidence)
    cited = set(_PMID_PATTERN.findall(text))
    cited.update(_PUBMED_URL_PATTERN.findall(text))
    return sorted(cited - known)


def _redact_fabricated_citations(text: str, fabricated: set[str]) -> str:
    """Remove sentences/fragments that cite a fabricated PMID.

    We neutralize the specific fabricated identifiers rather than deleting whole
    paragraphs, replacing them with an explicit marker so the reader can see a
    citation was removed instead of silently trusting a real-looking one.
    """
    if not fabricated:
        return text

    def _pmid_sub(match: re.Match) -> str:
        pmid = match.group(1)
        return "[unverified citation removed]" if pmid in fabricated else match.group(0)

    def _url_sub(match: re.Match) -> str:
        pmid = match.group(1)
        return "[unverified citation removed]" if pmid in fabricated else match.group(0)

    text = _PMID_PATTERN.sub(_pmid_sub, text)
    text = _PUBMED_URL_PATTERN.sub(_url_sub, text)
    return text


def validate_response(
    text: str,
    evidence: list[Evidence] | None = None,
    observations: list[ImageObservation] | None = None,
) -> ValidationResult:
    """Deterministically validate AND enforce the safety contract on an answer.

    Enforcement policy (in priority order):

    1. **Diagnostic overreach → WITHHELD.** If the answer asserts a definitive
       diagnosis, the substantive answer is not returned. It is replaced with a
       conservative message; the structured observations/evidence remain intact.
    2. **Fabricated citations → REDACTED.** PMIDs not present in the retrieved
       evidence are removed from the text and replaced with an explicit marker.
    3. **Otherwise → PASS.** Safe output is returned unchanged, with a standing
       safety notice appended if the answer lacks one.

    Warnings are always machine-readable and recorded regardless of action, so a
    caller can see *why* enforcement occurred. The safety notice never claims the
    output is thereby "safe"; it states the informational, non-diagnostic
    boundary.
    """
    evidence = evidence or []
    warnings: list[str] = []

    overconfident = detect_overconfident_diagnosis(text)
    personal = detect_personal_medical_claim(text)
    fabricated = detect_fabricated_citations(text, evidence)

    if overconfident:
        warnings.append(
            "Definitive-diagnosis language detected; answer withheld: "
            + "; ".join(sorted(set(overconfident)))
        )
    if personal:
        warnings.append(
            "Personal medical determination detected; answer withheld: "
            + "; ".join(sorted(set(personal)))
        )
    if fabricated:
        warnings.append(
            "Citations not present in retrieved evidence (removed): PMID "
            + ", PMID ".join(fabricated)
        )

    # Priority 1: withhold on diagnostic overreach.
    if overconfident:
        return ValidationResult(
            text=WITHHELD_DIAGNOSIS_MESSAGE + SAFETY_NOTICE,
            warnings=warnings,
            action=SafetyAction.WITHHELD,
        )

    # Priority 1b: withhold on personal medical determination (§18).
    if personal:
        return ValidationResult(
            text=WITHHELD_PERSONAL_MESSAGE + SAFETY_NOTICE,
            warnings=warnings,
            action=SafetyAction.WITHHELD,
        )

    # Priority 2: redact fabricated citations, preserve the rest.
    if fabricated:
        cleaned = _redact_fabricated_citations(text, set(fabricated))
        if "safety note:" not in cleaned.lower():
            cleaned = cleaned.rstrip() + SAFETY_NOTICE
        return ValidationResult(
            text=cleaned, warnings=warnings, action=SafetyAction.REDACTED
        )

    # Priority 3: pass through, ensuring a safety notice is present.
    result_text = text
    if "safety note:" not in text.lower():
        result_text = text.rstrip() + SAFETY_NOTICE
    return ValidationResult(text=result_text, warnings=warnings, action=SafetyAction.PASS)
