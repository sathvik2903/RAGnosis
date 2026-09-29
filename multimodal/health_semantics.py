"""Context-aware semantic parsing for public-health source text.

Phase-2 used naive substring matching: any occurrence of the word "outbreak"
became an active outbreak, "airborne" became person-to-person contagion, and
"alert" became an official alert. That produces false classifications on real
feed text such as "no outbreak has been detected", "previous outbreak, now
contained", "possible human-to-human transmission", or "alert lifted".

This module replaces that with **windowed, negation/tense/modality-aware**
classification. For each keyword occurrence it inspects a small surrounding
window for cues that flip or weaken the claim:

* NEGATION      -> the claim is denied ("no outbreak", "ruled out")
* PAST/CONTAINED-> the claim is not current ("previous outbreak, now contained")
* HYPOTHETICAL  -> the claim is unconfirmed ("possible/suspected/investigating")
* LOW/UNLIKELY  -> a risk statement, not an active event ("outbreak risk is low")
* REFERENCE/ELSEWHERE -> the term is discussed, not asserted here
* STRONG AFFIRMATIVE -> explicitly current ("ongoing/confirmed/declared")

Everything is pure and deterministic. When the rules cannot safely determine a
value, the result is an explicitly uncertain state (``unknown`` /
``no_current_outbreak_status_found`` / ``none``) rather than a manufactured
certainty. This is a conservative heuristic layer, not a natural-language
understanding guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .health_schemas import AlertLevel, StatusClassification, TransmissionClass

# Window sizes (characters) scanned around a matched keyword.
_BEFORE = 64
_AFTER = 48


# ---------------------------------------------------------------------------
# Shared cue vocabularies
# ---------------------------------------------------------------------------

# Denial of a claim. Includes both pre-keyword ("no outbreak") and post-keyword
# ("outbreak was ruled out") forms; both windows are searched.
_NEGATION_CUES = (
    "no ",
    "not ",
    "n't",
    "without",
    "never",
    "none",
    "absence of",
    "free of",
    "ruled out",
    "rule out",
    "ruling out",
    "no evidence",
    "no current",
    "no confirmed",
    "no active",
    "no sign",
    "no signs",
    "no known",
    "no reported",
    "no reports of",
    "denies",
    "denied",
    "dismissed",
    "debunk",
    "false report",
    "no longer",
)

# The event happened but is not current.
_PAST_CUES = (
    "previous",
    "previously",
    "past ",
    "former",
    "earlier",
    "prior ",
    "last year",
    "historic",
    "contained",
    "is over",
    "was over",
    "now over",
    "has ended",
    "have ended",
    "ended",
    "concluded",
    "subsided",
    "declared over",
    "under control",
    "brought under control",
    "controlled",
    "resolved",
    "closed",
    "stood down",
    "de-escalated",
    "downgraded",
    "lifted",
    "cancelled",
    "canceled",
    "withdrawn",
)

# The claim is unconfirmed / speculative.
_HYPOTHETICAL_CUES = (
    "possible",
    "possibly",
    "suspected",
    "suspect ",
    "potential",
    "potentially",
    "investigating",
    "investigation",
    "under investigation",
    "probe",
    "reports of a possible",
    "rumour",
    "rumor",
    "rumoured",
    "rumored",
    "alleged",
    "unconfirmed",
    "feared",
    "fears of",
    "concern",
    "concerns",
    "may be",
    "might be",
    "could be",
    "monitoring for",
    "watching for",
    "assessing",
)

# A risk/likelihood statement rather than an asserted active event.
_LOW_RISK_CUES = ("low", "minimal", "negligible", "unlikely", "reduced", "declining")
_RISK_CONTEXT_CUES = ("risk", "likelihood", "chance", "threat level")

# The term is being referenced/discussed, or the event is somewhere else.
_REFERENCE_CUES = (
    "discusses",
    "discussed",
    "guidance on",
    "information on",
    "learn about",
    "about ",
    "regarding",
    "overview of",
    "how to",
    "explainer",
)
_ELSEWHERE_CUES = (
    "another country",
    "other countries",
    "elsewhere",
    "abroad",
    "overseas",
    "in other regions",
    "outside the country",
    "internationally",
)

# Explicitly current -> overrides a weak/hypothetical reading.
_STRONG_AFFIRMATIVE_CUES = (
    "ongoing",
    "confirmed",
    "declared",
    "active",
    "current",
    "currently",
    "underway",
    "escalating",
    "growing",
    "spreading",
    "widening",
    "expanding",
    "surge",
    "surging",
    "reported cases",
    "cases reported",
    "has been reported",
    "have been reported",
)


def _window(text_lower: str, start: int, end: int) -> str:
    return text_lower[max(0, start - _BEFORE): min(len(text_lower), end + _AFTER)]


def _has(window: str, cues: tuple[str, ...]) -> bool:
    return any(cue in window for cue in cues)


@dataclass(frozen=True)
class _Mention:
    category: str  # affirmative|negated|past|hypothetical|low_risk|reference
    mapped: str    # the enum value implied when affirmative
    source_term: str


# ---------------------------------------------------------------------------
# Status (§6)
# ---------------------------------------------------------------------------

# Keyword -> normalized status bucket. Order matters only for term preservation.
_STATUS_KEYWORDS: tuple[tuple[str, StatusClassification], ...] = (
    ("pandemic", "pandemic"),
    ("epidemic", "epidemic"),
    ("outbreak", "outbreak"),
    ("cluster", "cluster"),
    ("endemic", "endemic"),
    ("sporadic", "sporadic"),
    ("isolated case", "sporadic"),
)

# Severity ordering for choosing the lead affirmative status.
_STATUS_RANK = {
    "pandemic": 5,
    "epidemic": 4,
    "outbreak": 3,
    "cluster": 2,
    "endemic": 1,
    "sporadic": 0,
}


def _categorize(window: str) -> str:
    """Classify a keyword mention from its surrounding window."""
    strong = _has(window, _STRONG_AFFIRMATIVE_CUES)
    # Negation is the most decisive signal.
    if _has(window, _NEGATION_CUES):
        return "negated"
    if _has(window, _PAST_CUES):
        return "past"
    # Risk-framed statement with a low/unlikely qualifier.
    if _has(window, _RISK_CONTEXT_CUES) and _has(window, _LOW_RISK_CUES):
        return "low_risk"
    # Hypothetical, unless an explicit "ongoing/confirmed" cue overrides it.
    if _has(window, _HYPOTHETICAL_CUES) and not strong:
        return "hypothetical"
    if _has(window, _REFERENCE_CUES) and not strong:
        return "reference"
    return "affirmative"


def analyze_status(
    term: str | None, text: str
) -> tuple[StatusClassification, str | None, str]:
    """Return (status, source_term, reason) for the source's status claim.

    ``term`` is an optional provider-supplied status term (e.g. from a structured
    field); ``text`` is the free-text title+summary. The source's own wording is
    preserved in the returned ``source_term`` whenever a keyword is present.
    """
    hay = f"{term or ''}. {text}".lower()
    mentions: list[_Mention] = []
    for keyword, mapped in _STATUS_KEYWORDS:
        for m in re.finditer(re.escape(keyword), hay):
            window = _window(hay, m.start(), m.end())
            category = _categorize(window)
            mentions.append(_Mention(category, mapped, keyword))

    preserved_term = term.strip() if term and term.strip() else None
    if not mentions:
        return "unknown", preserved_term, "no status keyword found in source text"

    # Prefer the strongest affirmative mention.
    affirmatives = [mn for mn in mentions if mn.category == "affirmative"]
    if affirmatives:
        lead = max(affirmatives, key=lambda mn: _STATUS_RANK.get(mn.mapped, -1))
        src_term = preserved_term or lead.source_term
        return lead.mapped, src_term, f"affirmative '{lead.source_term}' mention"

    categories = {mn.category for mn in mentions}
    if "hypothetical" in categories:
        return (
            "unknown",
            preserved_term or mentions[0].source_term,
            "status keyword(s) only appear in hypothetical/unconfirmed context",
        )
    # Negated, past, contained, or low-risk mentions only.
    if categories & {"negated", "past", "low_risk"}:
        return (
            "no_current_outbreak_status_found",
            preserved_term or mentions[0].source_term,
            "status keyword(s) only appear negated / past / low-risk",
        )
    # Reference-only mentions -> we cannot assert a current status.
    return (
        "unknown",
        preserved_term or mentions[0].source_term,
        "status keyword(s) only appear in a referential/definitional context",
    )


# ---------------------------------------------------------------------------
# Transmission (§7)
# ---------------------------------------------------------------------------

_CONTAGION_KEYWORDS = (
    "person-to-person",
    "person to person",
    "human-to-human",
    "human to human",
    "respiratory droplet",
    "airborne",
    "contagious",
    "spread between people",
    "spread from person",
)
_VECTOR_KEYWORDS = (
    "mosquito-borne",
    "mosquito borne",
    "mosquito",
    "vector-borne",
    "vector borne",
    "tick-borne",
    "tick borne",
    "sandfly",
    "aedes",
    "anopheles",
)
_FOODWATER_KEYWORDS = (
    "waterborne",
    "water-borne",
    "foodborne",
    "food-borne",
    "contaminated water",
    "contaminated food",
    "faecal-oral",
    "fecal-oral",
)
_ZOONOTIC_KEYWORDS = (
    "zoonotic",
    "zoonosis",
    "animal-to-human",
    "animal to human",
    "spillover",
    "from animals to people",
)
_ENVIRONMENTAL_KEYWORDS = ("environmental exposure", "environmental transmission")

_TRANSMISSION_GROUPS: tuple[tuple[tuple[str, ...], TransmissionClass], ...] = (
    (_CONTAGION_KEYWORDS, "contagious_person_to_person"),
    (_VECTOR_KEYWORDS, "vector_borne"),
    (_FOODWATER_KEYWORDS, "food_or_water_borne"),
    (_ZOONOTIC_KEYWORDS, "zoonotic"),
    (_ENVIRONMENTAL_KEYWORDS, "environmental"),
)


def analyze_transmission(term: str | None, text: str) -> tuple[TransmissionClass, str]:
    """Return (transmission_class, reason) for the source's transmission claim.

    Never inferred from symptoms or an image. A negated contagion statement
    yields ``not_person_to_person``; a hypothetical/referential statement yields
    ``unknown`` rather than a guessed class.
    """
    hay = f"{term or ''}. {text}".lower()

    affirmatives: list[tuple[int, TransmissionClass, str]] = []
    negated_contagion = False
    hypothetical_only = False

    for keywords, cls in _TRANSMISSION_GROUPS:
        for kw in keywords:
            for m in re.finditer(re.escape(kw), hay):
                window = _window(hay, m.start(), m.end())
                strong = _has(window, _STRONG_AFFIRMATIVE_CUES)
                if _has(window, _NEGATION_CUES):
                    if cls == "contagious_person_to_person":
                        negated_contagion = True
                    # A negated vector/food statement is simply not asserted.
                    continue
                if _has(window, _HYPOTHETICAL_CUES) and not strong:
                    hypothetical_only = True
                    continue
                if _has(window, _REFERENCE_CUES) or _has(window, _ELSEWHERE_CUES):
                    if not strong:
                        continue
                affirmatives.append((m.start(), cls, kw))

    if affirmatives:
        affirmatives.sort(key=lambda t: t[0])
        # Prefer a specific non-contagion class if one was explicitly stated,
        # otherwise take the first affirmative in text order.
        for _, cls, kw in affirmatives:
            if cls != "contagious_person_to_person":
                return cls, f"affirmative '{kw}' transmission statement"
        _, cls, kw = affirmatives[0]
        return cls, f"affirmative '{kw}' transmission statement"

    if negated_contagion:
        return "not_person_to_person", "source explicitly denies person-to-person spread"
    if hypothetical_only:
        return "unknown", "transmission mentioned only in a hypothetical/unconfirmed context"
    return "unknown", "no explicit transmission statement found"


# ---------------------------------------------------------------------------
# Alert (§16)
# ---------------------------------------------------------------------------

_ALERT_KEYWORDS: tuple[tuple[str, AlertLevel], ...] = (
    ("public health emergency of international concern", "official_alert"),
    ("pheic", "official_alert"),
    ("public health emergency", "official_alert"),
    ("emergency declared", "official_alert"),
    ("declared an emergency", "official_alert"),
    ("official alert", "official_alert"),
    ("red alert", "official_alert"),
    ("elevated risk", "elevated"),
    ("increased risk", "elevated"),
    ("heightened risk", "elevated"),
    ("high alert", "elevated"),
    ("high risk", "elevated"),
    ("advisory", "watch"),
    ("advised to remain vigilant", "watch"),
    ("remain vigilant", "watch"),
    ("vigilance", "watch"),
    ("watch", "watch"),
    ("monitoring", "watch"),
    ("under surveillance", "watch"),
)

_ALERT_RANK = {"none": 0, "watch": 1, "elevated": 2, "official_alert": 3}


def analyze_alert(text: str, stated_risk: str | None = None) -> tuple[AlertLevel, str | None]:
    """Return (alert_level, reason) from explicit, current source wording.

    A negated or past alert ("no official alert", "alert lifted", "emergency has
    ended") does not raise the level. The highest still-current affirmative
    level wins.
    """
    hay = f"{stated_risk or ''}. {text}".lower()
    best: AlertLevel = "none"
    best_reason: str | None = None
    for kw, level in _ALERT_KEYWORDS:
        for m in re.finditer(re.escape(kw), hay):
            window = _window(hay, m.start(), m.end())
            if _has(window, _NEGATION_CUES) or _has(window, _PAST_CUES):
                continue
            if _ALERT_RANK[level] > _ALERT_RANK[best]:
                best = level
                best_reason = kw
    return best, best_reason
