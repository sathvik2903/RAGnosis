"""Deterministic intent detection and capability routing (§13).

The router decides which capabilities the agent should invoke for a request. It
is a transparent, keyword/pattern based classifier — NOT another LLM (§26). It
never decides *what is spreading*; it only decides *whether to go and look*.

Capabilities:
- ``vision``           : an image was supplied.
- ``literature``       : PubMed retrieval (static biomedical questions).
- ``health_intelligence``: live disease/outbreak/surveillance lookup.

A static question ("what causes dengue?") routes to literature only. A current
question ("is there a dengue outbreak in Hyderabad right now?") additionally
routes to health intelligence and is marked geographic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


# Cues that the user is asking about *current* infectious-disease activity (§13).
_LIVE_HEALTH_CUES = (
    "outbreak",
    "outbreaks",
    "epidemic",
    "pandemic",
    "spreading",
    "spread of",
    "currently spreading",
    "going around",
    "current cases",
    "case surge",
    "surge in cases",
    "health alert",
    "public health alert",
    "disease alert",
    "disease trend",
    "disease trends",
    "current outbreak",
    "recent outbreak",
    "ongoing outbreak",
    "active cases",
    "infection rate",
    "infectious activity",
    "regional threat",
    "health threat",
    "circulating",
    "endemic right now",
    "right now",
    "at the moment",
    "these days",
    "this season",
    "latest",
    "recent",
    "currently",
    "as of now",
    "today",
    "this week",
    "this month",
    "up to date",
    "situation report",
    "situation update",
)

# Words that specifically imply contagion/transmissibility questions (§13).
_CONTAGION_CUES = (
    "contagious",
    "transmissible",
    "person to person",
    "person-to-person",
    "how does it spread",
    "is it spreading",
    "am i at risk",  # handled by safety layer, but still a live-intel cue
    "risk in my area",
    "risk in my region",
)

# Cues that the question is inherently geographic (§1, §13).
_GEO_CUES = (
    "in my area",
    "in my region",
    "in my city",
    "in my state",
    "in my country",
    "near me",
    "around here",
    "locally",
    "region",
    "regional",
    " in ",  # "outbreak in <place>"
    " at ",
)

# Cues that a question is STATIC/definitional and should stay literature-only.
_STATIC_CUES = (
    "what causes",
    "what is",
    "what are the symptoms",
    "symptoms of",
    "how is it treated",
    "treatment for",
    "pathophysiology",
    "mechanism of",
    "definition of",
    "what does it mean",
)


@dataclass(frozen=True)
class RouteDecision:
    """The router's transparent output (part of the execution trace, §17)."""

    route: str  # human-readable label, e.g. "vision+literature+health_intelligence"
    capabilities: list[str] = field(default_factory=list)
    geographic_intent: bool = False
    live_health_intent: bool = False
    reasons: list[str] = field(default_factory=list)

    def uses(self, capability: str) -> bool:
        return capability in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contains_any(text: str, cues: tuple[str, ...]) -> list[str]:
    return [c for c in cues if c in text]


def classify(question: str, has_image: bool, has_location: bool) -> RouteDecision:
    """Classify a request into a :class:`RouteDecision`.

    ``has_location`` reflects whether the user *explicitly* supplied a location;
    it can strengthen the geographic signal but never invents one.
    """
    q = f" {(question or '').lower().strip()} "
    capabilities: list[str] = []
    reasons: list[str] = []

    if has_image:
        capabilities.append("vision")
        reasons.append("image supplied -> vision")

    live_hits = _contains_any(q, _LIVE_HEALTH_CUES)
    contagion_hits = _contains_any(q, _CONTAGION_CUES)
    static_hits = _contains_any(q, _STATIC_CUES)
    geo_hits = _contains_any(q, _GEO_CUES)

    # Live-health intent: explicit "current activity" cues, OR contagion cues,
    # OR an explicit location paired with any disease/health framing. A purely
    # static/definitional question with no live cue stays literature-only.
    live_health_intent = bool(live_hits or contagion_hits)
    if has_location and not live_health_intent and not static_hits:
        # An explicit location with a non-definitional health question implies
        # the user cares about that place's current situation.
        live_health_intent = True
        reasons.append("explicit location with non-static question -> health_intelligence")

    if live_health_intent:
        capabilities.append("health_intelligence")
        if live_hits:
            reasons.append("current-activity cue(s): " + ", ".join(sorted(set(live_hits))))
        if contagion_hits:
            reasons.append("contagion cue(s): " + ", ".join(sorted(set(contagion_hits))))

    # Literature: default for any substantive text question. Always available as
    # grounding, and it is the sole route for static questions.
    if (question or "").strip():
        capabilities.append("literature")
        if static_hits and not live_health_intent:
            reasons.append("static/definitional question -> literature only")
        else:
            reasons.append("text question -> literature grounding")

    # Geographic intent: geo cues OR an explicit location OR live-health intent
    # that is inherently place-scoped (outbreak questions are about somewhere).
    geographic_intent = bool(geo_hits) or has_location
    if live_health_intent and not geographic_intent:
        # "Is there an outbreak right now?" is geographic even without a place;
        # this is what triggers the ASK-for-location behavior downstream (§1).
        geographic_intent = True
        reasons.append("live-health question is inherently geographic")
    elif geo_hits:
        reasons.append("geographic cue(s): " + ", ".join(sorted(set(geo_hits))).strip())

    # De-dupe while preserving order.
    capabilities = list(dict.fromkeys(capabilities))
    route = "+".join(capabilities) if capabilities else "none"

    return RouteDecision(
        route=route,
        capabilities=capabilities,
        geographic_intent=geographic_intent,
        live_health_intent=live_health_intent,
        reasons=reasons,
    )
