"""Shared evaluation outcome vocabulary (Phase-4/§7, Phase-5/§4).

Kept in its own module so both the component benchmark
(:mod:`evaluation.evaluators`) and the end-to-end agent benchmark
(:mod:`evaluation.agent_evaluators`) share one definition. Uncertainty is a
first-class outcome (``UNVERIFIED``) and is never silently treated as a pass.
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"
