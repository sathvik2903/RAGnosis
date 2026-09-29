"""RAGnosis offline evaluation framework (Phase-4 §6-§8).

This package is deliberately kept OUT of the production ``multimodal`` package:
it exercises the shipped code through its public interfaces but never becomes a
runtime dependency of the agent. It is fully offline and deterministic — it uses
injected fake providers and captured fixtures, requires no API keys, and never
makes live network calls.

Public entry points:

* :data:`BENCHMARK` — the adversarial benchmark dataset (a list of
  :class:`~evaluation.cases.EvalCase`).
* :func:`evaluation.runner.run` — evaluate the benchmark and return a
  :class:`~evaluation.runner.EvaluationReport`.
"""

from .cases import BENCHMARK, EvalCase
from .evaluators import CaseResult, Outcome, evaluate_case
from .runner import EvaluationReport, run

__all__ = [
    "BENCHMARK",
    "EvalCase",
    "CaseResult",
    "Outcome",
    "evaluate_case",
    "EvaluationReport",
    "run",
]
