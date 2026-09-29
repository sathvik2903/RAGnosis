"""Deterministic harness that drives the REAL ``AgentService.run()`` end to end.

Phase-5 §2: the end-to-end evaluator must exercise the actual composed agent
path (routing -> tools -> fusion -> generation -> safety -> AgentResponse), not
individual helper functions. This module builds an :class:`AgentService` wired
with injected fakes (vision, PubMed retrieval, health providers, generation) so
the whole flow is deterministic and runs offline with no API keys.

Nothing here calls the network. The only non-fake path is the optional live LLM
mode, which is opt-in and never part of CI (see ``run_agent_evaluation.py``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from multimodal.agent import AgentService
from multimodal.config import MultimodalConfig
from multimodal.health_intelligence import FeedItem, HealthIntelligence
from multimodal.schemas import Evidence, ImageObservation
from multimodal.service import MultimodalRAGService

# Fixed clock so freshness/timestamps are deterministic across the benchmark.
AGENT_EVAL_NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def _clock() -> datetime:
    return AGENT_EVAL_NOW


# ---------------------------------------------------------------------------
# Injected fakes
# ---------------------------------------------------------------------------

class FakeVision:
    """Returns preconfigured observations; never diagnoses."""

    def __init__(self, observations: list[ImageObservation] | None = None):
        self._observations = observations or []

    def configured(self) -> bool:
        return True

    def observe(self, path: str, question: str):
        return list(self._observations), "fake-vision"


class FakeRetriever:
    """Returns preconfigured PubMed/Neo4j-style evidence, or raises to simulate
    an unavailable retriever."""

    def __init__(self, evidence: list[Evidence] | None = None, raise_exc: Exception | None = None):
        self._evidence = evidence or []
        self._raise = raise_exc

    def search(self, question, observations, limit=5, raise_on_error=False):
        if self._raise is not None:
            raise self._raise
        return list(self._evidence)


class ScriptedGenerator:
    """A fake text generator whose output is a pure function of the prompt.

    Two modes:

    * ``text`` — always return this fixed string (used to inject deliberately
      faithful or unfaithful answers for grounding/safety evaluation).
    * ``fn`` — a callable ``prompt -> str`` for prompt-dependent behaviour.

    The generator is where the model would normally sit; making it scripted lets
    the grounding and safety evaluators check what the agent does with a *known*
    answer, deterministically and offline.
    """

    def __init__(self, text: str | None = None, fn: Callable[[str], str] | None = None):
        self._text = text
        self._fn = fn
        self.last_prompt: str | None = None

    def configured(self) -> bool:
        return True

    def generate(self, prompt: str):
        self.last_prompt = prompt
        if self._fn is not None:
            return self._fn(prompt), "scripted-gen"
        return (self._text if self._text is not None else "A grounded, "
                "population-level, evidence-based summary."), "scripted-gen"


class FakeHealthProvider:
    """A configured-feed-style health source (query-aware feed retrieval, NOT web
    search — §13). Returns preconfigured FeedItems or raises to simulate failure."""

    def __init__(self, name: str, tier: str, items: list[FeedItem], fail: bool = False):
        self.name = name
        self.tier = tier
        self._items = items
        self._fail = fail

    def fetch(self, locations):
        if self._fail:
            raise RuntimeError(f"simulated failure: {self.name}")
        return list(self._items)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_feed_item(spec: dict[str, Any], org: str, tier: str) -> FeedItem:
    d = dict(
        organization=org,
        tier=tier,
        title="",
        summary="",
        uri=None,
        published_at="2026-09-15T00:00:00+00:00",
        updated_at="2026-09-15T00:00:00+00:00",
        retrieved_at=AGENT_EVAL_NOW.isoformat(),
        geo_scope="national",
    )
    d.update(spec)
    return FeedItem(**d)


def build_providers(providers_spec: list[dict[str, Any]]) -> list[FakeHealthProvider]:
    providers: list[FakeHealthProvider] = []
    for p in providers_spec:
        items = [make_feed_item(s, p["name"], p["tier"]) for s in p.get("items", [])]
        providers.append(FakeHealthProvider(p["name"], p["tier"], items, p.get("fail", False)))
    return providers


def build_evidence(specs: list[dict[str, Any]]) -> list[Evidence]:
    out: list[Evidence] = []
    for s in specs:
        out.append(Evidence(
            source=s.get("source", "PubMed"),
            title=s.get("title", "study"),
            excerpt=s.get("excerpt", ""),
            uri=s.get("uri"),
            metadata=s.get("metadata", {}),
            kind=s.get("kind", "pubmed_evidence"),
        ))
    return out


def build_observations(specs: list[dict[str, Any]]) -> list[ImageObservation]:
    return [
        ImageObservation(
            label=s.get("label", "observation"),
            description=s.get("description", ""),
            confidence=s.get("confidence"),
            location=s.get("location"),
            caveat=s.get("caveat"),
        )
        for s in specs
    ]


def build_agent(
    *,
    providers_spec: list[dict[str, Any]] | None = None,
    evidence_specs: list[dict[str, Any]] | None = None,
    observation_specs: list[dict[str, Any]] | None = None,
    generator_text: str | None = None,
    generator_fn: Callable[[str], str] | None = None,
    retriever_raises: bool = False,
    live_generator: Any = None,
) -> tuple[AgentService, Any]:
    """Build a fully-injected AgentService plus the scripted generator handle.

    Returns ``(agent, generator)`` so a caller/evaluator can inspect the exact
    prompt the agent built if needed.
    """
    cfg = MultimodalConfig(health_cache_ttl=0)
    # Live mode swaps ONLY the generator for the real model while keeping every
    # other input deterministic (fake retrieval/vision/health). This isolates the
    # question "does a real LLM stay faithful to the supplied evidence?" and is
    # opt-in only (never CI) — see run_agent_evaluation.py --live.
    generator = live_generator or ScriptedGenerator(text=generator_text, fn=generator_fn)
    retriever = FakeRetriever(
        evidence=build_evidence(evidence_specs or []),
        raise_exc=RuntimeError("retriever down") if retriever_raises else None,
    )
    vision = FakeVision(build_observations(observation_specs or []))
    mm = MultimodalRAGService(config=cfg, vision=vision, retriever=retriever, generator=generator)
    providers = build_providers(providers_spec) if providers_spec is not None else []
    health = HealthIntelligence(cfg, providers=providers, clock=_clock)
    agent = AgentService(cfg, multimodal=mm, health=health)
    return agent, generator
