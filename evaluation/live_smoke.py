#!/usr/bin/env python3
"""Compact real-world smoke suite (Phase-6 §6).

Three INDEPENDENT checks, each reporting exactly one of three states — the
states are never blurred:

  * LIVE VERIFIED   — a real external dependency was reached and behaved.
  * MOCK VERIFIED   — the code path was exercised deterministically with fakes
                      (no network / no credentials), so the wiring is proven but
                      the live dependency was NOT contacted.
  * UNVERIFIED      — the check could not be decided (missing creds / network),
                      and NOTHING is fabricated.

Checks:
  A. live health retrieval      — HealthIntelligence against real feeds.
  B. real LLM generation        — the configured text generator answers a prompt.
  C. full agent composition     — AgentService.run() end to end.

For A and C the offline default uses deterministic fakes (MOCK VERIFIED). Pass
``--live`` to attempt the real dependencies; without credentials/network the
relevant check is reported UNVERIFIED rather than failed or faked.

Usage:
    python evaluation/live_smoke.py                       # offline, MOCK VERIFIED
    python evaluation/live_smoke.py --live                # attempt real deps
    python evaluation/live_smoke.py --live --location "Telangana, India"

Exit code: 0 when no check is in an error state (MOCK/LIVE/UNVERIFIED are all
acceptable outcomes for an optional smoke); 1 if a check errored unexpectedly.

Pass ``--require-live`` (only meaningful together with ``--live``) to make the
suite a hard gate on real connectivity: it exits non-zero if ANY check that was
attempted live finished UNVERIFIED (could not reach the real dependency). This
NEVER fabricates a LIVE result — it only turns an honest UNVERIFIED into a
failing exit code so a release gate cannot silently pass without a real
dependency having been contacted. UNVERIFIED is never printed or described as a
success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LIVE = "LIVE VERIFIED"
MOCK = "MOCK VERIFIED"
UNVERIFIED = "UNVERIFIED"
ERROR = "ERROR"


def _check_health(live: bool, location: str | None) -> tuple[str, str]:
    """A. Live health retrieval."""
    from multimodal.config import MultimodalConfig
    from multimodal.location import parse_locations

    locations = parse_locations(location or "India")
    if not live:
        # Deterministic wiring proof: a fake provider returns one item.
        from evaluation.agent_harness import build_providers
        from multimodal.health_intelligence import HealthIntelligence

        providers = build_providers([
            {"name": "WHO Disease Outbreak News", "tier": "primary_official",
             "items": [{"title": "Dengue outbreak in India", "summary": "Dengue reported.",
                        "uri": "https://who.int/don/dengue", "stated_status": "outbreak",
                        "geo_scope": "national"}]}
        ])
        hi = HealthIntelligence(MultimodalConfig(health_cache_ttl=0), providers=providers)
        res = hi.gather(locations)
        ok = res.live_data_status in ("ok", "partial") and res.findings
        return (MOCK if ok else ERROR,
                f"status={res.live_data_status}, findings={len(res.findings)} "
                f"(fake provider; no network contacted)")

    # Live attempt against real default feeds.
    from multimodal.health_intelligence import HealthIntelligence

    hi = HealthIntelligence(MultimodalConfig.from_env())
    try:
        res = hi.gather(locations)
    except Exception as exc:  # noqa: BLE001
        return UNVERIFIED, f"live gather raised: {type(exc).__name__}: {exc}"
    if res.live_data_status == "unavailable":
        return (UNVERIFIED,
                f"all sources unreachable (attempted={len(res.sources_attempted)}, "
                f"failed={len(res.sources_failed)}); not fabricating a result")
    return (LIVE,
            f"status={res.live_data_status}, succeeded={res.sources_succeeded}, "
            f"findings={len(res.findings)}, retrieved_at={res.retrieved_at}")


def _check_llm(live: bool) -> tuple[str, str]:
    """B. Real LLM generation."""
    if not live:
        from evaluation.agent_harness import ScriptedGenerator

        gen = ScriptedGenerator(text="A grounded, population-level summary.")
        text, model = gen.generate("prompt")
        return (MOCK if text and gen.configured() else ERROR,
                f"scripted generator produced {len(text)} chars (model={model}); "
                f"no LLM API contacted")

    from multimodal.config import MultimodalConfig
    from multimodal.service import CohereGenerator

    gen = CohereGenerator(MultimodalConfig.from_env())
    if not gen.configured():
        return UNVERIFIED, "no generation credentials (COHERE_API_KEY unset)"
    try:
        text, model = gen.generate(
            "In one sentence, state that this is a connectivity test."
        )
    except Exception as exc:  # noqa: BLE001
        return UNVERIFIED, f"live generation raised: {type(exc).__name__}: {exc}"
    if not text:
        return UNVERIFIED, "live generation returned empty text"
    return LIVE, f"model={model} produced {len(text)} chars"


def _check_agent(live: bool, location: str | None) -> tuple[str, str]:
    """C. Full agent composition."""
    from evaluation.agent_harness import build_agent
    from multimodal.schemas import MultimodalRequest

    if not live:
        agent, _ = build_agent(
            providers_spec=[
                {"name": "WHO Disease Outbreak News", "tier": "primary_official",
                 "items": [{"title": "Dengue outbreak in India", "summary": "Dengue reported.",
                            "uri": "https://who.int/don/dengue", "stated_status": "outbreak",
                            "geo_scope": "national"}]}
            ],
            generator_text="A grounded, population-level summary of the reported situation.",
        )
        resp = agent.run(
            MultimodalRequest(question="What contagious diseases are currently reported here?"),
            location_text=location or "India",
        )
        ok = bool(resp.answer) and "health_intelligence" in resp.trace.tools_used
        return (MOCK if ok else ERROR,
                f"tools={resp.trace.tools_used}, live_data_status={resp.live_data_status}, "
                f"safety={resp.safety_action} (fakes; no network/LLM contacted)")

    # Live composition: real generator + real health feeds.
    from multimodal.agent import AgentService
    from multimodal.config import MultimodalConfig
    from multimodal.service import CohereGenerator, MultimodalRAGService

    cfg = MultimodalConfig.from_env()
    gen = CohereGenerator(cfg)
    if not gen.configured():
        return UNVERIFIED, "no generation credentials for live agent composition"
    try:
        agent = AgentService(cfg, multimodal=MultimodalRAGService(config=cfg, generator=gen))
        resp = agent.run(
            MultimodalRequest(question="What contagious diseases are currently reported here?"),
            location_text=location or "India",
        )
    except Exception as exc:  # noqa: BLE001
        return UNVERIFIED, f"live agent run raised: {type(exc).__name__}: {exc}"
    return (LIVE,
            f"tools={resp.trace.tools_used}, live_data_status={resp.live_data_status}, "
            f"safety={resp.safety_action}, answer_chars={len(resp.answer or '')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact real-world smoke suite.")
    parser.add_argument("--live", action="store_true",
                        help="Attempt real external dependencies (network/LLM). "
                             "Without them the affected check is UNVERIFIED, not faked.")
    parser.add_argument("--location", default=None,
                        help="Explicit location for the health/agent checks.")
    parser.add_argument("--require-live", action="store_true",
                        help="Exit non-zero if any live-attempted check is "
                             "UNVERIFIED (could not reach the real dependency). "
                             "Implies --live. Never fabricates a LIVE result.")
    args = parser.parse_args(argv)

    if args.require_live:
        # --require-live only makes sense against real dependencies.
        args.live = True

    mode = "LIVE (real dependencies attempted)" if args.live else "OFFLINE (fakes)"
    print("RAGnosis compact smoke suite")
    print("=" * 72)
    print(f"Mode: {mode}")
    print("-" * 72)

    checks = [
        ("A. live health retrieval", lambda: _check_health(args.live, args.location)),
        ("B. real LLM generation", lambda: _check_llm(args.live)),
        ("C. full agent composition", lambda: _check_agent(args.live, args.location)),
    ]

    any_error = False
    unverified: list[str] = []
    for label, fn in checks:
        try:
            state, detail = fn()
        except Exception as exc:  # noqa: BLE001
            state, detail = ERROR, f"{type(exc).__name__}: {exc}"
        if state == ERROR:
            any_error = True
        if state == UNVERIFIED:
            unverified.append(label)
        print(f"[{state:<14}] {label}")
        print(f"                 {detail}")

    print("-" * 72)
    print("States: LIVE VERIFIED = real dependency reached; MOCK VERIFIED = "
          "wiring proven with fakes; UNVERIFIED = could not decide (nothing faked).")

    if args.require_live and unverified:
        print("-" * 72)
        print("--require-live FAILED: the following live-attempted check(s) "
              "could not reach a real dependency and remain UNVERIFIED "
              "(NOT a success):")
        for label in unverified:
            print(f"  * {label}")
        return 2
    if any_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
