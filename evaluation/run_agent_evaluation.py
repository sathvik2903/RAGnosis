#!/usr/bin/env python3
"""End-to-end agent evaluation entry point (Phase-5 §10, §14).

Usage:
    python evaluation/run_agent_evaluation.py            # offline, default
    python evaluation/run_agent_evaluation.py --live     # optional real LLM

Offline mode (default) drives the REAL ``AgentService.run()`` through injected
deterministic fakes for vision, PubMed retrieval, health feeds and generation.
It requires NO API keys and makes NO network calls, and is the mode wired into
CI (see tests/test_agent_evaluation.py).

Live mode (--live) is OPT-IN and NEVER part of CI. It keeps every input
deterministic but swaps in the real text generator (Cohere) so you can check
whether an actual model stays faithful to the supplied evidence. It requires the
relevant credentials in the environment; the detector self-test / unsafe-output
red-team cases are skipped in live mode because they depend on scripted text.

Exits 0 when no case FAILED, 1 otherwise. Results are an end-to-end CONTRACT
pass rate over the fixtures, NOT a measure of LLM factual accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.agent_runner import LAST_AUDIT, format_report, run  # noqa: E402


def _build_live_generator():
    """Construct the real text generator, or exit clearly if unavailable."""
    from multimodal.config import MultimodalConfig
    from multimodal.service import CohereGenerator

    cfg = MultimodalConfig.from_env()
    gen = CohereGenerator(cfg)
    if not gen.configured():
        print(
            "ERROR: --live requested but no generation credentials are "
            "configured (set COHERE_API_KEY). Refusing to fabricate live "
            "results.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return gen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real text generator instead of scripted fakes "
        "(opt-in, requires credentials, never CI). This is LIVE MODEL CONTRACT "
        "VALIDATION, not a factual-accuracy measurement.",
    )
    parser.add_argument(
        "--audit",
        metavar="PATH",
        default=None,
        help="Write an auditable JSON record per case (observable execution "
        "metadata and final answers only — never chain-of-thought) to PATH.",
    )
    args = parser.parse_args(argv)

    live_gen = _build_live_generator() if args.live else None
    want_audit = bool(args.audit) or bool(live_gen)
    report = run(live_generator=live_gen, audit=want_audit)
    print(format_report(report, live=bool(live_gen)))

    if args.audit:
        import json

        payload = {
            "mode": "live_model_contract_validation" if live_gen
            else "offline_contract",
            "model": (type(live_gen).__name__ if live_gen else "scripted-gen"),
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "unverified": report.unverified,
            "ok": report.ok,
            "cases": list(LAST_AUDIT),
        }
        with open(args.audit, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nAudit record written to {args.audit} ({len(LAST_AUDIT)} cases).")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
