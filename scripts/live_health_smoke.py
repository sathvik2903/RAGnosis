#!/usr/bin/env python3
"""Optional live smoke test for the health-intelligence capability (§4, §23).

This is the ONLY code path that hits the network, and only when you run it
explicitly. It is NOT part of the automated test suite / CI. It fabricates
nothing: if a source is unreachable it is reported as failed, and unavailable
sources are never turned into successful status.

Usage:
    python scripts/live_health_smoke.py --location "Hyderabad, Telangana, India"
    python scripts/live_health_smoke.py --location "India" --location "Brazil"
    python scripts/live_health_smoke.py --location "Telangana, India" \
        --query "What contagious diseases are currently being reported here?"

For each source it prints: organization, source URL, HTTP/result status,
publication date, updated date, and number of source items. For each normalized
finding it prints: disease name, classification, geographic scope, location
relevance, risk, alert level, uncertainty, and freshness. It also prints the
overall retrieval timestamp.

Configure sources via HEALTH_SOURCE_FEEDS (see docs) or rely on the built-in
defaults. Respect source terms and rate limits; do not hammer public health
infrastructure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multimodal.config import MultimodalConfig  # noqa: E402
from multimodal.health_intelligence import HealthIntelligence  # noqa: E402
from multimodal.location import parse_locations  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live health-intelligence smoke test.")
    parser.add_argument(
        "--location",
        action="append",
        default=[],
        help="Explicit location (repeatable). E.g. --location 'India'.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional question used only to prioritise relevant findings.",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the per-source diagnostic probe (still runs the full gather).",
    )
    args = parser.parse_args(argv)

    location_text = "; ".join(args.location) if args.location else None
    locations = parse_locations(location_text)

    config = MultimodalConfig.from_env()
    hi = HealthIntelligence(config)

    print("RAGnosis live health-intelligence smoke test")
    print("=" * 72)
    print(f"Requested location(s): {location_text or '(none)'}")
    print(f"Parsed:                {[l.normalized for l in locations] or '(none)'}")
    print(f"Query:                 {args.query or '(none)'}")
    print(f"Cache TTL:             {config.health_cache_ttl}s")

    # --- per-source diagnostics (§4) ---------------------------------------
    if not args.no_probe:
        print("-" * 72)
        print("PER-SOURCE PROBE (live):")
        for r in hi.probe(locations):
            print(f"  * {r['organization']} [{r['tier']}]")
            print(f"      url:              {r['url']}")
            print(f"      result status:    {r['status']}")
            if r["status"] == "ok":
                print(f"      source items:     {r['item_count']}")
                print(f"      newest published: {r['newest_published']}")
                print(f"      newest updated:   {r['newest_updated']}")
            else:
                print(f"      error:            {r['error']}")

    # --- full pipeline -----------------------------------------------------
    result = hi.gather(locations, query=args.query)

    print("-" * 72)
    print(f"Retrieved at:         {result.retrieved_at}")
    print(f"Live data status:     {result.live_data_status}")
    print(f"Sources attempted:    {', '.join(result.sources_attempted) or '(none)'}")
    print(f"Sources succeeded:    {', '.join(result.sources_succeeded) or '(none)'}")
    print(f"Sources failed:       {', '.join(result.sources_failed) or '(none)'}")
    print(f"From cache:           {result.from_cache}")
    print(f"Normalized findings:  {len(result.findings)}")
    for note in result.notes:
        print(f"  note: {note}")
    print("-" * 72)

    for f in result.findings:
        print(f"* {f.disease_name} [{f.status}] scope={f.geo_scope}")
        if f.status_source_term:
            print(f"    source term:   {f.status_source_term!r}")
        print(f"    location:      {f.requested_location} -> {f.relevance_to_location}")
        print(f"    transmission:  {f.transmission_class} (transmissible={f.transmissible})")
        print(f"    risk:          {f.risk_assessment}")
        print(f"    severity:      {f.severity if f.severity is not None else '(none)'}")
        print(f"    alert:         {f.alert_level}")
        print(f"    freshness:     {f.freshness_state}")
        print(f"    published/updated: {f.published_at} / {f.updated_at}")
        if f.conflict_summary:
            print(f"    conflict:      {f.conflict_summary}")
        if f.data_gap_state:
            print(f"    data gap:      {f.data_gap_state}")
        if f.uncertainty:
            print(f"    uncertainty:   {f.uncertainty}")
        for s in f.sources:
            print(
                f"      - {s.organization} ({s.tier}) "
                f"pub={s.published_at} upd={s.updated_at} {s.uri or ''}"
            )

    # Exit code: 0 if any live data was obtained, 2 if all sources unavailable.
    return 0 if result.live_data_status != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
