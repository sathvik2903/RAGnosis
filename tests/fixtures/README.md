# Source fixtures for offline parser validation

These files let the parsing/normalization pipeline be validated against the
**structure** of real authoritative sources fully offline (no network in CI).

Provenance is stated honestly (Phase-4 §9): the sandbox blocks general outbound
HTTPS from Python, so these are **structurally faithful, representative samples**
modeled on the real endpoints' formats and observed content (via out-of-band
inspection on 2026-09-20), trimmed to a few items and reconstructed. They are
**NOT raw live captures** and must never be presented as current outbreak status.
They exist solely to prove the parsers handle each source's real shape.

| File | Source organization | Endpoint | Capture type | Modeled on |
| --- | --- | --- | --- | --- |
| `who_don_sample.json` | WHO Disease Outbreak News | https://www.who.int/api/news/diseaseoutbreaknews | representative (synthetic structure, real format) | 2026-09-20 |
| `cdc_newsroom_sample.rss` | CDC Online Newsroom | https://tools.cdc.gov/api/v2/resources/media/132608.rss | representative (synthetic structure, real format) | 2026-09-20 |
| `ecdc_cdtr_sample.xml` | ECDC Communicable Disease Threats Report | https://www.ecdc.europa.eu/en/taxonomy/term/2942/feed | representative (synthetic structure, real format) | 2026-09-20 |
| `paho_sample.rss` | PAHO/WHO Americas | https://www.paho.org/en/rss.xml | representative (synthetic structure, real format) | 2026-09-20 |

The four endpoints above were verified reachable and returning current content
out-of-band on 2026-09-20 (see `docs/AGENT_ARCHITECTURE.md`). To validate against
genuinely live bytes, run the optional `scripts/live_health_smoke.py` from an
environment with outbound network access — it is never part of CI.
