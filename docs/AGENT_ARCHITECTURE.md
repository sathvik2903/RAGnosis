# RAGnosis Agent & Live Health Intelligence

## What RAGnosis is

> RAGnosis is an evidence grounded biomedical research agent that combines graph
> knowledge, biomedical literature, multimodal observations, and current public
> health information while explicitly preserving source provenance and
> uncertainty.

RAGnosis is **not** a diagnostic system, a diagnostic device, or a clinically
validated tool. It does not diagnose a user, predict whether a user is or will be
infected, infer personal medical status from symptoms or an image, or authorize
treatment.

## Where the agent fits

The agent (`multimodal/agent.py`) is a thin orchestration layer that **composes**
the existing multimodal pipeline with a new live-health-intelligence capability.
It does not replace or fork the existing flow — vision, PubMed retrieval,
generation, and the deterministic safety layer are reused unchanged; live health
intelligence is simply **one more tool** the agent can invoke.

```text
user request (question + optional image + optional explicit location)
        |
        v
  intent + capability routing        (multimodal/routing.py, deterministic, no LLM)
        |
        v
  explicit location resolution       (multimodal/location.py; ASK if missing + geographic)
        |
        +--> vision observations      (multimodal/vision.py)          [if image]
        +--> biomedical literature    (multimodal/retrieval.py)       [if text question]
        +--> live health intelligence (multimodal/health_intelligence.py) [if live intent]
        |
        v
  evidence fusion (provenance kinds retained)   (multimodal/agent.py)
        |
        v
  grounded generation                (multimodal/service.py generator)
        |
        v
  deterministic safety enforcement   (multimodal/safety.py)
        |
        v
  structured answer + provenance + timestamps + execution trace
```

## Routing (deterministic, no second LLM)

`multimodal/routing.py` classifies each request with transparent keyword/pattern
rules — never another LLM (a second model for routing was explicitly avoided).

- **Static / definitional** questions ("what causes dengue?", "symptoms of
  measles") route to **literature only**. They never trigger a live lookup.
- **Current-activity** questions ("is there a dengue outbreak right now?",
  "what's spreading in my area?", "is measles contagious currently?", "health
  alert", "disease trend", "regional threat") additionally invoke **live health
  intelligence**, and are treated as inherently **geographic**.
- An **image** adds the **vision** capability. Combined requests compose vision +
  literature + live health as needed.

The route, the tools used, and the human-readable reasons are surfaced in the
execution trace (below). The router decides *whether to go and look* — it never
decides *what is spreading*.

## Location handling (explicit only)

`multimodal/location.py` enforces the core policy: **location is only ever what
the user explicitly supplies.** It is never inferred from IP address, browser
locale, system settings, account data, or model knowledge.

- The user's original text is always preserved (`raw`) alongside a conservative
  normalization (`normalized`).
- Granularity is kept distinguishable — `city`, `state_or_region`, `country`, or
  `unknown` — and is never upgraded. A national/regional source is never
  presented as if it were city-level surveillance.
- Multiple locations ("Hyderabad and Mumbai", "Telangana; Kerala") are split and
  **evaluated separately**.
- If the question is geographic but **no** location was supplied, the agent
  **asks first** (`needs_location=True` with a clear prompt) instead of guessing.

## Live health intelligence pipeline

`multimodal/health_intelligence.py` actually fetches current public-health
information at request time. It never asks an LLM "what is spreading", and model
training knowledge is never presented as the current situation.

Pipeline: **discover → fetch → parse → normalize → dedupe → geo-relevance →
freshness → conflict resolution → alert classification.**

- **Providers are injectable.** `DiseaseSurveillanceProvider` / `HealthNewsProvider`
  are protocols. The real adapter, `FeedProvider`, reads RSS/Atom feeds over HTTP
  with a strict timeout and a descriptive User-Agent. Tests inject fakes, so the
  entire capability runs offline.
- **Authoritative source priority (tiers).** `primary_official` (WHO / WHO DON,
  CDC, national agencies) and `regional_official` (ECDC, PAHO, state departments)
  take precedence over `secondary` (media) for status and risk. Secondary
  reporting is allowed but always labeled secondary.
- **Source metadata retained per item:** organization, tier, title, URI,
  published/updated dates, per-item and overall `retrieved_at`, and geo scope.

### Per-disease finding fields

Each `DiseaseFinding` reports (using the source's own wording where possible):

- disease name; whether transmissible and the transmission class
  (`contagious_person_to_person`, `not_person_to_person`, `vector_borne`,
  `food_or_water_borne`, `zoonotic`, `environmental`, `unknown`) — **never
  inferred** from an image, a symptom, or the mere presence of a headline;
- status classification (`outbreak`, `epidemic`, `pandemic`, `endemic`,
  `cluster`, `sporadic`, `no_current_outbreak_status_found`, `conflicting`,
  `unknown`) plus the source's own term, classification source, and date;
- geographic scope (`global` / `national` / `regional` / `local` / `unknown`),
  never collapsed into another scope;
- affected areas and relevance to the requested location (`direct`, `regional`,
  `imported_risk`, `global_context`, `unknown`) — a global headline is never
  presented as local;
- official risk if stated, otherwise the literal `"risk assessment unavailable"`;
  severity only if authoritative. **Risk and severity are never manufactured.**
- alert level (`none` / `watch` / `elevated` / `official_alert`) derived only
  from explicit evidence, with the source and reason retained;
- uncertainty / data-gap state (`low_media_coverage`, `limited_surveillance_data`,
  `reporting_delay`, `incomplete_reporting`, `conflicting_reports`,
  `insufficient_local_data`, `uncertain`) — these describe the **evidence**, and
  **never** accuse a source of concealment or downplaying;
- freshness (`published_at`, `updated_at`, `retrieved_at`, and a
  `freshness_state` of `current` / `recent` / `stale` / `unknown`).

### Conflict resolution

When sources disagree on status, the finding is marked `conflicting`, the primary
/ most-recent official source is used as the lead, and a `conflict_summary`
preserves each source's exact wording, tier, scope, and date.

### Freshness

`freshness_state` is computed from the newest of `updated_at` / `published_at`
against configurable thresholds (`HEALTH_CURRENT_DAYS`, `HEALTH_RECENT_DAYS`). A
missing date yields `unknown` — never a guessed "current". The UI shows
`Last checked: <timestamp>`.

### Caching

A short-lived, configurable in-memory cache (`HEALTH_CACHE_TTL`, default 900s;
`0` disables it) avoids hammering public-health infrastructure. Cached results are
flagged (`from_cache`, `cache_hits`) and the real retrieval timestamp is always
shown. Nothing is cached indefinitely.

### Failure semantics (no silent fallback)

- All sources unreachable → `live_data_status="unavailable"`. Model knowledge is
  **not** used as a substitute; the answer says current data could not be
  retrieved.
- Sources reachable but nothing relevant → `no_relevant_current_data`. Absence of
  a report is explicitly **not** treated as proof that no outbreak exists.
- Some sources fail → `partial`, with the failed sources named.

## Evidence fusion

The agent fuses evidence while retaining provenance kinds (`Evidence.kind`):
`image_observation`, `neo4j_evidence`, `pubmed_evidence`, `health_surveillance`
(primary/regional official), and `health_news` (secondary media). The prompt
groups evidence by kind so the model keeps each source type distinct and preserves
uncertainty.

### Stable provenance ids and finding↔evidence linkage (Phase 4)

Every fused `Evidence` object carries a deterministic, content-derived
`evidence_id` (assigned by `schemas.assign_evidence_ids`). Every live-health
`DiseaseFinding` carries a deterministic `finding_id` and the `evidence_ids`
that back it, so the structured response links **finding → evidence → source**
without the model having to invent provenance in prose. Ids are stable across
runs (SHA-1 of content, never random) and survive JSON serialization. The
generation prompt instructs the model to rely on this structured linkage and to
label each statement as OBSERVED (image), RETRIEVED (from a source), CURRENT
(fresh live data), or NOT ESTABLISHED.

## Source ranking (explicit and inspectable)

Findings for the same disease from multiple sources are merged with a documented
precedence, applied in `HealthIntelligence._merge_contributions`:

1. **Tier** — `primary_official` > `regional_official` > `secondary`.
2. **Recency** — within a tier, the most recent `updated_at`/`published_at` leads.
3. **Scope** — geographic relevance to the requested location is preserved per
   source and never collapsed.

The highest-precedence item becomes the *lead* whose status/risk is reported, but
a newer secondary article can **never** silently override a directly relevant
primary-official status: when known statuses differ, the finding status becomes
`conflicting`, a `conflict_summary` preserves each source's organization, tier,
scope, date, and exact wording, and all contributing sources are retained in
`finding.sources`. This is exercised by the `source_quality` evaluation cases.

## Safety (population vs. individual)

The deterministic safety layer (`multimodal/safety.py`) adds a
**personal-medical-determination** check on top of the existing diagnostic-
overreach and fabricated-citation enforcement. If the generated answer tells the
user they are/​will be infected, infers their personal infection status, or
presents treatment as authorized for them, the substantive answer is **withheld**
and replaced with a conservative message. Population- and region-level statements
("cases are rising in the region") are allowed. All checks are pure, deterministic,
and unit-tested — they run *after* generation and change what the user sees, not
just what is logged.

## Execution trace (not chain-of-thought)

Every agent response includes a structured `trace` (§17): `route`, `tools_used`
(including `generation`), `router_reasons`, `retrieval_status`,
`live_data_status`, `locations`, `last_checked`, and `safety_action`. Phase 4
adds declarative execution metadata:

* `sources` — per-source result map, e.g. `{"WHO Disease Outbreak News": "ok",
  "CDC Newsroom": "failed"}` (`ok` / `cache` / `failed`).
* `evidence_counts` — counts keyed by provenance kind, e.g.
  `{"health_surveillance": 4, "pubmed_evidence": 3}`.
* `used_current_data` — whether live data actually informed the answer.
* `cache_hit` — whether any live result was served from the short-lived cache.
* `conflicts_present` — whether contradictory evidence was found.

This is declarative provenance only — it never exposes model chain-of-thought or
hidden reasoning. Example:

```json
{
  "route": "health_intelligence+literature",
  "tools_used": ["literature", "health_intelligence", "generation"],
  "location": "Telangana, India",
  "sources": {"WHO Disease Outbreak News": "ok", "CDC Newsroom": "ok"},
  "evidence_counts": {"health_surveillance": 4, "pubmed_evidence": 3},
  "live_data_status": "partial",
  "used_current_data": true,
  "cache_hit": false,
  "conflicts_present": false,
  "safety_action": "pass"
}
```

## Query-aware retrieval (deterministic, no second LLM)

`HealthIntelligence.gather(locations, query=...)` builds a `QueryContext`
(`build_query_context`) from the user's question and resolved locations. It
splits query tokens into **disease terms** (present in the transparent lexicon),
**keyword terms**, and **location terms** (from the resolved hierarchy — never
inferred). Each finding is scored deterministically (disease match 4 > keyword 2
> location 1); the score sets `matched_query`/`query_score`/`query_relevance_reason`
and drives ordering. This never changes *what* a finding says or *whether* it is
reported — only its priority. There is no second LLM and no web search: if the
configured sources only support feeds, that is documented honestly rather than
pretending arbitrary search exists. The layer still distinguishes **no relevant
finding** (`no_relevant_current_data`) from **source unavailable**
(`unavailable`) from **source has unrelated content** (findings exist but did not
match the query).

## Offline evaluation framework (`evaluation/`)

A separate `evaluation/` package (kept out of the runtime `multimodal` package)
provides a deterministic, offline benchmark:

* `evaluation/cases.py` — an adversarial `BENCHMARK` covering routing, geography,
  status/transmission/alert semantics, source quality (precedence, conflict,
  freshness, availability), disease discovery, safety, and citation grounding.
* `evaluation/evaluators.py` — per-kind evaluators returning `PASS` / `FAIL` /
  `NOT_APPLICABLE` / `UNVERIFIED` (an unexpected error is `UNVERIFIED`, never a
  silent pass).
* `evaluation/runner.py` + `evaluation/run_evaluation.py` — run everything and
  print totals, per-category results, key metric accuracies, and any failures.

Run it with:

```bash
python evaluation/run_evaluation.py
```

It needs no API keys and makes no network calls. The benchmark is also asserted
by `tests/test_evaluation.py` so a regression fails the normal offline suite.

The `evaluation/cases.py` benchmark is a **component** benchmark: it exercises
the routing, semantic, source-quality, safety and citation building blocks in
isolation. Its pass rate is a contract check on those components — it is **not**
a measure of LLM factual accuracy.

### End-to-end agent benchmark

A second, complementary benchmark exercises the **whole composed agent** rather
than individual helpers. It drives the real `AgentService.run()` once per case
through injected deterministic fakes (vision, PubMed retrieval, health feeds and
a scripted text generator) and then applies a fixed battery of *dimension*
evaluators to the resulting `AgentResponse`:

* `evaluation/agent_harness.py` — builds a fully-injected `AgentService`; the
  scripted generator lets a case supply a deliberately faithful *or* unfaithful
  answer so grounding/safety can be checked deterministically and offline. An
  opt-in live mode swaps in the real generator only.
* `evaluation/agent_cases.py` — realistic end-to-end cases: static biomedical,
  current regional, image, combined multimodal, multi-location, unavailable /
  no-relevant / partial live data, conflicting evidence, personal-medical and
  unsafe-output red-team, geographic honesty, and current-data freshness.
* `evaluation/agent_evaluators.py` — eleven dimension evaluators (tool
  selection, location handling, live-data state, answer grounding, citation
  grounding, provenance completeness, uncertainty preservation, safety, conflict
  handling, execution-trace correctness, geographic honesty), each returning
  `PASS` / `FAIL` / `NOT_APPLICABLE` / `UNVERIFIED`. Grounding is measured as
  **fidelity to the supplied fixtures**, never general truth, and includes
  detector self-tests where a deliberately ungrounded answer *must* be flagged.
* `evaluation/agent_runner.py` + `evaluation/run_agent_evaluation.py` — run all
  cases and print an **end-to-end contract pass rate** per dimension.

```bash
python evaluation/run_agent_evaluation.py            # offline, default (CI)
python evaluation/run_agent_evaluation.py --live     # optional real LLM, never CI
```

The offline mode needs no API keys and makes no network calls, and is asserted
by `tests/test_agent_evaluation.py`. `--live` keeps every input deterministic
but uses the real generator to probe whether an actual model stays faithful to
the supplied evidence; it requires credentials, skips the scripted-only
self-tests, and never runs in CI. The reported figure is an end-to-end
**contract** pass rate over deterministic fixtures — it is **not** a claim about
LLM factual accuracy, real-time surveillance coverage, or diagnostic validity.
The health capability here is query-aware **feed retrieval**, not web search.

`AgentEvaluationReport.ok` (and the component report's `ok`) is true only when
there are **zero failed AND zero unverified** cases: an undecidable case is
never silently reported as success.

### Evaluation naming (use precisely)

Three distinct kinds of result, never conflated:

* **component contract pass rate** — `evaluation/run_evaluation.py`, the 63-case
  building-block benchmark.
* **end to end contract pass rate** — `evaluation/run_agent_evaluation.py`, the
  agent-composition benchmark driving the real `AgentService.run()`.
* **live model contract validation** — `evaluation/run_agent_evaluation.py
  --live`, the same contract checked with the real generator when credentials
  exist; reported `UNVERIFIED` when they do not.

None of these is "LLM accuracy", "medical accuracy", or "diagnostic accuracy".
The strongest defensible statement is that RAGnosis has an **end-to-end
contract-tested agent architecture** whose routing, evidence provenance,
uncertainty handling, geographic scoping, citation handling, and medical safety
boundaries are evaluated deterministically, with optional live model and live
source validation when the runtime permits.

#### Grounding evaluator — what it does and does not do

The grounding dimension is **deterministic structured comparison against
controlled fixtures**, not general factual verification. It detects fixture-level
violations: unsupported disease/condition claims (static lexicon AND
source-supplied out-of-lexicon names), fabricated citations (PMIDs absent from
the fused evidence), fabricated dates, absence-as-fact under
unavailable/no-relevant data, active-spread/outbreak certainty without an active
finding, unsupported transmissibility claims, local-presence claims for
imported-only findings, and uncertainty-to-certainty conversion. Its explicit
**limitation**: paraphrase and semantic entailment are not verified; a
determined paraphrase that avoids the tracked tokens could evade it. This is why
grounding is framed as a contract check, never as truth verification.

### Auditable evaluation output

`run_agent_evaluation.py --audit PATH` writes a JSON record per case:
case id, category, question, location, route, tools used, fixture evidence,
model, final answer, safety action, live-data status, and each contract
dimension outcome, plus a UTC timestamp. Records contain **only observable
execution metadata and the final answer** — never chain-of-thought or provider
`reasoning_content`. Live mode writes the same records under the
`live_model_contract_validation` mode label.

### Compact real-world smoke suite

`evaluation/live_smoke.py` runs three independent checks — (A) live health
retrieval, (B) real LLM generation, (C) full agent composition — each reporting
exactly one of `LIVE VERIFIED` (real dependency reached), `MOCK VERIFIED`
(wiring proven with fakes, nothing contacted), or `UNVERIFIED` (could not decide;
nothing fabricated). Offline it is all `MOCK VERIFIED`; with `--live` and no
credentials/network the affected checks are `UNVERIFIED`, never faked or failed.

## API

The composed agent is served at `POST /agent` (the original `/analyze` endpoint is
unchanged). Unlike `/analyze`, the image is **optional** and an explicit
`location` field scopes any regional lookup.

```bash
# text + live health, explicitly scoped
curl -X POST http://localhost:8001/agent \
  -F "question=Is there a current cholera outbreak I should know about?" \
  -F "location=Hyderabad, Telangana, India"

# combined image + literature + live health
curl -X POST http://localhost:8001/agent \
  -F "question=Describe this rash and any current outbreaks in my area" \
  -F "location=India" \
  -F "image=@sample.png"
```

The response includes `answer`, `modality`, `route`, `trace`, `needs_location`,
`location_prompt`, `observations`, `evidence` (with `kind`), `health` (the full
live-intelligence result), `limitations`, `warnings`, `retrieval_status`,
`live_data_status`, `last_checked`, and `safety_action`.

The browser demo at `GET /` (`multimodal/demo.html`) exposes question + optional
image + optional location, and renders the route, tools, per-source status map,
evidence-counts-by-type, current-data/cache/conflict flags, the disease-finding
table (status, transmission, affected areas, relevance, risk, alert, freshness,
sources with tier and excerpt, query-match indicator, finding id), scope,
`Last checked` timestamp, observations, grouped evidence, safety status, and
limitations. A legend labels what was OBSERVED / RETRIEVED / CURRENT / NOT
ESTABLISHED so model interpretation is never shown as source fact.

## Configuration reference (live health)

| Variable | Purpose | Default |
| --- | --- | --- |
| `HEALTH_TIMEOUT` | Per-source HTTP timeout (s) | `15` |
| `HEALTH_USER_AGENT` | User-Agent sent to feeds | `RAGnosis-HealthIntelligence/1.0 (...)` |
| `HEALTH_CACHE_TTL` | Cache lifetime (s); `0` disables | `900` |
| `HEALTH_MAX_ITEMS_PER_SOURCE` | Max items parsed per feed | `40` |
| `HEALTH_CURRENT_DAYS` | Freshness "current" threshold (days) | `14` |
| `HEALTH_RECENT_DAYS` | Freshness "recent" threshold (days) | `60` |
| `HEALTH_SOURCE_FEEDS` | Override feeds: `org|tier|scope|url` per line/`;` | (built-in defaults) |

`tier` ∈ `primary_official` / `regional_official` / `secondary`;
`scope` ∈ `global` / `national` / `regional` / `local` / `unknown`.

When `HEALTH_SOURCE_FEEDS` is empty, the built-in providers are used: the **WHO
Disease Outbreak News JSON provider** (`WHODiseaseOutbreakNewsProvider`, which
calls the OData JSON API — WHO DON has **no** RSS feed) plus the RSS feeds in
`DEFAULT_HEALTH_FEEDS` (CDC Online Newsroom, ECDC Communicable Disease Threats
Report, PAHO/WHO Americas). Setting `HEALTH_SOURCE_FEEDS` replaces the RSS feeds;
the WHO DON JSON provider is always included automatically. Endpoints are
configurable — no endpoint is hardcoded into the analysis logic.

### Source verification log (Phase 3, verified 2026-09-20)

Endpoints were checked out-of-band (the sandbox blocks general outbound HTTPS
from Python, so `live_health_smoke.py` cannot reach them from inside CI — a
Python `SSLError` there means "sandbox egress blocked", not "source down").

| Source | Endpoint | Result |
| --- | --- | --- |
| WHO DON (JSON) | `who.int/api/news/diseaseoutbreaknews` | **LIVE**, current 2026 |
| CDC Newsroom | `tools.cdc.gov/api/v2/resources/media/132608.rss` | **LIVE**, current 2026 |
| ECDC CDTR | `ecdc.europa.eu/en/taxonomy/term/2942/feed` | **LIVE**, current 2026 |
| PAHO Americas | `paho.org/en/rss.xml` | **LIVE**, current 2026 |
| WHO DON RSS | `who.int/feeds/entity/csr/don/en/rss.xml` | **404** (no such feed) |
| NCDC RSS | `idsp.mohfw.gov.in/.../rss.php` | **404** |
| India PIB RSS | `pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3` | LIVE but all-ministry / Hindi / non-standard → secondary only |

Real captured samples from the four live sources are stored under
`tests/fixtures/` so the parsing pipeline is validated against actual source
structure entirely offline (`tests/test_health_live_fixtures.py`).

### India & Telangana coverage (honest status)

- **Out of the box there is NO India- or Telangana-specific official
  machine-readable feed.** India findings surface only when a default source
  (chiefly WHO DON) explicitly names India in an item.
- Authoritative Indian surveillance (IDSP/IHIP weekly bulletins, NCDC) is
  published as **PDFs / dashboards without a stable RSS/JSON feed**, and NCDC's
  historical `rss.php` returns 404 — so none is added as a default (adding an
  unreachable/invented endpoint is worse than none).
- The **India PIB** press-release RSS is genuinely live but is all-ministry (not
  health-scoped) and served in Hindi despite `Lang=1`; it is documented in
  `.env.example` as an **optional, secondary** source to configure, not a default.
- **No official Telangana state health feed** could be verified. A Telangana or
  Hyderabad query therefore reports whatever the configured global/official
  sources say about that scope and **honestly states the coverage gap** rather
  than silently narrowing scope or fabricating local data.
- To add India/Telangana coverage, configure verified official endpoints via
  `HEALTH_SOURCE_FEEDS`. Do not claim "India coverage" merely because the feed
  list is configurable.

## Optional live smoke test

`scripts/live_health_smoke.py` is the **only** code path that touches the network,
and only when run explicitly. It is **not** part of CI.

```bash
python scripts/live_health_smoke.py --location "Hyderabad, Telangana, India"
python scripts/live_health_smoke.py --location "India" --location "Brazil"
python scripts/live_health_smoke.py --location "Telangana, India" \
    --query "What contagious diseases are currently reported in this region?"
```

It runs a **per-source probe** (organization, source URL, HTTP/result status,
item count, newest publication/update dates) and then the full pipeline
(retrieval timestamp, sources attempted/succeeded/failed, normalized findings
with disease name, classification, geo scope, location relevance, transmission,
risk, alert level, freshness, uncertainty, and provenance). The optional
`--query` prioritises query-relevant findings deterministically. If a source is
unreachable it is reported as **failed** and unavailable sources are never turned
into success — nothing is fabricated.

## Testing

All tests run offline by default with injected fake providers
(`tests/test_health_intelligence.py`, `tests/test_agent.py`,
`tests/test_routing.py`, `tests/test_location.py`,
`tests/test_safety_personal.py`, `tests/test_agent_api.py`). They cover location
behaviors, freshness, timestamps, disease discovery, transmission parsing, status
classification, geo-scope, source precedence, conflict resolution, dedupe,
underreported/secondary-only handling, data-gap states, all-unavailable,
partial-failure, no-relevant-data, alert classification, provenance retention,
routing, combined flows, population-vs-individual safety, and the API contract.

## Limitations

- Live public-health information reflects only the sources reachable at the shown
  retrieval time. It is **not** real-time surveillance (a mathematical claim we do
  not make) and **not** globally exhaustive coverage.
- Absence of a current report is not proof that no outbreak exists.
- Current status and risk come from official sources when available; RAGnosis
  never invents a severity or risk score, and never asserts intentional
  concealment or downplaying without an authoritative source.
- RAGnosis reports population- and region-level information; it does not assess an
  individual's infection status and is not a diagnostic system.
