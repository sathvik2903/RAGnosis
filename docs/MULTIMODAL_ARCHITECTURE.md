# RAGnosis Multimodal Architecture

## Purpose

The multimodal track extends RAGnosis from text-only biomedical retrieval into an evidence-grounded research workflow that can accept an image together with a question.

> **See also:** [`AGENT_ARCHITECTURE.md`](AGENT_ARCHITECTURE.md) documents the composed agent
> (`POST /agent`) that adds a **live disease & health intelligence** capability on top of this
> pipeline — deterministic routing, explicit-only location handling, authoritative surveillance
> feeds with provenance and freshness, conflict handling, and population-vs-individual safety.
> Live health intelligence is one more tool for the same agent; it does not replace the pipeline
> described here.

The design intentionally separates concerns so that no single stage can silently
turn an uncertain observation into a confident medical claim:

1. **Configuration** (`multimodal/config.py`): every environment-driven setting resolved in one place; no secrets read at import time.
2. **Image inspection** (`multimodal/image.py`): validates the uploaded file, guards against corruption and decompression bombs, records non-sensitive technical metadata and a SHA-256 digest, and normalizes the image for transport.
3. **Vision observation** (`multimodal/vision.py`): a configurable vision-language model produces conservative observations. Observations are not diagnoses. Parsing is a pure function; the HTTP call is injectable.
4. **Evidence retrieval** (`multimodal/retrieval.py`): PubMed is queried using the question and observed concepts. Retrieved records remain explicit evidence objects with titles, excerpts, PMIDs, and URLs. Retrieval degrades gracefully when the network is unavailable.
5. **Grounded generation** (`multimodal/service.py`): the language model receives the question, observations, retrieved evidence, and safety policy. It must distinguish observations, evidence, uncertainty, and conclusions.
6. **Deterministic safety validation and enforcement** (`multimodal/safety.py`): after generation, code (not the model) checks the answer and *enforces* the safety contract. Diagnostic overreach causes the substantive answer to be **withheld** (replaced with a conservative message); fabricated citations are **redacted** from the text. Safe output passes through unchanged. Every decision is recorded as a machine-readable `safety_action` (`pass` / `redacted` / `withheld`) plus warnings.

```text
image + question
       |
       v
 config (env)
       |
       v
 image validation / SHA-256 / normalize
       |
       v
 configurable vision model  --> observations + uncertainty (parsed, clamped)
       |
       v
 biomedical retrieval (PubMed) --> cited evidence  (graceful on failure)
       |
       v
 constrained generation (Cohere, primary + fallback)
       |
       v
 deterministic safety validation --> warnings + safety notice
       |
       v
 evidence-grounded response
```

## Data flow and boundaries

- **Extraction, normalization, computation, decision, explanation, and evaluation are separate modules.** This makes each stage independently testable and keeps the failure surface small.
- **The model never invents numerical or factual grounding.** Confidence values are clamped to `[0, 1]` or dropped; PMIDs cited in the answer are checked against the evidence actually retrieved; retrieval status is passed into the prompt so the model is told explicitly when *not* to cite.
- **Every collaborator is injectable.** `VisionProvider` takes a `poster`, `BiomedicalRetriever` takes a `requests.Session`, and `MultimodalRAGService` takes `vision`, `retriever`, and `generator`. The entire pipeline can therefore be exercised offline with no API keys and no network.

## Why this boundary matters

The project is a research assistant, not a validated diagnostic medical device. The system therefore must not convert a model's visual observation into a definitive disease diagnosis or treatment instruction. A medical-image workflow can have regulatory implications when software is intended to acquire, process, or analyze medical images for clinical purposes, so intended use and validation must remain explicit.

## Preventing hallucination

Three independent mechanisms, none of which trusts the model to police itself:

1. **Prompt discipline** — the shared safety instruction (`safety.build_system_instruction`) forbids diagnosis, fabricated sources, and identity inference, and demands separation of observation vs. evidence vs. uncertainty.
2. **Structured grounding** — observations and evidence are passed as JSON; the retrieval status tells the model whether any literature exists to cite.
3. **Deterministic post-checks with enforcement** — `safety.validate_response` runs after generation and *changes what the user sees*, not just what is logged:
   - **Diagnostic overreach → withheld.** If the answer asserts a definitive diagnosis (negation-aware, so "cannot confirm cancer" is *not* flagged), the substantive answer is not returned; it is replaced with a conservative message while the structured observations/evidence remain available.
   - **Fabricated citations → redacted.** Any PMID not present in the retrieved evidence is removed from the text and replaced with an explicit marker.
   - **Safe output → passes through** unchanged, with a standing safety notice appended if absent.

   These checks are pure functions, fully unit-tested (including false-positive cases such as discussing cancer as a possibility, quoting evidence, and explicitly stating an image cannot confirm a condition), and cannot be overridden by model output. This is a conservative heuristic layer — it reduces overconfident/fabricated output but is **not** a mathematical guarantee of medical safety, and the system never claims that a warning makes unsafe output safe.

## Provider abstraction

`multimodal/vision.py` uses an OpenAI-compatible HTTP interface. The model and endpoint are configuration values:

- `OPENAI_API_KEY`
- `MULTIMODAL_BASE_URL`
- `MULTIMODAL_VISION_MODEL`
- `MULTIMODAL_TIMEOUT`

This permits a hosted VLM today and a local/self-hosted adapter later without changing the orchestration layer. Generation is likewise configurable (`COHERE_MULTIMODAL_RAG_MODEL`, `COHERE_MULTIMODAL_FALLBACK_MODEL`) with automatic fallback when the primary model is unavailable.

## Configuration reference

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Vision provider key | (unset) |
| `MULTIMODAL_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `MULTIMODAL_VISION_MODEL` | Vision model id | `gpt-4.1-mini` |
| `MULTIMODAL_TIMEOUT` | Vision HTTP timeout (s) | `90` |
| `COHERE_API_KEY` | Generation key | (unset) |
| `COHERE_MULTIMODAL_RAG_MODEL` | Primary generation model (Cohere Chat V2) | `command-a-plus-05-2026` |
| `COHERE_MULTIMODAL_FALLBACK_MODEL` | Fallback generation model (Cohere Chat V2) | `command-a-reasoning-08-2025` |
| `PUBMED_TIMEOUT` | PubMed HTTP timeout (s) | `15` |
| `PUBMED_TOOL` / `PUBMED_EMAIL` / `PUBMED_API_KEY` | NCBI etiquette / rate limit | `RAGnosis` / (unset) / (unset) |
| `MULTIMODAL_MAX_UPLOAD_BYTES` | Max upload size | `12582912` |
| `MULTIMODAL_MAX_IMAGE_PIXELS` | Decompression-bomb cap | `40000000` |
| `MULTIMODAL_MAX_VISION_DIMENSION` | Longest edge sent to VLM | `2048` |
| `HEALTH_TIMEOUT` | Live-health per-source HTTP timeout (s) | `15` |
| `HEALTH_CACHE_TTL` | Live-health cache lifetime (s); `0` disables | `900` |
| `HEALTH_CURRENT_DAYS` / `HEALTH_RECENT_DAYS` | Freshness thresholds (days) | `14` / `60` |
| `HEALTH_MAX_ITEMS_PER_SOURCE` | Max items parsed per feed | `40` |
| `HEALTH_USER_AGENT` | User-Agent sent to health feeds | `RAGnosis-HealthIntelligence/1.0 (...)` |
| `HEALTH_SOURCE_FEEDS` | Live-health feeds `org|tier|scope|url` | (built-in defaults) |

See [`AGENT_ARCHITECTURE.md`](AGENT_ARCHITECTURE.md) for the full live-health configuration
reference, source tiers, and the `POST /agent` endpoint.

## Demo flow (for a judge)

1. Set `OPENAI_API_KEY` (vision) and `COHERE_API_KEY` (generation) in the environment — see the configuration reference below. Without them the API still starts and returns clear `503 not configured` responses.
2. Start the API: `python multimodal_api.py` (default port `8001`).
3. Open the API host in a browser (`GET /`) to use the built-in demo page: enter a question, choose an image, and submit. The page shows the answer, the `safety_action`, any safety warnings, image observations, and retrieved evidence with PubMed links.
4. Or call it directly with `curl` (see below).

The production text-chat app (`app.py`, port `8000`) is intentionally separate and text-only; the multimodal capability lives in its own service with its own UI.

## API

Run:

```bash
python multimodal_api.py
```

Browser demo UI:

```text
open http://localhost:8001/
```

Health:

```bash
curl http://localhost:8001/health
```

Analysis:

```bash
curl -X POST http://localhost:8001/analyze \
  -F "question=Describe the observable features that should be reviewed by a clinician" \
  -F "image=@sample.png"
```

The response includes:

- `answer`
- `modality`
- `observations`
- `evidence`
- `limitations`
- `vision_model`
- `generation_model`
- `retrieval_status` (`ok` | `empty` | `unavailable` | `skipped`)
- `warnings` (deterministic safety-validation findings)
- `safety_action` (`pass` | `redacted` | `withheld`)
- `image_metadata`

The API also serves a self-contained browser demo at `GET /` (`multimodal/demo.html`) using same-origin relative requests, so a judge can open the API host directly and use image + question in the browser.

Response fields also include `safety_action` (`pass` / `redacted` / `withheld`) describing what the enforcement layer did.

Status codes: `400` for client input problems (missing question/image, unsupported type, corrupt image), `413` for uploads over the size limit, `503` when a required provider credential is not configured, and `502` for upstream provider/generation failures (with a generic client message; full detail is logged server-side only).

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs entirely offline (no API keys, no network): vision HTTP, PubMed HTTP, and generation are all injected. Coverage spans image validation and normalization (including MIME-follows-content and palette/large-image conversion), VLM JSON parsing (clean, fenced, malformed), PubMed query building and XML parsing (dedup, missing abstracts, malformed, non-JSON, HTTP error), the retrieval `unavailable` vs `empty` distinction, generation primary/fallback behaviour (and the absence of a retry loop), the safety **detection, enforcement, and false-positive** cases, end-to-end orchestration, and the HTTP API contract (400 / 413 / 502 / 503, and non-leaking error messages).

## Failure modes and how they are handled

| Failure | Behaviour |
| --- | --- |
| Missing image / unsupported type / corrupt image | `ImageValidationError` -> HTTP 400 |
| Upload over size limit | HTTP 413 |
| Provider credential not configured | `NotConfiguredError` -> HTTP 503 (names the env var, no secret) |
| Vision returns malformed JSON | Parsed to zero observations; pipeline continues |
| Mislabeled image (e.g. JPEG bytes named `.png`) | Transport MIME follows decoded content, never the extension |
| PubMed unreachable / throttled / non-JSON | `retrieval_status="unavailable"`, empty evidence, generation continues |
| No literature found | `retrieval_status="empty"` (distinct from `unavailable`); model told not to cite |
| Primary generation model unavailable | Automatic fallback model (each model attempted at most once; no retry loop) |
| Non-model generation error (e.g. timeout) | Propagated; fallback is **not** triggered so real errors are not masked |
| Model asserts a definitive diagnosis | Answer **withheld** (`safety_action="withheld"`); observations/evidence retained |
| Model invents a PMID | Citation **redacted** from text (`safety_action="redacted"`) |
| Other upstream/generation failure | HTTP 502 with a generic client message; full detail logged server-side only |

## Important limitations

- The vision adapter is a provider integration, not a clinically validated tumour, cancer, or disease detector.
- No diagnostic performance claim should be made from this repository without task-specific datasets, validation, calibration, and expert review.
- PubMed retrieval improves grounding but does not make retrieved literature patient-specific advice.
- Images are processed temporarily by the API and deleted after the request.
- The current implementation does not persist patient images or biometric representations.
- The safety validator is a conservative heuristic layer: it reduces, but cannot mathematically guarantee the absence of, overconfident or fabricated output.

## How this scales

- Stateless request handling behind gunicorn; workers scale horizontally.
- Injectable collaborators allow swapping a hosted VLM for a self-hosted one, or PubMed for an internal evidence index, without touching orchestration.
- The pure parsing/validation functions are cheap and deterministic, so throughput is bounded by the external providers, which can be pooled, cached, or rate-limited independently.
