# Experimental and historical RAGnosis work

This directory is not the Render production application.

Production is the Flask service at the repository root (`app.py`, `index.html`). It uses Neo4j Aura and Cohere over environment variables. Do not run these notebooks or Colab exports as the web service.

## What is here

| File | Role |
| --- | --- |
| `MRDRAG(OriginalButCustomisedPipeline)Encrypted.ipynb` | Colab RAG experiment against Neo4j Aura with a Groq LLM backend, enhanced keyword search, and multi-hop traversal. |
| `RAG(OriginalButCustomisedPipelinePrototype)Encrypted.ipynb` | Earlier prototype of the same Neo4j RAG pipeline. |
| `EmbeddingsEnrichmentEncrpted.py` | Exported Colab graph enrichment pipeline. |
| `GraphReconstructionEncrypted.py` | Exported Colab graph reconstruction pipeline. |
| `MRD-RAG-MileStone1.drawio` | Architecture sketch from milestone 1. |
| `Milestone1(MRD-RAG).gslides` | Google Slides stub from milestone 1. |

These artifacts document how the biomedical graph was built and how RAG was prototyped. They are reference material.

## How this relates to production

```
experiments (this folder)          production (repo root)
-------------------------          ----------------------
Colab / notebook RAG               Flask app.py
Groq (historical prototype)        Cohere Chat API
Graph build / enrichment           Remote Neo4j Aura queries
Not served by gunicorn             gunicorn app:app
```

Shared idea: retrieve medical context from Neo4j, then generate an answer with an LLM. The deployed service is the Flask text chat only.

## Environment variables

If you rerun experimental pipelines, use the same names as production. Do not put secrets in these files.

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `COHERE_API_KEY` (production chat)
- `ENRICHMENT_DIR` (optional; graph CSVs default to `Database/EnrichmentReport`)

Older notebook cells may still mention Colab paths or other LLM backends. Treat those as historical. They are not the Render configuration.

## Graph data

Offline graph files live in `Database/` at the repository root. Similarity ingestion is `scripts/EmbeddingsIngestion.py`, which reads `NEO4J_*` from the environment and CSVs from `Database/EnrichmentReport`.
