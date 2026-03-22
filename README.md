# AI Pipeline Debugger

A tool for debugging AI/ML data pipelines using LLMs and log analysis.

## Status

Early scaffolding. More to come.

## Ingestion

The `ingestion` package collects logs from local files, journald, or
container runtimes and parses them into structured records that the rest
of the pipeline can reason about.

## LLM Providers

Supports OpenAI and a local Ollama provider via a thin `llm.router`
abstraction. Choose a provider per request or fall back automatically.

## Retrieval

Past incidents are embedded into a Qdrant collection. The retriever
returns the top-k most similar prior failures to ground the LLM
explanation.

## API

A FastAPI service exposes `/debug`, `/index`, and `/health`. See
`api/schemas.py` for request/response models.

```bash
uvicorn api.main:app --reload
```
