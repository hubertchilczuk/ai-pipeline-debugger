# AI Pipeline Debugger

A tool for debugging AI/ML data pipelines using LLMs and log analysis.

## Status

Early scaffolding. More to come.

## Ingestion

The `ingestion` package collects logs from local files, journald, or
container runtimes and parses them into structured records that the rest
of the pipeline can reason about.
