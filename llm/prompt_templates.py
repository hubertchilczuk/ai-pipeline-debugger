"""Prompt templates for log analysis."""
from __future__ import annotations

SYSTEM_ANALYZE = """You are an expert Site Reliability Engineer specializing in data pipelines
(Apache Spark, Apache Airflow, dbt). Diagnose failures from log excerpts.

Always reply with STRICT JSON matching this schema:
{
  "error_type": "<short canonical name, e.g. OutOfMemoryError>",
  "root_cause": "<one paragraph>",
  "suggested_fix": "<actionable steps, numbered>",
  "severity": "low|medium|high|critical",
  "confidence": <float 0.0-1.0, how sure are you>,
  "tags": ["<tag1>", "<tag2>"]
}
Do not include any text outside the JSON object.
"""

USER_ANALYZE_TEMPLATE = """Pipeline: {pipeline}
Stage: {stage}
Timestamp: {timestamp}

--- LOG EXCERPT ---
{log_excerpt}
--- END LOG ---

Similar past incidents (most relevant first):
{retrieved_context}
"""


SYSTEM_PARSE_STRUCTURE = """You extract structured fields from a single multi-line log block.
Reply STRICT JSON with keys: error_class, message, file, line, stack_summary.
If a field is unknown, use null. No prose."""
