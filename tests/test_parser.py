"""Unit tests for the log parser."""
from __future__ import annotations

import pytest

from ingestion.parser import ExtractionMethod, LogParser


@pytest.mark.asyncio
async def test_parses_python_traceback() -> None:
    log = (
        'Traceback (most recent call last):\n'
        '  File "/app/features.py", line 42, in build\n'
        "    return df['user_id'].fillna(0)\n"
        "KeyError: 'user_id'\n"
    )
    parsed = await LogParser().parse(log)
    assert parsed.error_class == "KeyError"
    assert parsed.file == "/app/features.py"
    assert parsed.line == 42
    assert parsed.extraction_method is ExtractionMethod.REGEX


@pytest.mark.asyncio
async def test_parses_spark_oom() -> None:
    log = (
        "2025-01-12 03:14:21 ERROR Executor: Exception in task 0.0 in stage 4.0\n"
        "org.apache.spark.SparkException: Job aborted due to stage failure"
    )
    parsed = await LogParser().parse(log)
    assert parsed.error_class == "org.apache.spark.SparkException"
    assert "spark" in parsed.tags
    assert parsed.timestamp is not None


@pytest.mark.asyncio
async def test_parses_airflow_exception() -> None:
    log = "airflow.exceptions.AirflowException: Bash command failed. Exit code 2."
    parsed = await LogParser().parse(log)
    assert parsed.error_class == "AirflowException"
    assert "airflow" in parsed.tags


@pytest.mark.asyncio
async def test_unknown_log_without_llm_returns_unknown() -> None:
    parsed = await LogParser().parse("totally unstructured noise without anything useful")
    assert parsed.extraction_method is ExtractionMethod.UNKNOWN
    assert parsed.error_class is None
