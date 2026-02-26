"""Log parser: regex-first, LLM-fallback for unstructured stack traces.

Strategy
--------
1. Strip ANSI / common timestamp prefixes.
2. Try a battery of compiled regexes for known engines (Spark, Airflow, Python).
3. If none match, optionally hand the snippet to an LLM with a tight extraction prompt.
4. Always return a `ParsedLog` (with `extraction_method` for observability).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final, Protocol

from llm.base import LLMProviderError, LLMRequest, LLMResponse
from llm.prompt_templates import SYSTEM_PARSE_STRUCTURE


class _Generator(Protocol):
    """Anything that exposes `async generate(LLMRequest) -> LLMResponse`.

    Lets the parser accept either a raw provider OR the router (which is not a
    provider but has the same generate() signature).
    """

    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class ExtractionMethod(StrEnum):
    REGEX = "regex"
    LLM = "llm"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ParsedLog:
    error_class: str | None
    message: str
    stack_summary: str | None
    file: str | None
    line: int | None
    timestamp: datetime | None
    raw_excerpt: str
    extraction_method: ExtractionMethod
    tags: list[str] = field(default_factory=list)


_ANSI_RE: Final = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TS_PREFIX_RE: Final = re.compile(
    r"^\[?(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?\s*"
)

# Engine-specific patterns. Order matters: most specific first.
_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "python",
        re.compile(
            r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+).*?\n'
            r"(?P<error_class>[A-Z][\w.]*Error|[A-Z][\w.]*Exception):\s*(?P<message>.+)",
            re.DOTALL,
        ),
    ),
    (
        "spark",
        re.compile(
            r"(?P<error_class>org\.apache\.spark\.[\w.]+(?:Exception|Error))"
            r"(?::\s*(?P<message>[^\n]+))?",
        ),
    ),
    (
        "airflow",
        re.compile(
            r"airflow\.exceptions\.(?P<error_class>\w+)(?::\s*(?P<message>[^\n]+))?",
        ),
    ),
    (
        "java",
        re.compile(
            r"(?P<error_class>(?:java|com|org)\.[\w.$]+(?:Exception|Error))"
            r"(?::\s*(?P<message>[^\n]+))?",
        ),
    ),
]


class LogParser:
    """Parses raw log blocks into `ParsedLog`. Optionally uses an LLM as a fallback."""

    def __init__(self, llm_fallback: _Generator | None = None) -> None:
        self._llm = llm_fallback

    async def parse(self, raw: str) -> ParsedLog:
        cleaned = self._clean(raw)
        timestamp = self._extract_timestamp(cleaned)

        for engine, pattern in _PATTERNS:
            if match := pattern.search(cleaned):
                groups = match.groupdict()
                return ParsedLog(
                    error_class=groups.get("error_class"),
                    message=(groups.get("message") or "").strip(),
                    stack_summary=self._summarize_stack(cleaned),
                    file=groups.get("file"),
                    line=int(groups["line"]) if groups.get("line") else None,
                    timestamp=timestamp,
                    raw_excerpt=cleaned[:4000],
                    extraction_method=ExtractionMethod.REGEX,
                    tags=[engine],
                )

        if self._llm is not None:
            return await self._parse_with_llm(cleaned, timestamp)

        return ParsedLog(
            error_class=None,
            message=cleaned[:200],
            stack_summary=self._summarize_stack(cleaned),
            file=None,
            line=None,
            timestamp=timestamp,
            raw_excerpt=cleaned[:4000],
            extraction_method=ExtractionMethod.UNKNOWN,
        )

    # ---------- helpers ----------
    @staticmethod
    def _clean(raw: str) -> str:
        return _ANSI_RE.sub("", raw).strip()

    @staticmethod
    def _extract_timestamp(text: str) -> datetime | None:
        if match := _TS_PREFIX_RE.search(text):
            ts = match.group("ts").replace(",", ".").replace(" ", "T")
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return None
        return None

    @staticmethod
    def _summarize_stack(text: str, max_lines: int = 8) -> str | None:
        frames = [ln for ln in text.splitlines() if ln.lstrip().startswith(("at ", "File "))]
        if not frames:
            return None
        return "\n".join(frames[:max_lines])

    async def _parse_with_llm(self, cleaned: str, ts: datetime | None) -> ParsedLog:
        request = LLMRequest(
            system_prompt=SYSTEM_PARSE_STRUCTURE,
            user_prompt=cleaned[:4000],
            temperature=0.0,
            max_tokens=400,
            json_mode=True,
        )
        try:
            response = await self._llm.generate(request)
            data = json.loads(response.content)
        except (LLMProviderError, json.JSONDecodeError):
            data = {}

        return ParsedLog(
            error_class=data.get("error_class"),
            message=(data.get("message") or cleaned[:200]).strip(),
            stack_summary=data.get("stack_summary") or self._summarize_stack(cleaned),
            file=data.get("file"),
            line=data.get("line") if isinstance(data.get("line"), int) else None,
            timestamp=ts,
            raw_excerpt=cleaned[:4000],
            extraction_method=ExtractionMethod.LLM,
        )
