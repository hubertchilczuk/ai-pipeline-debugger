"""Log line parser."""
import re
from dataclasses import dataclass


@dataclass
class ParsedRecord:
    timestamp: str
    level: str
    message: str


class LogParser:
    """Parses raw log lines into structured records."""

    def parse(self, line: str) -> ParsedRecord | None:
        raise NotImplementedError
