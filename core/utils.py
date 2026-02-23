"""Shared utilities."""
from __future__ import annotations

import re
from typing import Any

_NUMBERED_PREFIX = re.compile(r"^\s*\d+\.")


def to_str(val: Any) -> str:
    """Coerce LLM output (str, list, None) to a single human-readable string.

    When a list arrives, render as a numbered list — but skip renumbering if items
    are already numbered (otherwise we get "1. 1. ...").
    """
    if val is None:
        return ""
    if isinstance(val, list):
        items = [str(item) for item in val if item is not None and str(item).strip()]
        if not items:
            return ""
        if all(_NUMBERED_PREFIX.match(item) for item in items):
            return "\n".join(items)
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    return str(val) if val else ""
