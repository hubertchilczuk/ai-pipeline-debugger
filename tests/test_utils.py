"""Unit tests for core.utils.to_str."""
from __future__ import annotations

from core.utils import to_str


def test_to_str_passthrough() -> None:
    assert to_str("hello") == "hello"


def test_to_str_none_and_empty() -> None:
    assert to_str(None) == ""
    assert to_str("") == ""
    assert to_str([]) == ""


def test_to_str_numbers_unnumbered_list() -> None:
    assert to_str(["alpha", "beta"]) == "1. alpha\n2. beta"


def test_to_str_skips_renumbering_when_already_numbered() -> None:
    assert to_str(["1. alpha", "2. beta"]) == "1. alpha\n2. beta"


def test_to_str_filters_empty_items() -> None:
    assert to_str(["alpha", "", None, "beta"]) == "1. alpha\n2. beta"
