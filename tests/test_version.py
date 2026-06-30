"""Smoke tests for package metadata."""

import langgraph_hierarchies


def test_version() -> None:
    assert langgraph_hierarchies.__version__ == "0.0.8"
