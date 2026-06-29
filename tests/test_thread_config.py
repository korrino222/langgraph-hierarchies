"""Unit tests for LangSmith thread config normalization."""

from __future__ import annotations

from uuid import UUID

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.tracing.config import (
    build_invoke_config,
    normalize_thread_config,
    resolve_thread_id,
)


def test_resolve_from_metadata_thread_id() -> None:
    config: RunnableConfig = {"metadata": {"thread_id": "meta-thread"}}

    assert resolve_thread_id(config) == "meta-thread"


def test_resolve_from_metadata_session_id() -> None:
    config: RunnableConfig = {"metadata": {"session_id": "session-thread"}}

    assert resolve_thread_id(config) == "session-thread"


def test_resolve_from_configurable() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "cfg-thread"}}

    assert resolve_thread_id(config) == "cfg-thread"


def test_resolve_from_context() -> None:
    context = BaseContext(thread_id="ctx-thread")

    assert resolve_thread_id(None, context) == "ctx-thread"


def test_precedence_metadata_over_configurable() -> None:
    config: RunnableConfig = {
        "metadata": {"thread_id": "meta-wins"},
        "configurable": {"thread_id": "cfg-loses"},
    }

    assert resolve_thread_id(config) == "meta-wins"


def test_empty_thread_id_ignored() -> None:
    config: RunnableConfig = {
        "metadata": {"thread_id": ""},
        "configurable": {"thread_id": "   "},
    }
    context = BaseContext(thread_id="")

    assert resolve_thread_id(config, context) is None


def test_uuid_coercion() -> None:
    thread_uuid = UUID("12345678-1234-5678-1234-567812345678")
    config: RunnableConfig = {"configurable": {"thread_id": thread_uuid}}

    assert resolve_thread_id(config) == "12345678-1234-5678-1234-567812345678"


def test_normalize_backfills_all_slots() -> None:
    config: RunnableConfig = {"configurable": {"thread_id": "only-configurable"}}

    normalized = normalize_thread_config(config)

    assert normalized["metadata"]["thread_id"] == "only-configurable"
    assert normalized["metadata"]["session_id"] == "only-configurable"
    assert normalized["configurable"]["thread_id"] == "only-configurable"


def test_normalize_does_not_overwrite_explicit_metadata() -> None:
    config: RunnableConfig = {
        "metadata": {"thread_id": "explicit-meta"},
        "configurable": {"thread_id": "cfg-thread"},
    }

    normalized = normalize_thread_config(config)

    assert normalized["metadata"]["thread_id"] == "explicit-meta"
    assert normalized["metadata"]["session_id"] == "explicit-meta"
    assert normalized["configurable"]["thread_id"] == "cfg-thread"


def test_conflicting_thread_and_session_metadata() -> None:
    config: RunnableConfig = {
        "metadata": {
            "thread_id": "thread-a",
            "session_id": "session-b",
        }
    }

    normalized = normalize_thread_config(config)

    assert normalized["metadata"]["thread_id"] == "thread-a"
    assert normalized["metadata"]["session_id"] == "session-b"


def test_normalize_noop_when_unset() -> None:
    config: RunnableConfig = {"recursion_limit": 25}

    normalized = normalize_thread_config(config)

    assert normalized == {"recursion_limit": 25}


def test_build_invoke_config_populates_all_slots() -> None:
    config = build_invoke_config(
        thread_id="cli-thread",
        run_name="demo",
        tags=["example"],
        recursion_limit=100,
    )

    assert config["run_name"] == "demo"
    assert config["tags"] == ["example"]
    assert config["recursion_limit"] == 100
    assert config["metadata"]["thread_id"] == "cli-thread"
    assert config["metadata"]["session_id"] == "cli-thread"
    assert config["configurable"]["thread_id"] == "cli-thread"


@pytest.mark.parametrize(
    ("value",),
    [
        (None,),
        ("",),
        ("   ",),
    ],
)
def test_resolve_thread_id_ignores_blank_values(value: str | None) -> None:
    config: RunnableConfig = {"metadata": {"thread_id": value}}

    assert resolve_thread_id(config) is None
