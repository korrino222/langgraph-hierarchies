"""US-01 state reducer and defaults tests."""

from dataclasses import dataclass

import pytest
from langchain_core.messages import HumanMessage

from langgraph_hierarchies.state.reducers import (
    reduce_messages,
    reduce_todo_lists,
    reducer_upsert,
)
from langgraph_hierarchies.state.schema import create_base_state_defaults
from langgraph_hierarchies.types import Progress


def test_reduce_messages_replaces_by_id() -> None:
    first = HumanMessage(content="hello", id="msg-1")
    second = HumanMessage(content="updated", id="msg-1")
    merged = reduce_messages([first], [second])
    assert len(merged) == 1
    assert merged[0].content == "updated"


def test_reduce_messages_appends_new_ids() -> None:
    first = HumanMessage(content="one", id="msg-1")
    second = HumanMessage(content="two", id="msg-2")
    merged = reduce_messages([first], [second])
    assert len(merged) == 2


def test_reduce_todo_lists_true_priority() -> None:
    left = {"plan": {"item-a": True, "item-b": False}}
    right = {"plan": {"item-b": False}}
    merged = reduce_todo_lists(left, right)
    assert merged["plan"]["item-a"] is True
    assert merged["plan"]["item-b"] is False


def test_reducer_upsert_deduplicates_by_id() -> None:
    @dataclass
    class Item:
        id: str
        value: str

    existing = [Item("a", "old"), Item("b", "keep")]
    new = [Item("a", "new"), Item("c", "added")]
    result = reducer_upsert(existing, new)
    assert [item.value for item in result] == ["new", "keep", "added"]


def test_reducer_upsert_raises_without_id() -> None:
    with pytest.raises(ValueError, match="reducer_upsert requires"):
        reducer_upsert([], [object()])


def test_create_base_state_defaults_keys() -> None:
    defaults = create_base_state_defaults()
    assert defaults["messages"] == []
    assert defaults["max_iterations"] == 40
    assert defaults["is_finished"] is False


def test_reduce_progress_keeps_higher_counts() -> None:
    from langgraph_hierarchies.state.reducers import reduce_progress

    left = {"agent": Progress(scheduled_executions=1, finished_executions=0)}
    right = {"agent": Progress(scheduled_executions=2, finished_executions=1)}
    result = reduce_progress(left, right)
    assert result["agent"].scheduled_executions == 2
    assert result["agent"].finished_executions == 1
