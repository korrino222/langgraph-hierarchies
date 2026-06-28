"""US-02 SubchainPolicy entry/exit isolation tests."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_hierarchies.graphs.compiled import SubchainPolicy
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults
from langgraph_hierarchies.types import Progress

pytestmark = pytest.mark.us02


class PolicyTestGraph(SimpleGraph):
    name = "policy_test"
    description = "Subchain policy test graph"


def _compile_with_policy(policy: SubchainPolicy | None = None):
    graph = PolicyTestGraph(
        state_schema=BaseState,
        subchain_policy=policy,
    )
    return graph.compile_graph()


def test_entry_pushes_snapshot_onto_subchain_stack() -> None:
    compiled = _compile_with_policy(SubchainPolicy())
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="parent msg")]

    compiled.entry_hook(state)

    stack = state["__subchain_stack__"]
    assert len(stack) == 1
    assert stack[0]["agent_name"] == "policy_test"
    assert stack[0]["saved_state"]["messages"][0].content == "parent msg"
    assert stack[0]["saved_state"]["messages"] is not state["messages"]


def test_entry_clears_messages_todo_lists_and_iteration() -> None:
    compiled = _compile_with_policy(SubchainPolicy())
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="parent msg")]
    state["todo_lists"] = {"planner": {"task": False}}
    state["iteration_number"] = 7

    result = compiled.entry_hook(state)

    assert result["messages"] == []
    assert result["todo_lists"] == {}
    assert result["iteration_number"] == 0


def test_entry_respects_clear_messages_false() -> None:
    compiled = _compile_with_policy(SubchainPolicy(clear_messages=False))
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="keep me")]

    result = compiled.entry_hook(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "keep me"


def test_entry_respects_reset_iteration_false() -> None:
    compiled = _compile_with_policy(SubchainPolicy(reset_iteration=False))
    state = create_base_state_defaults()
    state["iteration_number"] = 12

    result = compiled.entry_hook(state)

    assert result["iteration_number"] == 12


def test_entry_clears_operator_chat_when_disabled() -> None:
    compiled = _compile_with_policy(
        SubchainPolicy(preserve_chat_with_operator=False),
    )
    state = create_base_state_defaults()
    state["chat_with_operator"] = [HumanMessage(content="operator msg")]

    result = compiled.entry_hook(state)

    assert result["chat_with_operator"] == []


def test_entry_preserves_operator_chat_by_default() -> None:
    compiled = _compile_with_policy(SubchainPolicy())
    state = create_base_state_defaults()
    state["chat_with_operator"] = [HumanMessage(content="operator msg")]

    result = compiled.entry_hook(state)

    assert result["chat_with_operator"][0].content == "operator msg"


def test_entry_overrides_max_iterations() -> None:
    compiled = _compile_with_policy(SubchainPolicy(max_iterations=10))
    state = create_base_state_defaults()
    state["max_iterations"] = 100

    result = compiled.entry_hook(state)

    assert result["max_iterations"] == 10


def test_exit_restores_parent_and_merges_fields() -> None:
    compiled = _compile_with_policy(
        SubchainPolicy(merge_fields=["todo_list"]),
    )
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="parent msg")]
    state["todo_list"] = {"parent": True}

    compiled.entry_hook(state)

    state["messages"] = [AIMessage(content="subagent internal")]
    state["todo_list"] = {"child": True}
    state["current_agent_report"] = "subagent report"

    result = compiled.exit_hook(state)

    assert result["messages"][0].content == "parent msg"
    assert result["todo_list"] == {"child": True}
    assert result["current_agent_report"] == "subagent report"


def test_exit_propagates_progress_and_report_as_infrastructure() -> None:
    compiled = _compile_with_policy(SubchainPolicy(merge_fields=[]))
    state = create_base_state_defaults()
    state["current_agent_report"] = "parent report"
    state["progress"] = {"overall": Progress(scheduled_executions=1)}

    compiled.entry_hook(state)

    state["current_agent_report"] = "subagent report"
    state["progress"] = {
        "overall": Progress(scheduled_executions=2, finished_executions=1)
    }

    result = compiled.exit_hook(state)

    assert result["current_agent_report"] == "subagent report"
    assert result["progress"]["overall"].scheduled_executions == 2
    assert result["progress"]["overall"].finished_executions == 1


def test_exit_drops_discard_fields() -> None:
    compiled = _compile_with_policy(
        SubchainPolicy(discard_fields=["current_agent_report"])
    )
    state = create_base_state_defaults()
    state["current_agent_report"] = "should be discarded"

    compiled.entry_hook(state)
    result = compiled.exit_hook(state)

    assert "current_agent_report" not in result


def test_round_trip_restores_parent_state_and_clears_stack() -> None:
    compiled = _compile_with_policy(SubchainPolicy(merge_fields=["todo_list"]))
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="parent msg")]
    state["todo_list"] = {"parent": False}

    compiled.entry_hook(state)
    state["messages"] = [AIMessage(content="subagent internal")]
    state["todo_list"] = {"child": True}

    result = compiled.exit_hook(state)

    assert result["messages"][0].content == "parent msg"
    assert result["todo_list"] == {"child": True}
    assert result["__subchain_stack__"] == []


def test_no_policy_passthrough() -> None:
    compiled = _compile_with_policy(None)
    state = create_base_state_defaults()
    state["messages"] = [HumanMessage(content="stays")]
    state["iteration_number"] = 5
    state["todo_lists"] = {"planner": {"task": False}}

    result = compiled.entry_hook(state)

    assert len(result["messages"]) == 1
    assert result["iteration_number"] == 5
    assert result["todo_lists"] == {"planner": {"task": False}}
    assert result.get("__subchain_stack__", []) == []
