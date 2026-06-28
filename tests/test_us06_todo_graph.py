"""US-06 TodoGraph gating tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.graphs.todo import TodoGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults
from langgraph_hierarchies.tools.builtins import todo_complete, todo_write

pytestmark = pytest.mark.us06


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted-model"

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


class TodoTestGraph(TodoGraph):
    name = "todo_test"
    description = "TodoGraph test agent"


def _report_call(report: str = "done") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_to_supervisor",
                "args": {"report": report},
                "id": "report-1",
                "type": "tool_call",
            }
        ],
    )


def test_report_blocked_routes_to_todo_incomplete() -> None:
    graph = TodoTestGraph(state_schema=BaseState, context_schema=BaseContext)
    graph.compile_graph()

    state = create_base_state_defaults()
    state["todo_list"] = {"task-a": False}
    tool_call = {
        "name": "report_to_supervisor",
        "args": {"report": "done"},
        "id": "report-1",
        "type": "tool_call",
    }
    reasoning_result = AIMessage(content="", tool_calls=[tool_call])

    route = graph._handle_tool_call(state, reasoning_result, tool_call)
    assert route == "todo_incomplete"


def test_todo_incomplete_appends_fail_message() -> None:
    graph = TodoTestGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        todo_check_fail_message="Custom gate failure.",
    )
    graph.compile_graph()

    tool_call = {
        "name": "report_to_supervisor",
        "args": {"report": "done"},
        "id": "report-1",
        "type": "tool_call",
    }
    state = create_base_state_defaults()
    state["messages"] = [AIMessage(content="", tool_calls=[tool_call])]

    update = graph.todo_incomplete(state)
    assert "Custom gate failure." in update["messages"][0].content


def test_report_allowed_when_all_todos_complete() -> None:
    graph = TodoTestGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()
    context = BaseContext(model=ScriptedModel(responses=[_report_call("finished")]))

    state = create_base_state_defaults()
    state["todo_list"] = {"task-a": True, "task-b": True}

    result = compiled.invoke(
        state,
        config=RunnableConfig(recursion_limit=25),
        context=context,
    )

    assert result["current_agent_report"] == "finished"


def test_custom_todo_check_honored() -> None:
    graph = TodoTestGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        todo_check=lambda state: state.get("current_agent_report") == "ready",
    )
    graph.compile_graph()

    blocked_state = create_base_state_defaults()
    blocked_call = {
        "name": "report_to_supervisor",
        "args": {"report": "done"},
        "id": "report-1",
        "type": "tool_call",
    }
    blocked_result = AIMessage(content="", tool_calls=[blocked_call])
    assert (
        graph._handle_tool_call(blocked_state, blocked_result, blocked_call)
        == "todo_incomplete"
    )

    allowed_state = create_base_state_defaults()
    allowed_state["current_agent_report"] = "ready"
    assert (
        graph._handle_tool_call(allowed_state, blocked_result, blocked_call)
        != "todo_incomplete"
    )


def test_todo_toolkit_attached() -> None:
    graph = TodoTestGraph(state_schema=BaseState, context_schema=BaseContext)
    graph.compile_graph()

    tool_names = {tool.name for tool in graph.tools}
    assert todo_write.name in tool_names
    assert todo_complete.name in tool_names
