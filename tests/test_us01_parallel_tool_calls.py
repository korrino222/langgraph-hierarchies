"""US-01 ReactGraph parallel tool-call handling tests."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults

pytestmark = pytest.mark.us01


@tool
def alpha_work(value: str = "a") -> str:
    """Return alpha-prefixed work output."""
    return f"alpha:{value}"


@tool
def beta_work(value: str = "b") -> str:
    """Return beta-prefixed work output."""
    return f"beta:{value}"


class ScriptedModel(BaseChatModel):
    """Deterministic model returning scripted AIMessages in order."""

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


class ParallelReactGraph(ReactGraph):
    name = "parallel_agent"
    description = "Agent for parallel tool-call tests"


class WorkerGraph(ReactGraph):
    name = "worker"
    description = "Worker subagent for parallel tool-call tests"


class OrchestratorGraph(ReactGraph):
    name = "orchestrator"
    description = "Parent orchestrator for parallel subagent tests"

    def compile_graph(self, *args, **kwargs):
        worker = WorkerGraph(
            state_schema=BaseState,
            context_schema=BaseContext,
        ).compile_graph()
        return super().compile_graph(*args, compiled_subgraphs=[worker], **kwargs)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


def _report_message(report: str = "done") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[_tool_call("report_to_supervisor", {"report": report}, "report-1")],
    )


def _worker_call(call_id: str, task: str = "do work") -> dict[str, Any]:
    return _tool_call(
        "worker",
        {"task": task, "task_scope": "worker only", "task_iterations": 0},
        call_id,
    )


def test_determine_action_dispatches_parallel_flat_tools() -> None:
    graph = ParallelReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        tools=[alpha_work, beta_work],
    )
    graph.compile_graph()

    state = {
        **create_base_state_defaults(),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("alpha_work", {"value": "a"}, "call-1"),
                    _tool_call("beta_work", {"value": "b"}, "call-2"),
                ],
            )
        ],
    }

    result = graph.determine_action(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(send.node == "tool" for send in result)

    call_ids = []
    for send in result:
        isolated = send.arg["messages"][-1]
        assert len(isolated.tool_calls) == 1
        call_ids.append(isolated.tool_calls[0]["id"])
    assert call_ids == ["call-1", "call-2"]


def test_determine_action_blocks_parallel_subagent_calls() -> None:
    graph = OrchestratorGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        tools=[alpha_work],
    )
    compiled = graph.compile_graph()

    state = {
        **create_base_state_defaults(),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _worker_call("call-sub-1"),
                    _tool_call("alpha_work", {"value": "x"}, "call-flat-1"),
                ],
            )
        ],
    }
    graph.compiled_subgraphs = compiled.compiled_subgraphs

    assert graph.determine_action(state) == "invalid_parallel_subgraph_call"


def test_invalid_parallel_subgraph_call_emits_message_per_tool_call() -> None:
    graph = ParallelReactGraph(state_schema=BaseState, context_schema=BaseContext)
    state = {
        **create_base_state_defaults(),
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    _worker_call("call-1"),
                    _worker_call("call-2"),
                    _worker_call("call-3"),
                ],
            )
        ],
    }

    update = graph.invalid_parallel_subgraph_call(state)
    messages = update["messages"]
    assert len(messages) == 3
    assert {message.tool_call_id for message in messages} == {
        "call-1",
        "call-2",
        "call-3",
    }
    assert all("sub-agent" in message.content for message in messages)


def test_e2e_parallel_flat_tools_execute_all_calls() -> None:
    parallel_flat = AIMessage(
        content="",
        tool_calls=[
            _tool_call("alpha_work", {"value": "a"}, "call-1"),
            _tool_call("beta_work", {"value": "b"}, "call-2"),
        ],
    )
    graph = ParallelReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        tools=[alpha_work, beta_work],
    )
    compiled = graph.compile_graph()
    context = BaseContext(
        model=ScriptedModel(responses=[parallel_flat, _report_message("done")]),
    )

    result = compiled.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=25),
        context=context,
    )

    assert result["current_agent_report"] == "done"
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert {message.tool_call_id for message in tool_messages} == {"call-1", "call-2"}
    assert any("alpha:a" in message.content for message in tool_messages)
    assert any("beta:b" in message.content for message in tool_messages)


def test_e2e_parallel_subagent_blocked_then_recovers() -> None:
    parallel_blocked = AIMessage(
        content="",
        tool_calls=[_worker_call("call-sub-1"), _worker_call("call-sub-2")],
    )
    single_delegate = AIMessage(
        content="",
        tool_calls=[_worker_call("call-sub-3")],
    )
    worker_report = _report_message("child done")
    parent_finish = AIMessage(
        content="",
        tool_calls=[_tool_call("finish_task", {"result": "parent done"}, "finish-1")],
    )

    orchestrator = OrchestratorGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
    )
    root = orchestrator.compile_as_root()
    context = BaseContext(
        model=ScriptedModel(
            responses=[
                parallel_blocked,
                single_delegate,
                worker_report,
                parent_finish,
            ]
        ),
    )

    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )

    assert result["current_agent_report"] == "parent done"
    assert result["is_finished"] is True

    blocked_tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id in {"call-sub-1", "call-sub-2"}
    ]
    assert len(blocked_tool_messages) == 2
    assert all("sub-agent" in message.content for message in blocked_tool_messages)
