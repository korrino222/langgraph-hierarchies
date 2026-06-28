"""US-04 compile_as_root and unified CompiledGraph invocation tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph._internal._runnable import RunnableSeq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel._utils import find_subgraph_pregel
from langgraph.types import Command

from langgraph_hierarchies.graphs.compiled import CONFIG_KEY_RUNTIME, CompiledGraph
from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults

pytestmark = pytest.mark.us04


class ScriptModel(BaseChatModel):
    """Deterministic model returning scripted tool calls in order."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "script-model"

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


class ReportReactGraph(ReactGraph):
    name = "report_agent"
    description = "Reports to supervisor in one step"


class WorkerGraph(ReactGraph):
    name = "worker"
    description = "Worker subagent"


class ParentGraph(ReactGraph):
    name = "parent"
    description = "Parent orchestrator"

    def compile_graph(self, *args, **kwargs):
        worker = WorkerGraph(
            state_schema=BaseState,
            context_schema=BaseContext,
        ).compile_graph()
        return super().compile_graph(*args, compiled_subgraphs=[worker], **kwargs)


class HookTrackingGraph(SimpleGraph):
    name = "hook_tracker"
    description = "Tracks entry hook execution"

    entry_hook_calls = 0

    def entry_hook(self, graph, state, config=None, **kwargs):
        type(self).entry_hook_calls += 1
        state = dict(state)
        state["entry_hook_ran"] = True
        return state

    async def aentry_hook(self, graph, state, config=None, **kwargs):
        return self.entry_hook(graph, state, config, **kwargs)


def _report_message(report: str = "done") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_to_supervisor",
                "args": {"report": report},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def test_compile_as_root_stores_state_defaults() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_as_root(state_defaults={"max_iterations": 99})

    assert compiled._root_state_defaults == {"max_iterations": 99}


def test_compile_as_root_enables_interrupts() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    graph.compile_as_root()

    assert graph._enable_interrupts is True


def test_root_defaults_merge_without_overwriting_provided_values() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_as_root(
        state_defaults={
            "max_iterations": 99,
            "current_agent_report": "default report",
        }
    )

    state = create_base_state_defaults()
    state["max_iterations"] = 12
    state.pop("current_agent_report")

    merged = compiled.after_entry_hook(state)

    assert merged["max_iterations"] == 12
    assert merged["current_agent_report"] == "default report"


def test_invoke_injects_runtime_into_config() -> None:
    context = BaseContext(model=ScriptModel(responses=[_report_message()]))
    config: RunnableConfig = {"configurable": {"thread_id": "us04-context"}}

    injected = CompiledGraph._inject_context(config, context)

    assert CONFIG_KEY_RUNTIME in injected["configurable"]
    assert injected["configurable"][CONFIG_KEY_RUNTIME].context is context


def test_invoke_reaches_reasoning_via_injected_context() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()
    context = BaseContext(model=ScriptModel(responses=[_report_message("from model")]))

    result = compiled.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=25),
        context=context,
    )

    assert result["current_agent_report"] == "from model"


def test_nested_hierarchy_propagates_context_to_child() -> None:
    worker_report = _report_message("child done")
    delegate = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "worker",
                "args": {
                    "task": "do work",
                    "task_scope": "worker only",
                    "task_iterations": 0,
                },
                "id": "parent-call-1",
                "type": "tool_call",
            }
        ],
    )
    finish = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "finish_task",
                "args": {"result": "parent done"},
                "id": "parent-call-2",
                "type": "tool_call",
            }
        ],
    )
    context = BaseContext(
        model=ScriptModel(responses=[delegate, worker_report, finish]),
    )

    parent = ParentGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
    )
    root = parent.compile_as_root()

    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )

    assert result["current_agent_report"] == "parent done"


def test_stream_runs_entry_hooks() -> None:
    HookTrackingGraph.entry_hook_calls = 0
    graph = HookTrackingGraph(state_schema=BaseState)
    compiled = graph.compile_as_root(state_defaults={"current_agent_report": "seeded"})
    state = create_base_state_defaults()
    state["current_agent_report"] = None

    chunks = list(
        compiled.stream(
            state,
            config=RunnableConfig(recursion_limit=10),
            stream_mode="values",
        )
    )

    assert HookTrackingGraph.entry_hook_calls == 1
    assert chunks
    assert chunks[0].get("current_agent_report") == "seeded"


def test_astream_runs_entry_hooks() -> None:
    HookTrackingGraph.entry_hook_calls = 0
    graph = HookTrackingGraph(state_schema=BaseState)
    compiled = graph.compile_as_root(state_defaults={"current_agent_report": "seeded"})
    state = create_base_state_defaults()
    state["current_agent_report"] = None

    async def _collect() -> list:
        chunks = []
        async for chunk in compiled.astream(
            state,
            config=RunnableConfig(recursion_limit=10),
            stream_mode="values",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert HookTrackingGraph.entry_hook_calls == 1
    assert chunks
    assert chunks[0].get("current_agent_report") == "seeded"


def test_command_invoke_bypasses_entry_hooks() -> None:
    graph = HookTrackingGraph(state_schema=BaseState)
    compiled = graph.compile_as_root()

    with patch.object(compiled._compiled, "invoke", return_value={}) as inner_invoke:
        with patch.object(compiled, "entry_hook") as entry_hook:
            compiled.invoke(
                Command(resume={"messages": []}),
                config=RunnableConfig(recursion_limit=10),
            )
            entry_hook.assert_not_called()
            inner_invoke.assert_called_once()


def test_compiled_graph_is_runnable_seq_subclass() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()

    assert isinstance(compiled, RunnableSeq)


def test_find_subgraph_pregel_discovers_inner_pregel() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()

    discovered = find_subgraph_pregel(compiled)

    assert discovered is compiled._compiled


def test_get_state_with_subgraphs_and_checkpointer() -> None:
    checkpointer = MemorySaver()
    parent = ParentGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
    )
    root = parent.compile_as_root(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": "us04-subgraphs"}}

    worker_report = _report_message("child done")
    delegate = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "worker",
                "args": {
                    "task": "do work",
                    "task_scope": "worker only",
                    "task_iterations": 0,
                },
                "id": "parent-call-1",
                "type": "tool_call",
            }
        ],
    )
    finish = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "finish_task",
                "args": {"result": "parent done"},
                "id": "parent-call-2",
                "type": "tool_call",
            }
        ],
    )
    context = BaseContext(
        model=ScriptModel(responses=[delegate, worker_report, finish]),
    )

    root.invoke(
        create_base_state_defaults(),
        config=config,
        context=context,
    )

    snapshot = root.get_state(config, subgraphs=True)

    assert snapshot is not None
    assert snapshot.values["current_agent_report"] == "parent done"
