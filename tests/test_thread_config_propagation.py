"""Integration tests for thread config propagation through CompiledGraph."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

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


class RootGraph(SimpleGraph):
    name = "root"
    description = "Root graph for config propagation tests"

    def computation_node(self, state: dict) -> dict:
        return {"current_agent_report": "ok"}


class ChildGraph(SimpleGraph):
    name = "child_capture"
    description = "Child graph for config propagation tests"

    def computation_node(self, state: dict) -> dict:
        return {"current_agent_report": "child ok"}


class ParentGraph(ReactGraph):
    name = "parent_capture"
    description = "Parent graph dispatching to child"

    reports_to_supervisor = False

    def compile_graph(self, *args, **kwargs):
        child = ChildGraph(
            state_schema=BaseState,
            context_schema=BaseContext,
        ).compile_graph()
        return super().compile_graph(*args, compiled_subgraphs=[child], **kwargs)


def _delegate_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "child_capture",
                "args": {
                    "task": "do work",
                    "task_scope": "child only",
                    "task_iterations": 0,
                },
                "id": "parent-call-1",
                "type": "tool_call",
            }
        ],
    )


def _finish_message() -> AIMessage:
    return AIMessage(
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


def test_compiled_invoke_normalizes_configurable_only() -> None:
    graph = RootGraph(state_schema=BaseState)
    compiled = graph.compile_as_root()

    with patch.object(compiled._compiled, "invoke", return_value={}) as inner_invoke:
        compiled.invoke(
            create_base_state_defaults(),
            config={"configurable": {"thread_id": "t-configurable"}},
        )
        config_passed = inner_invoke.call_args.args[1]

    assert config_passed["metadata"]["thread_id"] == "t-configurable"
    assert config_passed["metadata"]["session_id"] == "t-configurable"
    assert config_passed["configurable"]["thread_id"] == "t-configurable"


def test_compiled_invoke_normalizes_context_only() -> None:
    graph = RootGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_as_root()
    context = BaseContext(thread_id="t-context")

    with patch.object(compiled._compiled, "invoke", return_value={}) as inner_invoke:
        compiled.invoke(
            create_base_state_defaults(),
            config=RunnableConfig(recursion_limit=10),
            context=context,
        )
        config_passed = inner_invoke.call_args.args[1]

    assert config_passed["metadata"]["thread_id"] == "t-context"
    assert config_passed["metadata"]["session_id"] == "t-context"
    assert config_passed["configurable"]["thread_id"] == "t-context"


def test_thread_id_reaches_child_node_config() -> None:
    parent = ParentGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
    )
    root = parent.compile_as_root()
    child = root.compiled_subgraphs[0]
    captured_configs: list[RunnableConfig] = []
    original_invoke = child._compiled.invoke

    def capturing_invoke(input, config=None, **kwargs):
        captured_configs.append(config)
        return original_invoke(input, config, **kwargs)

    child._compiled.invoke = capturing_invoke
    context = BaseContext(
        model=ScriptModel(responses=[_delegate_message(), _finish_message()]),
    )

    root.invoke(
        create_base_state_defaults(),
        config={"configurable": {"thread_id": "t-child"}},
        context=context,
    )

    assert captured_configs
    child_config = captured_configs[0]
    assert child_config is not None
    assert child_config["metadata"]["thread_id"] == "t-child"
    assert child_config["metadata"]["session_id"] == "t-child"
    assert child_config["configurable"]["thread_id"] == "t-child"


def test_command_invoke_still_normalized() -> None:
    graph = RootGraph(state_schema=BaseState)
    compiled = graph.compile_as_root()

    with patch.object(compiled._compiled, "invoke", return_value={}) as inner_invoke:
        with patch.object(compiled, "entry_hook") as entry_hook:
            compiled.invoke(
                Command(resume={"messages": []}),
                config={"configurable": {"thread_id": "t-command"}},
            )
            entry_hook.assert_not_called()
            inner_invoke.assert_called_once()
            config_passed = inner_invoke.call_args.args[1]

    assert config_passed["metadata"]["thread_id"] == "t-command"
    assert config_passed["metadata"]["session_id"] == "t-command"
    assert config_passed["configurable"]["thread_id"] == "t-command"


def test_stream_entrypoint_normalized() -> None:
    graph = RootGraph(state_schema=BaseState)
    compiled = graph.compile_as_root()

    with patch.object(
        compiled._compiled, "stream", return_value=iter([{}])
    ) as inner_stream:
        list(
            compiled.stream(
                create_base_state_defaults(),
                config={"configurable": {"thread_id": "t-stream"}},
                stream_mode="values",
            )
        )
        config_passed = inner_stream.call_args.args[1]

    assert config_passed["metadata"]["thread_id"] == "t-stream"
    assert config_passed["metadata"]["session_id"] == "t-stream"
    assert config_passed["configurable"]["thread_id"] == "t-stream"


def test_prepare_config_returns_dict_when_context_none() -> None:
    graph = RootGraph(state_schema=BaseState)
    compiled = graph.compile_as_root()

    prepared = compiled._prepare_config(None, None)

    assert isinstance(prepared, dict)
