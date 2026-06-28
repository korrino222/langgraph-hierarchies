"""Compiled graph wrapper and subchain policy."""

from __future__ import annotations

import random
import string
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel

if TYPE_CHECKING:
    from langgraph_hierarchies.graphs.base import BaseGraph


@dataclass
class SubchainPolicy:
    """Declarative subchain entry/exit policy (semantics applied in US-02)."""

    clear_messages: bool = True
    preserve_chat_with_operator: bool = True
    reset_iteration: bool = True
    max_iterations: int | None = None
    merge_fields: list[str] = field(default_factory=list)
    discard_fields: list[str] = field(default_factory=list)


class CompiledGraph(Runnable):
    """Runnable wrapper around a compiled LangGraph with hook scaffolding."""

    name: str
    node_label: str
    description: str
    args_schema: type[BaseModel]
    graph: BaseGraph
    _compiled: CompiledStateGraph
    compiled_subgraphs: list[CompiledGraph]
    as_tool: bool
    subchain_policy: SubchainPolicy | None
    _root_state_defaults: dict | None

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        graph: BaseGraph,
        compiled: CompiledStateGraph,
        compiled_subgraphs: list[CompiledGraph] | None = None,
        as_tool: bool = True,
        subchain_policy: SubchainPolicy | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.graph = graph
        self._compiled = compiled
        self._compiled.name = name
        self.compiled_subgraphs = compiled_subgraphs or []
        self.as_tool = as_tool
        self.subchain_policy = subchain_policy
        self._root_state_defaults = None

        suffix = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        self.node_label = f"{self.name}_{suffix}"

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(input, Command):
            return self._compiled.invoke(input, config, context=context, **kwargs)

        state = self.entry_hook(input, config)
        state = self.after_entry_hook(state, config)
        state = self._compiled.invoke(state, config, context=context, **kwargs)
        state = self.exit_hook(state, config)
        return self.after_exit_hook(state, config)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(input, Command):
            return await self._compiled.ainvoke(input, config, context=context, **kwargs)

        state = await self.aentry_hook(input, config)
        state = self.after_entry_hook(state, config)
        state = await self._compiled.ainvoke(state, config, context=context, **kwargs)
        state = await self.aexit_hook(state, config)
        return self.after_exit_hook(state, config)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._compiled.stream(input, config, context=context, **kwargs)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for chunk in self._compiled.astream(input, config, context=context, **kwargs):
            yield chunk

    def after_entry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self._root_state_defaults:
            for key, default_value in self._root_state_defaults.items():
                if key not in state or state[key] is None:
                    state[key] = default_value
        state["iteration_number"] = 0
        return state

    def entry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return self.graph.entry_hook(self, state, config)

    async def aentry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return await self.graph.aentry_hook(self, state, config)

    def exit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return self.graph.exit_hook(self, state, config)

    async def aexit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return await self.graph.aexit_hook(self, state, config)

    def after_exit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self.as_tool and state.get("current_tool_call"):
            tool_call = state["current_tool_call"]
            report = state.get("current_agent_report", "")
            tool_message = ToolMessage(
                content=report,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                id=str(uuid4()),
            )
            state["messages"] = state.get("messages", []) + [tool_message]
        return state
