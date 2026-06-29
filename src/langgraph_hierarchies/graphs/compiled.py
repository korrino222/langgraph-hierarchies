"""Compiled graph wrapper and subchain policy."""

from __future__ import annotations

import copy
import random
import string
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph._internal._runnable import RunnableSeq
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

try:
    from langgraph._internal._constants import CONFIG_KEY_RUNTIME
except ImportError:  # pragma: no cover - pin fallback
    CONFIG_KEY_RUNTIME = "__pregel_runtime"

if TYPE_CHECKING:
    from langgraph_hierarchies.graphs.base import BaseGraph

from langgraph_hierarchies.tracing.config import (
    ensure_runnable_config_dict,
    normalize_thread_config,
)


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

    @property
    def steps(self) -> list:
        """Expose inner Pregel for LangGraph subgraph discovery."""
        return [self._compiled]

    @staticmethod
    def _inject_context(
        config: RunnableConfig | None,
        context: Any,
    ) -> RunnableConfig:
        config = dict(config or {})
        configurable = dict(config.get("configurable", {}))
        runtime = Runtime(context=context)
        parent_runtime = configurable.get(CONFIG_KEY_RUNTIME)
        if parent_runtime is not None:
            runtime = parent_runtime.merge(runtime)
        configurable[CONFIG_KEY_RUNTIME] = runtime
        config["configurable"] = configurable
        return config

    def _prepare_config(
        self,
        config: RunnableConfig | None,
        context: Any,
    ) -> RunnableConfig:
        config = ensure_runnable_config_dict(config)
        config = normalize_thread_config(config, context)
        if context is not None:
            config = self._inject_context(config, context)
        return config

    def _final_stream_state(
        self,
        config: RunnableConfig | None,
        fallback: dict,
        last_chunk: Any,
    ) -> dict:
        if isinstance(last_chunk, dict):
            return last_chunk
        if self._compiled.checkpointer is not None:
            snapshot = self._compiled.get_state(config)
            if snapshot is not None:
                return snapshot.values
        return fallback

    async def _afinal_stream_state(
        self,
        config: RunnableConfig | None,
        fallback: dict,
        last_chunk: Any,
    ) -> dict:
        if isinstance(last_chunk, dict):
            return last_chunk
        if self._compiled.checkpointer is not None:
            snapshot = await self._compiled.aget_state(config)
            if snapshot is not None:
                return snapshot.values
        return fallback

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> Any:
        config = self._prepare_config(config, context)
        if isinstance(input, Command):
            return self._compiled.invoke(input, config, **kwargs)

        state = self.entry_hook(input, config)
        state = self.after_entry_hook(state, config)
        state = self._compiled.invoke(state, config, **kwargs)
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
        config = self._prepare_config(config, context)
        if isinstance(input, Command):
            return await self._compiled.ainvoke(input, config, **kwargs)

        state = await self.aentry_hook(input, config)
        state = self.after_entry_hook(state, config)
        state = await self._compiled.ainvoke(state, config, **kwargs)
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
        config = self._prepare_config(config, context)
        if isinstance(input, Command):
            yield from self._compiled.stream(input, config, **kwargs)
            return

        state = self.entry_hook(input, config)
        state = self.after_entry_hook(state, config)
        last_chunk: Any = None
        for chunk in self._compiled.stream(state, config, **kwargs):
            last_chunk = chunk
            yield chunk
        state = self._final_stream_state(config, state, last_chunk)
        state = self.exit_hook(state, config)
        self.after_exit_hook(state, config)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        context: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        config = self._prepare_config(config, context)
        if isinstance(input, Command):
            async for chunk in self._compiled.astream(input, config, **kwargs):
                yield chunk
            return

        state = await self.aentry_hook(input, config)
        state = self.after_entry_hook(state, config)
        last_chunk: Any = None
        async for chunk in self._compiled.astream(state, config, **kwargs):
            last_chunk = chunk
            yield chunk
        state = await self._afinal_stream_state(config, state, last_chunk)
        state = await self.aexit_hook(state, config)
        self.after_exit_hook(state, config)

    def get_state(self, config: RunnableConfig, *, subgraphs: bool = False):
        return self._compiled.get_state(config, subgraphs=subgraphs)

    async def aget_state(self, config: RunnableConfig, *, subgraphs: bool = False):
        return await self._compiled.aget_state(config, subgraphs=subgraphs)

    def get_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        return self._compiled.get_state_history(
            config,
            filter=filter,
            before=before,
            limit=limit,
        )

    async def aget_state_history(
        self,
        config: RunnableConfig,
        *,
        filter: dict | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        return await self._compiled.aget_state_history(
            config,
            filter=filter,
            before=before,
            limit=limit,
        )

    def update_state(
        self,
        config: RunnableConfig,
        values: dict | None,
        as_node: str | None = None,
    ):
        return self._compiled.update_state(config, values, as_node=as_node)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: dict | None,
        as_node: str | None = None,
    ):
        return await self._compiled.aupdate_state(config, values, as_node=as_node)

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

    def _snapshot_parent_state(self, state: dict) -> None:
        snapshot = {k: v for k, v in state.items() if k != "__subchain_stack__"}
        snapshot = copy.deepcopy(snapshot)
        stack = state.setdefault("__subchain_stack__", [])
        stack.append({"agent_name": self.name, "saved_state": snapshot})
        if "progress" not in state:
            state["progress"] = {}

    def _apply_entry_policy(self, state: dict) -> None:
        policy = self.subchain_policy
        if policy is None:
            return

        if policy.clear_messages:
            state["messages"] = []

        state["todo_lists"] = {}

        if policy.reset_iteration:
            state["iteration_number"] = 0

        if policy.max_iterations is not None:
            state["max_iterations"] = policy.max_iterations

        if not policy.preserve_chat_with_operator:
            state["chat_with_operator"] = []

    def _apply_exit_policy(self, state: dict) -> dict:
        policy = self.subchain_policy
        stack = state.get("__subchain_stack__", [])
        if policy is None or not stack:
            return state

        frame = stack.pop()
        saved = frame["saved_state"]

        if "progress" in state:
            saved["progress"] = state["progress"]
        if "current_agent_report" in state:
            saved["current_agent_report"] = state["current_agent_report"]

        for field_name in policy.merge_fields:
            if field_name in state:
                saved[field_name] = state[field_name]

        for field_name in policy.discard_fields:
            saved.pop(field_name, None)

        saved["__subchain_stack__"] = stack
        return saved

    def entry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self.subchain_policy is not None:
            self._snapshot_parent_state(state)
            state = self.graph.entry_hook(self, state, config)
            self._apply_entry_policy(state)
            return state

        return self.graph.entry_hook(self, state, config)

    async def aentry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self.subchain_policy is not None:
            self._snapshot_parent_state(state)
            state = await self.graph.aentry_hook(self, state, config)
            self._apply_entry_policy(state)
            return state

        return await self.graph.aentry_hook(self, state, config)

    def exit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        state = self._apply_exit_policy(state)
        return self.graph.exit_hook(self, state, config)

    async def aexit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        state = self._apply_exit_policy(state)
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


RunnableSeq.register(CompiledGraph)
