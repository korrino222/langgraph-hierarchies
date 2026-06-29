"""Compiled graph wrapper and subagent policy."""

from __future__ import annotations

import copy
import random
import string
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableSequence,
)
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
class SubagentPolicy:
    """Declarative subagent entry/exit policy (semantics applied in US-02)."""

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
    runnable: Runnable
    compiled_subgraphs: list[CompiledGraph]
    as_tool: bool
    subagent_policy: SubagentPolicy | None
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
        subagent_policy: SubagentPolicy | None = None,
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
        self.subagent_policy = subagent_policy
        self._root_state_defaults = None

        suffix = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        self.node_label = f"{self.name}_{suffix}"

        self.runnable = self._build_runnable()

    def _build_runnable(self) -> Runnable:
        """Assemble the hook pipeline that wraps the inner compiled graph.

        Async-capable hooks are wrapped in ``RunnableLambda`` so the chain
        works under both ``invoke`` and ``ainvoke``; the sync fallbacks are
        pass-throughs. Execution order (outermost first):

            aentry -> entry -> after_entry -> compiled -> aexit -> exit -> after_exit

        Building the chain as an explicit ``Runnable`` (rather than running
        hooks by hand) is what lets the pipeline survive being embedded as a
        graph node: LangGraph flattens it via ``steps`` and every hook -- most
        importantly ``after_exit_hook``, which emits the answering
        ``ToolMessage`` -- still runs.
        """
        aentry_step = RunnableLambda(
            func=self._sync_aentry_hook,
            afunc=self.aentry_hook,
        )
        aexit_step = RunnableLambda(
            func=self._sync_aexit_hook,
            afunc=self.aexit_hook,
        )
        core = self.after_entry_hook | self._compiled | aexit_step
        core = self.entry_hook | core | self.exit_hook
        return aentry_step | core | self.after_exit_hook

    @property
    def steps(self) -> list:
        """Expose the hook pipeline steps for ``find_subgraph_pregel`` traversal.

        Call ``find_subgraph_pregel(compiled.runnable)`` (or pass the wrapper's
        ``runnable``) to locate the embedded inner ``Pregel``. The wrapper
        itself must **not** be registered as a virtual ``RunnableSeq`` subclass:
        if it were, LangGraph would flatten the node into individual hook steps
        and detach the inner graph from the trace tree (see DD-10).
        """
        inner = self.runnable
        if isinstance(inner, (RunnableSequence, RunnableSeq)):
            return inner.steps
        return [inner]

    def _sync_aentry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        """Sync pass-through for the async entry hook."""
        return state

    def _sync_aexit_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        """Sync pass-through for the async exit hook."""
        return state

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
        return self.runnable.invoke(input, config, **kwargs)

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
        return await self.runnable.ainvoke(input, config, **kwargs)

    def _resolve_final_values(
        self,
        config: RunnableConfig | None,
        last_chunk: Any,
    ) -> dict | None:
        """Resolve the final state for exit hooks after streaming.

        Prefers the checkpointer snapshot; falls back to the last streamed
        chunk (which is the full state under ``stream_mode="values"``) when no
        checkpointer is configured.
        """
        if self._compiled.checkpointer is not None:
            snapshot = self._compiled.get_state(config)
            if snapshot is not None and snapshot.values:
                return snapshot.values
        if isinstance(last_chunk, dict):
            return last_chunk
        return None

    def _stream_with_hooks(
        self,
        input: Any,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream the inner graph with entry/exit hooks run around it.

        Used when Pregel-specific kwargs (``stream_mode``/``subgraphs``) must
        reach ``_compiled`` directly. Hooks need an active LangGraph config
        context, so we bind ``var_child_runnable_config`` before invoking them.
        """
        from langchain_core.runnables.config import (
            ensure_config,
            var_child_runnable_config,
        )

        cfg = ensure_config(config)
        token = var_child_runnable_config.set(cfg)
        try:
            state = self.after_entry_hook(self.entry_hook(input, cfg), cfg)
        finally:
            var_child_runnable_config.reset(token)

        last_chunk: Any = None
        for chunk in self._compiled.stream(state, config, **kwargs):
            last_chunk = chunk
            yield chunk

        final_values = self._resolve_final_values(config, last_chunk)
        if final_values:
            token = var_child_runnable_config.set(cfg)
            try:
                self.after_exit_hook(self.exit_hook(final_values, cfg), cfg)
            finally:
                var_child_runnable_config.reset(token)

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

        if "stream_mode" in kwargs or "subgraphs" in kwargs:
            yield from self._stream_with_hooks(input, config, **kwargs)
            return

        yield from self.runnable.stream(input, config, **kwargs)

    async def _astream_with_hooks(
        self,
        input: Any,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        from langchain_core.runnables.config import (
            ensure_config,
            var_child_runnable_config,
        )

        cfg = ensure_config(config)
        token = var_child_runnable_config.set(cfg)
        try:
            state = self.after_entry_hook(self.entry_hook(input, cfg), cfg)
        finally:
            var_child_runnable_config.reset(token)

        last_chunk: Any = None
        async for chunk in self._compiled.astream(state, config, **kwargs):
            last_chunk = chunk
            yield chunk

        final_values = self._resolve_final_values(config, last_chunk)
        if final_values:
            token = var_child_runnable_config.set(cfg)
            try:
                self.after_exit_hook(self.exit_hook(final_values, cfg), cfg)
            finally:
                var_child_runnable_config.reset(token)

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

        if "stream_mode" in kwargs or "subgraphs" in kwargs:
            async for chunk in self._astream_with_hooks(input, config, **kwargs):
                yield chunk
            return

        async for chunk in self.runnable.astream(input, config, **kwargs):
            yield chunk

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
        snapshot = {k: v for k, v in state.items() if k != "__subagent_stack__"}
        snapshot = copy.deepcopy(snapshot)
        stack = state.setdefault("__subagent_stack__", [])
        stack.append({"agent_name": self.name, "saved_state": snapshot})
        if "progress" not in state:
            state["progress"] = {}

    def _apply_entry_policy(self, state: dict) -> None:
        policy = self.subagent_policy
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
        policy = self.subagent_policy
        stack = state.get("__subagent_stack__", [])
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

        saved["__subagent_stack__"] = stack
        return saved

    def entry_hook(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        if self.subagent_policy is not None:
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
        if self.subagent_policy is not None:
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
