"""Base graph class-as-factory with phased compilation."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from pydantic import BaseModel

from langgraph_hierarchies.graphs.compiled import CompiledGraph, SubagentPolicy

logger = logging.getLogger(__name__)


class BaseGraphArgsSchema(BaseModel):
    """Default empty args schema for graphs not exposed as tools."""


class BaseGraph(StateGraph):
    """Class-as-factory base for hierarchy graphs."""

    name: str
    description: str
    args_schema: type[BaseModel]
    include_in_progress: bool = True

    _UNSET = object()

    def __init__(
        self,
        state_schema: type[Any],
        config: RunnableConfig | None = None,
        args_schema: type[BaseModel] = BaseGraphArgsSchema,
        include_in_progress=_UNSET,
        subagent_policy: SubagentPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_schema=state_schema, graph_id=self.name, **kwargs)
        self.config = config or {}
        self.conditional_states: dict[Any, str] = {}
        self.args_schema = args_schema
        if include_in_progress is not self._UNSET:
            self.include_in_progress = include_in_progress
        self.subagent_policy = subagent_policy
        self.compiled_subgraphs: list[CompiledGraph] = []
        self._enable_interrupts = False

    def build_topology(self) -> None:
        """Register core nodes and edges. Subclasses must override."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_topology()"
        )

    def compile_graph(
        self,
        *,
        checkpointer=None,
        enable_interrupts: bool = False,
        **kwargs: Any,
    ) -> CompiledGraph:
        logger.info("BaseGraph.compile_graph: %s - starting compilation", self.name)

        self._prepare(**kwargs)
        self.build_topology()
        self._attach_subgraphs()
        self._finalize_topology()

        compile_kwargs: dict[str, Any] = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer
        self._enable_interrupts = enable_interrupts
        graph_compiled = self.compile(**compile_kwargs)
        return self._wrap_compiled(graph_compiled)

    def compile_as_root(
        self,
        *,
        checkpointer=None,
        state_defaults: dict | None = None,
        enable_interrupts: bool = True,
        **kwargs: Any,
    ) -> CompiledGraph:
        """Compile for top-level invocation with root defaults and interrupts enabled."""
        compiled = self.compile_graph(
            checkpointer=checkpointer,
            enable_interrupts=enable_interrupts,
            **kwargs,
        )
        compiled._root_state_defaults = state_defaults or {}
        return compiled

    def _prepare(self, **kwargs: Any) -> None:
        if "compiled_subgraphs" in kwargs:
            self.compiled_subgraphs = kwargs["compiled_subgraphs"]

    def _attach_subgraphs(self) -> None:
        return None

    def _finalize_topology(self) -> None:
        return None

    def _wrap_compiled(self, graph_compiled) -> CompiledGraph:
        return CompiledGraph(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            graph=self,
            compiled=graph_compiled,
            compiled_subgraphs=getattr(self, "compiled_subgraphs", []),
            subagent_policy=self.subagent_policy,
        )

    def entry_hook(
        self,
        graph: CompiledGraph,
        state: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        return state

    async def aentry_hook(
        self,
        graph: CompiledGraph,
        state: dict,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict:
        return state

    def exit_hook(
        self,
        graph: CompiledGraph,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return state

    async def aexit_hook(
        self,
        graph: CompiledGraph,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        return state
