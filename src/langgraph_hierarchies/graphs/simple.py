"""Deterministic computation graph type."""

from langchain_core.runnables import RunnableConfig
from langgraph.constants import END

from langgraph_hierarchies.graphs.base import BaseGraph


class SimpleGraph(BaseGraph):
    """Single-node graph for deterministic computation."""

    def computation_node(
        self,
        state: dict,
        config: RunnableConfig,
    ) -> dict:
        """Override to implement deterministic computation."""
        return state

    def build_topology(self) -> None:
        self.add_node("computation_node", self.computation_node)
        self.add_edge("computation_node", END)
        self.set_entry_point("computation_node")
