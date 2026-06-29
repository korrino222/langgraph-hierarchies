"""IRS reporting hierarchy fixture — compile root and build context."""

from __future__ import annotations

from irs_hierarchy.agents import compile_root
from irs_hierarchy.data import NUM_POSITIONS
from irs_hierarchy.model import RuleBasedModel
from langgraph_hierarchies.state.context import BaseContext


def build_context(*, num_positions: int = NUM_POSITIONS) -> BaseContext:
    return BaseContext(model=RuleBasedModel.for_pipeline(num_positions))


__all__ = ["build_context", "compile_root"]
