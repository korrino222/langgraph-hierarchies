"""Runnable IRS reporting hierarchy example."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from examples.irs_reporting.agents import compile_root
from examples.irs_reporting.data import NUM_POSITIONS
from examples.irs_reporting.model import RuleBasedModel
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults


def build_context(*, num_positions: int = NUM_POSITIONS) -> BaseContext:
    return BaseContext(model=RuleBasedModel.for_pipeline(num_positions))


def main() -> None:
    root = compile_root(num_positions=NUM_POSITIONS)
    context = build_context(num_positions=NUM_POSITIONS)
    recursion_limit = NUM_POSITIONS * 8 + 100
    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=recursion_limit),
        context=context,
    )
    print(result["current_agent_report"])


if __name__ == "__main__":
    main()
