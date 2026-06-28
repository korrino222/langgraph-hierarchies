"""US-03 iteration budget enforcement tests."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import Field

from langgraph_hierarchies.graphs.react import (
    MAX_ITERATION_THRESHOLD,
    ReactGraph,
)
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults


class CaptureModel(BaseChatModel):
    """Model that records the last prompt and returns an empty tool-less reply."""

    last_messages: list = []

    @property
    def _llm_type(self) -> str:
        return "capture-model"

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.last_messages = messages
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def bind_tools(self, tools, **kwargs):
        return self


class ScriptModel(BaseChatModel):
    """Deterministic model that always requests a configured tool."""

    tool_name: str = "report_to_supervisor"
    tool_args: dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "script-model"

    def __init__(
        self,
        tool_name: str = "report_to_supervisor",
        tool_args: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tool_name=tool_name,
            tool_args=tool_args or {},
            **kwargs,
        )

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": self.tool_args,
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


class BudgetReactGraph(ReactGraph):
    name = "budget_agent"
    description = "Agent for iteration budget tests"


@tool
def dummy_work(value: str = "work") -> str:
    """Perform dummy work for iteration budget tests."""
    return value


def _runtime(model: BaseChatModel) -> Any:
    runtime = type("Runtime", (), {})()
    runtime.context = BaseContext(model=model)
    return runtime


def test_constructor_max_iterations_override() -> None:
    graph = BudgetReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        max_iterations=12,
    )
    assert graph.max_iterations == 12


def test_system_clamps_task_iterations() -> None:
    graph = BudgetReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        max_iterations=20,
    )
    result = graph.system(
        {
            **create_base_state_defaults(),
            "current_agent_args": {"task_iterations": 100},
        }
    )
    assert result["max_iterations"] == 20


def test_system_uses_task_iterations_when_lower() -> None:
    graph = BudgetReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        max_iterations=20,
    )
    result = graph.system(
        {
            **create_base_state_defaults(),
            "current_agent_args": {"task_iterations": 8},
        }
    )
    assert result["max_iterations"] == 8


def test_reasoning_countdown_normal() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    model = CaptureModel()
    state = {**create_base_state_defaults(), "max_iterations": 50, "messages": []}

    graph.reasoning(state, RunnableConfig(), _runtime(model))

    content = model.last_messages[-1].content
    assert "Iteration 1 of 50 — 49 iterations left" in content
    assert "WRAP UP NOW" not in content
    assert "BUDGET EXHAUSTED" not in content


def test_reasoning_countdown_wrap_up() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    model = CaptureModel()
    state = {
        **create_base_state_defaults(),
        "max_iterations": 50,
        "iteration_number": 47,
        "messages": [],
    }

    graph.reasoning(state, RunnableConfig(), _runtime(model))

    content = model.last_messages[-1].content
    assert "Iteration 48 of 50 — 2 iterations left — WRAP UP NOW" in content


def test_reasoning_countdown_budget_exhausted() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    model = CaptureModel()
    state = {
        **create_base_state_defaults(),
        "max_iterations": 50,
        "iteration_number": 50,
        "messages": [],
    }

    graph.reasoning(state, RunnableConfig(), _runtime(model))

    content = model.last_messages[-1].content
    assert "BUDGET EXHAUSTED — Iteration 51 of 50" in content
    assert "You MUST call report_to_supervisor NOW" in content


def test_determine_action_rejects_non_reporting_at_soft_limit() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    state = {
        **create_base_state_defaults(),
        "max_iterations": 5,
        "iteration_number": 5,
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "dummy_work",
                        "args": {"value": "x"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    }

    assert graph.determine_action(state) == "reject_non_reporting_call"


def test_determine_action_allows_report_at_soft_limit() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    state = {
        **create_base_state_defaults(),
        "max_iterations": 5,
        "iteration_number": 5,
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "report_to_supervisor",
                        "args": {"report": "done"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    }

    result = graph.determine_action(state)
    assert isinstance(result, list)
    assert result[0].node == "empty_back"


def test_determine_action_forced_exit_at_hard_limit() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    max_iters = 5
    state = {
        **create_base_state_defaults(),
        "max_iterations": max_iters,
        "iteration_number": max_iters + MAX_ITERATION_THRESHOLD,
        "messages": [AIMessage(content="", tool_calls=[])],
    }

    assert graph.determine_action(state) == "forced_exit"


def test_forced_exit_produces_structured_report() -> None:
    graph = BudgetReactGraph(state_schema=BaseState, context_schema=BaseContext)
    state = {
        **create_base_state_defaults(),
        "max_iterations": 5,
        "iteration_number": 10,
        "current_agent_report": "partial result",
    }

    result = graph.forced_exit(state)
    assert isinstance(result, Command)
    report = result.update["current_agent_report"]
    assert report.startswith("[FORCED EXIT]")
    assert "max_iterations=5" in report
    assert "iterations_used=10" in report
    assert "Partial progress:\npartial result" in report


def test_e2e_forced_exit_when_budget_exhausted() -> None:
    graph = BudgetReactGraph(
        state_schema=BaseState,
        context_schema=BaseContext,
        max_iterations=3,
        tools=[dummy_work],
    )
    compiled = graph.compile_graph()
    context = BaseContext(model=ScriptModel("dummy_work", {"value": "keep going"}))

    result = compiled.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )

    assert result["current_agent_report"].startswith("[FORCED EXIT]")
    assert "max_iterations=3" in result["current_agent_report"]
