"""US-01 ReactGraph loop tests."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults

pytestmark = pytest.mark.us01


class ScriptModel(BaseChatModel):
    """Deterministic model that always requests report_to_supervisor."""

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
        message = AIMessage(
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
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


class ReportReactGraph(ReactGraph):
    name = "report_agent"
    description = "Reports to supervisor in one step"


def test_react_graph_runs_report_loop() -> None:
    graph = ReportReactGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()
    context = BaseContext(model=ScriptModel())

    result = compiled.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=25),
        context=context,
    )
    assert result["current_agent_report"] == "done"
