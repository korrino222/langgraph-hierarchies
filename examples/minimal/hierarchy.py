"""Minimal parent/child hierarchy runnable without an LLM API key."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults


class ScriptedModel(BaseChatModel):
    """Deterministic model that delegates to a child, then finishes."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted-model"

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


class WorkerAgent(ReactGraph):
    name = "worker"
    description = "Completes a delegated task and reports upward"


class OrchestratorAgent(ReactGraph):
    name = "orchestrator"
    description = "Delegates work to a worker subagent"

    def compile_graph(self, *args, **kwargs):
        worker = WorkerAgent(
            state_schema=BaseState,
            context_schema=BaseContext,
        ).compile_graph()
        return super().compile_graph(*args, compiled_subgraphs=[worker], **kwargs)


def build_context() -> BaseContext:
    worker_report = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_to_supervisor",
                "args": {"report": "worker finished"},
                "id": "worker-call-1",
                "type": "tool_call",
            }
        ],
    )
    delegate_to_worker = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "worker",
                "args": {
                    "task": "Process the payload",
                    "task_scope": "Worker scope only",
                    "task_iterations": 0,
                },
                "id": "parent-call-1",
                "type": "tool_call",
            }
        ],
    )
    finish_root = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "finish_task",
                "args": {"result": "orchestrator complete"},
                "id": "parent-call-2",
                "type": "tool_call",
            }
        ],
    )
    return BaseContext(
        model=ScriptedModel(responses=[delegate_to_worker, worker_report, finish_root])
    )


def main() -> None:
    orchestrator = OrchestratorAgent(
        state_schema=BaseState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
    )
    root = orchestrator.compile_as_root(
        state_defaults=create_base_state_defaults(),
    )
    context = build_context()
    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )
    print(result["current_agent_report"])


if __name__ == "__main__":
    main()
