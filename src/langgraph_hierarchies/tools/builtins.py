"""Built-in tools for hierarchy agents."""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from langgraph_hierarchies.exceptions import AgentStuckError

try:
    from langchain_core.tools import InjectedToolCallId
except ImportError:  # pragma: no cover - older langchain-core fallback
    from langgraph.prebuilt import InjectedToolCallId  # type: ignore[no-redef]


@tool
def report_to_supervisor(
    report: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    request_iterations: int = 0,
) -> Command:
    """Report the current agent's work to its supervising agent."""
    if request_iterations > 0:
        report += (
            f"\n\n[REQUEST_ITERATIONS: {request_iterations}] "
            "This agent requests additional iterations to complete the task."
        )

    return Command(
        update={
            "current_agent_report": report,
            "messages": [
                ToolMessage(
                    content=report,
                    tool_call_id=tool_call_id,
                    id=str(uuid.uuid4()),
                )
            ],
        }
    )


@tool
def finish_task(
    result: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Mark the current task complete and return the final result."""
    return Command(
        update={
            "current_agent_report": result,
            "is_finished": True,
            "messages": [
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call_id,
                    id=str(uuid.uuid4()),
                )
            ],
        }
    )


@tool
def raise_exception(
    reason: str,
    subject_agent: str = "",
) -> str:
    """Signal that the agent is stuck and cannot continue."""
    raise AgentStuckError(
        f"Agent raised exception: {reason}",
        subject_agent=subject_agent,
    )


def tool_fail_message(tool_call: ToolCall, error_message: str) -> ToolMessage:
    """Build an error ToolMessage for a failed tool call."""
    return ToolMessage(
        content=f"Error: {error_message}",
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        id=str(uuid.uuid4()),
    )


def internal_toolkit() -> list[BaseTool]:
    """Tools for agents that report to a supervisor."""
    return [report_to_supervisor, raise_exception]


def supervisor_toolkit() -> list[BaseTool]:
    """Tools for root/supervisor agents without a supervisor."""
    return [raise_exception]
