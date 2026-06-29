"""Base state schema shared by all hierarchy graphs."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolCall
from langgraph.managed import RemainingSteps

from langgraph_hierarchies.state.reducers import (
    reduce_current_agent_args,
    reduce_current_agent_report,
    reduce_current_tool_call,
    reduce_file_refs,
    reduce_is_cancelled,
    reduce_is_finished,
    reduce_iteration_number,
    reduce_messages,
    reduce_progress,
    reduce_subagent_stack,
    reduce_todo_list,
    reduce_todo_lists,
)
from langgraph_hierarchies.types import Progress


class BaseState(TypedDict):
    """Common mutable execution state for hierarchy agents."""

    messages: Annotated[list[AnyMessage], reduce_messages]
    todo_list: Annotated[dict[str, bool], reduce_todo_list]
    todo_lists: Annotated[dict[str, dict[str, bool]], reduce_todo_lists]
    chat_with_operator: Annotated[list[AnyMessage], reduce_messages]
    current_agent_args: Annotated[dict, reduce_current_agent_args]
    current_agent_report: Annotated[str, reduce_current_agent_report]
    current_tool_call: Annotated[ToolCall | None, reduce_current_tool_call]
    is_finished: Annotated[bool, reduce_is_finished]
    is_cancelled: Annotated[bool, reduce_is_cancelled]
    progress: Annotated[dict[str, Progress], reduce_progress]
    iteration_number: Annotated[int, reduce_iteration_number]
    max_iterations: Annotated[int, reduce_iteration_number]
    file_refs: Annotated[list, reduce_file_refs]
    __subagent_stack__: Annotated[list, reduce_subagent_stack]
    remaining_steps: RemainingSteps


def create_base_state_defaults() -> dict:
    """Return empty defaults for all BaseState channels."""
    return {
        "messages": [],
        "todo_list": {},
        "todo_lists": {},
        "chat_with_operator": [],
        "current_agent_args": {},
        "current_agent_report": "",
        "current_tool_call": None,
        "is_finished": False,
        "is_cancelled": False,
        "progress": {},
        "iteration_number": 0,
        "max_iterations": 40,
        "file_refs": [],
        "__subagent_stack__": [],
    }
