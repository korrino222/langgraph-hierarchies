"""TODO-gated ReAct graph type."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.types import Send

from langgraph_hierarchies.graphs.react import MAX_ITERATIONS, ReactGraph
from langgraph_hierarchies.tools.builtins import todo_toolkit, tool_fail_message

DEFAULT_TODO_CHECK_FAIL_MESSAGE = (
    "You still have open TODO items; complete them before reporting."
)


def default_todo_check(state: dict) -> bool:
    """Allow reporting only when every TODO item is complete."""
    todo_list = state.get("todo_list") or {}
    if not todo_list:
        return False
    return all(todo_list.values())


class TodoGraph(ReactGraph):
    """ReAct graph that blocks reporting until the TODO list is complete."""

    todo_check: Callable[[dict], bool] | None = None
    todo_check_fail_message: str = DEFAULT_TODO_CHECK_FAIL_MESSAGE

    def __init__(
        self,
        tools=None,
        config=None,
        include_in_progress=ReactGraph._UNSET,
        max_iterations: int = MAX_ITERATIONS,
        todo_check: Callable[[dict], bool] | None = None,
        todo_check_fail_message: str = DEFAULT_TODO_CHECK_FAIL_MESSAGE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            tools,
            config,
            include_in_progress,
            max_iterations,
            *args,
            **kwargs,
        )
        self.todo_check = todo_check
        self.todo_check_fail_message = todo_check_fail_message
        self.additional_tools = list(self.additional_tools) + todo_toolkit()

    def _passes_todo_gate(self, state: dict) -> bool:
        checker = self.todo_check or default_todo_check
        return checker(state)

    def _handle_tool_call(
        self,
        state: dict,
        reasoning_result,
        tool_call: dict,
    ) -> Send | str | None:
        call_name = tool_call["name"]
        if call_name in {"report_to_supervisor", "finish_task"}:
            if not self._passes_todo_gate(state):
                return "todo_incomplete"
        return super()._handle_tool_call(state, reasoning_result, tool_call)

    def todo_incomplete(self, state: dict) -> dict:
        result = state["messages"][-1]
        messages = [
            tool_fail_message(tool_call, self.todo_check_fail_message)
            for tool_call in result.tool_calls
        ]
        return {"messages": messages}

    def build_topology(self) -> None:
        super().build_topology()
        self.add_node("todo_incomplete", self.todo_incomplete)
        self.add_edge("todo_incomplete", "reasoning")
        self.conditional_states["todo_incomplete"] = "todo_incomplete"
