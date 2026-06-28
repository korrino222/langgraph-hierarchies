"""ReAct tool-calling graph type."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, convert_runnable_to_tool
from langgraph.constants import END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from langgraph_hierarchies.exceptions import AgentStuckError
from langgraph_hierarchies.graphs.base import BaseGraph
from langgraph_hierarchies.graphs.compiled import CompiledGraph
from langgraph_hierarchies.tools.builtins import internal_toolkit, supervisor_toolkit

logger = logging.getLogger(__name__)

DEFAULT_MESSAGE_SYSTEM = """
You are a helpful assistant operating within a multi-agent system.
Complete the task within your scope, then report to your supervisor.
"""

DEFAULT_MESSAGE_REASONING = "What should we do now?"

MAX_ITERATIONS = 50


class ReactArgsSchema(BaseModel):
    task: str = Field(
        default="",
        description="Specific task instructions for this agent.",
    )
    task_scope: str = Field(
        default="",
        description="Boundaries of what this agent is allowed to do.",
    )
    task_iterations: int = Field(
        default=0,
        description="Maximum iterations for this agent; 0 uses the compile-time default.",
    )


class ReactGraph(BaseGraph):
    """Tool-calling ReAct loop with optional subgraph delegation."""

    reports_to_supervisor: bool
    tools: list[BaseTool]
    max_iterations: int
    message_system: str = DEFAULT_MESSAGE_SYSTEM
    message_reasoning: str = DEFAULT_MESSAGE_REASONING

    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        config: RunnableConfig | None = None,
        include_in_progress=BaseGraph._UNSET,
        max_iterations: int = MAX_ITERATIONS,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("args_schema", ReactArgsSchema)
        if include_in_progress is not BaseGraph._UNSET:
            kwargs["include_in_progress"] = include_in_progress
        super().__init__(*args, **kwargs)

        self.reports_to_supervisor = kwargs.get("reports_to_supervisor", True)
        if self.reports_to_supervisor:
            self.tools = internal_toolkit()
        else:
            self.tools = supervisor_toolkit()

        self.additional_tools = tools or []
        self.compiled_subgraphs_as_tools: list[BaseTool] = []
        self.compiled_subgraphs: list[CompiledGraph] = []
        self.compiled_subgraphs_front: list[CompiledGraph] = []
        self.compiled_subgraphs_back: list[CompiledGraph] = []
        self.config = config or {}
        self.conditional_states: dict[Hashable, str] = {}
        self.max_iterations = max_iterations
        self.current_front = "system"
        self.current_back = "empty_back"

    def system(self, state: dict) -> dict:
        messages = [SystemMessage(content=self.message_system)]
        agent_args = state.get("current_agent_args", {})

        task = agent_args.get("task", "")
        if task:
            messages.append(HumanMessage(content=f"[TASK FROM SUPERVISOR]\n{task}"))

        task_scope = agent_args.get("task_scope", "")
        if task_scope:
            messages.append(
                HumanMessage(
                    content=(
                        f"[TASK SCOPE]\n"
                        f"Your scope is LIMITED TO: {task_scope}\n"
                        "Do NOT perform work outside this scope."
                    )
                )
            )

        task_iterations = agent_args.get("task_iterations", 0)
        if task_iterations > 0:
            effective_max = min(task_iterations, self.max_iterations)
        else:
            effective_max = self.max_iterations

        return {
            "messages": messages,
            "iteration_number": 0,
            "max_iterations": effective_max,
        }

    def reasoning(
        self,
        state: dict,
        config: RunnableConfig,
        runtime: Runtime,
    ) -> dict:
        current_iteration = state.get("iteration_number", 0) + 1
        max_iters = state.get("max_iterations", self.max_iterations)
        iterations_left = max_iters - current_iteration
        iteration_info = (
            f"\n\n[Iteration {current_iteration} of {max_iters} — "
            f"{iterations_left} iterations left]"
        )
        reasoning_message = HumanMessage(content=self.message_reasoning + iteration_info)

        model = runtime.context.model
        if model is None:
            raise ValueError(
                f"{self.name}: runtime.context.model is None. "
                "Pass a BaseChatModel via context= at invocation time."
            )
        if isinstance(model, str):
            raise TypeError(
                f"{self.name}: runtime.context.model must be a BaseChatModel instance, "
                f"not a string."
            )

        model_with_tools = model.bind_tools(self.tools)
        response = model_with_tools.invoke(state["messages"] + [reasoning_message], config)

        return {
            "messages": [reasoning_message, response],
            "iteration_number": current_iteration,
        }

    def empty_back(self, state: dict) -> Command:
        reasoning_result = state["messages"][-1]
        report_args = reasoning_result.tool_calls[0]["args"]

        if "report" in report_args:
            updated_report = report_args["report"]
        elif "result" in report_args:
            updated_report = report_args["result"]
        else:
            updated_report = str(report_args)

        return Command(update={"current_agent_report": updated_report})

    def final_back(self, state: dict) -> dict:
        return state

    def determine_action(self, state: dict) -> str | list[Send]:
        reasoning_result = state["messages"][-1]
        if len(reasoning_result.tool_calls) == 0:
            return "reasoning"

        tool_call = reasoning_result.tool_calls[0]
        logger.info("Tool call: %s", tool_call["name"])

        try:
            result = self._handle_tool_call(state, reasoning_result, tool_call)
        except AgentStuckError:
            raise

        if isinstance(result, str):
            return result
        if isinstance(result, Send):
            return [result]
        return "tool"

    def _handle_tool_call(
        self,
        state: dict,
        reasoning_result,
        tool_call: dict,
    ) -> Send | str | None:
        call_name = tool_call["name"]

        for subgraph in self.compiled_subgraphs:
            if call_name == subgraph.name:
                return Send(
                    subgraph.node_label,
                    {
                        **state,
                        "current_agent_args": tool_call["args"],
                        "current_tool_call": tool_call,
                    },
                )

        if call_name == "raise_exception":
            reason = tool_call["args"].get("reason", "No reason provided")
            subject_agent = tool_call["args"].get("subject_agent") or ""
            raise AgentStuckError(
                f"Agent raised exception: {reason}",
                reporting_agent=self.name,
                subject_agent=subject_agent,
            )

        if call_name == "report_to_supervisor":
            if len(reasoning_result.tool_calls) > 1:
                return "invalid_call_report_to_supervisor"
            content = tool_call["args"].get("report", "")
            if not content:
                return None
            return Send("empty_back", {**state, "current_agent_report": content})

        if call_name == "finish_task":
            if len(reasoning_result.tool_calls) > 1:
                return "invalid_call_report_to_supervisor"
            return Send(
                "empty_back",
                {**state, "current_agent_report": tool_call["args"]["result"]},
            )

        return None

    def invalid_call_report_to_supervisor(self, state: dict) -> dict:
        from langgraph_hierarchies.tools.builtins import tool_fail_message

        result = state["messages"][-1]
        messages = [
            tool_fail_message(
                tool_call,
                "Multiple tool calls detected alongside report_to_supervisor. "
                "Call other tools first, then report in a separate step.",
            )
            for tool_call in result.tool_calls
        ]
        return {"messages": messages}

    def build_topology(self) -> None:
        self.add_node("system", self.system)
        self.add_node("reasoning", self.reasoning)
        self.add_node("empty_back", self.empty_back)
        self.add_node("final_back", self.final_back)
        self.add_node("invalid_call_report_to_supervisor", self.invalid_call_report_to_supervisor)

        self.add_edge("system", "reasoning")
        self.add_edge("invalid_call_report_to_supervisor", "reasoning")
        self.add_edge("final_back", END)

        self.conditional_states = {
            "reasoning": "reasoning",
            "tool": "tool",
            "invalid_call_report_to_supervisor": "invalid_call_report_to_supervisor",
            "END": "empty_back",
        }

    def _prepare(self, **kwargs: Any) -> None:
        self.compiled_subgraphs = kwargs.get("compiled_subgraphs") or []
        self.compiled_subgraphs_front = kwargs.get("compiled_subgraphs_front") or []
        self.compiled_subgraphs_back = kwargs.get("compiled_subgraphs_back") or []

    def _attach_subgraphs(self) -> None:
        for subgraph in self.compiled_subgraphs:
            if subgraph.as_tool:
                self._merge_subgraph_as_tool(subgraph)
        self._merge_front_compiled_subgraphs()
        self._merge_back_compiled_subgraphs()

    def _finalize_topology(self) -> None:
        self.set_entry_point(self.current_front)

        self.tools = list(self.tools)
        self.tools.extend(self.additional_tools)
        self.tools.extend(self.compiled_subgraphs_as_tools)

        tool_node = ToolNode(self.tools)
        self.add_node("tool", tool_node)
        self.add_edge("tool", "reasoning")

        self.add_conditional_edges(
            "reasoning",
            self.determine_action,
            self.conditional_states,
        )
        self.add_edge(self.current_back, "final_back")

    def _merge_subgraph_as_tool(self, subgraph: CompiledGraph) -> None:
        self.add_node(subgraph.node_label, subgraph)
        self.add_edge(subgraph.node_label, "reasoning")

        graph_tool = convert_runnable_to_tool(
            subgraph,
            name=subgraph.name,
            description=subgraph.description,
            args_schema=subgraph.args_schema,
        )
        self.compiled_subgraphs_as_tools.append(graph_tool)
        self.conditional_states[subgraph.node_label] = subgraph.node_label

    def _merge_front_compiled_subgraphs(self) -> None:
        self.current_front = "system"
        for subgraph in reversed(self.compiled_subgraphs_front):
            self.add_node(subgraph.node_label, subgraph)
            self.add_edge(subgraph.node_label, self.current_front)
            self.current_front = subgraph.node_label

    def _merge_back_compiled_subgraphs(self) -> None:
        self.current_back = "empty_back"
        for subgraph in self.compiled_subgraphs_back:
            self.add_node(subgraph.node_label, subgraph)
            self.add_edge(self.current_back, subgraph.node_label)
            self.current_back = subgraph.node_label
