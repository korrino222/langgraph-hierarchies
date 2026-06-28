"""US-01 SimpleGraph tests."""

from langgraph_hierarchies.graphs.compiled import CompiledGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults


class EchoSimpleGraph(SimpleGraph):
    name = "echo_simple"
    description = "Echoes a marker into state"

    def computation_node(self, state: dict, config=None) -> dict:
        return {"current_agent_report": "computed"}


def test_simple_graph_runs_computation_node() -> None:
    graph = EchoSimpleGraph(state_schema=BaseState)
    compiled = graph.compile_graph()
    assert isinstance(compiled, CompiledGraph)

    result = compiled.invoke(create_base_state_defaults())
    assert result["current_agent_report"] == "computed"
