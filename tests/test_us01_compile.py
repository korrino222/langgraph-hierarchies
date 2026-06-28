"""US-01 compile pipeline tests."""

import pytest

from langgraph_hierarchies.graphs.compiled import CompiledGraph
from langgraph_hierarchies.graphs.react import ReactArgsSchema, ReactGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.state.schema import BaseState

pytestmark = pytest.mark.us01


class CompileTestGraph(SimpleGraph):
    name = "compile_test"
    description = "Compile test graph"


def test_compile_graph_returns_compiled_graph() -> None:
    graph = CompileTestGraph(state_schema=BaseState)
    compiled = graph.compile_graph()
    assert isinstance(compiled, CompiledGraph)
    assert compiled.name == "compile_test"
    assert compiled.description == "Compile test graph"


def test_compile_as_root_stores_state_defaults() -> None:
    graph = CompileTestGraph(state_schema=BaseState)
    defaults = {"current_agent_report": "seed"}
    compiled = graph.compile_as_root(state_defaults=defaults)
    assert compiled._root_state_defaults == defaults


def test_react_graph_defaults_to_react_args_schema() -> None:
    class DefaultReactGraph(ReactGraph):
        name = "react_default"
        description = "React default args schema"

    graph = DefaultReactGraph(state_schema=BaseState)
    compiled = graph.compile_graph()
    assert compiled.args_schema is ReactArgsSchema
