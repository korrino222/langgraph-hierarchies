"""US-05 LangGraph compatibility harness meta-tests."""

from __future__ import annotations

import re
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults

pytestmark = pytest.mark.us05

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

STORY_STATUS: dict[str, str] = {
    "us01": "implemented",
    "us02": "implemented",
    "us03": "implemented",
    "us04": "implemented",
    "us05": "implemented",
    "us06": "implemented",
    "us07": "planned",
}


def _parse_version(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _version_at_least(left: str, right: str) -> bool:
    return _parse_version(left) >= _parse_version(right)


def _version_below(left: str, right: str) -> bool:
    return _parse_version(left) < _parse_version(right)


def _parse_langgraph_pin(pyproject_text: str) -> tuple[str, str | None]:
    match = re.search(
        r'"langgraph>=([\d.]+)(?:,<([\d.]+))?"',
        pyproject_text,
    )
    if match is None:
        msg = "Could not parse langgraph pin from pyproject.toml"
        raise AssertionError(msg)
    return match.group(1), match.group(2)


def _parse_readme_langgraph_version(readme_text: str) -> str:
    match = re.search(
        r"\|\s*0\.0\.x\s*\|\s*([\d.]+)",
        readme_text,
    )
    if match is None:
        msg = "Could not parse documented langgraph version from README.md"
        raise AssertionError(msg)
    return match.group(1)


def _story_modules(story: str) -> list[Path]:
    story_num = story.removeprefix("us")
    return sorted(TESTS_DIR.glob(f"test_us{story_num}_*.py"))


def test_implemented_stories_have_regression_modules() -> None:
    for story, status in STORY_STATUS.items():
        if status != "implemented":
            continue
        modules = _story_modules(story)
        assert modules, f"{story} is implemented but has no test_us*_*.py module"


def test_story_modules_declare_matching_markers() -> None:
    for path in sorted(TESTS_DIR.glob("test_us*.py")):
        match = re.match(r"test_us(\d\d)_", path.name)
        if match is None:
            continue
        story = f"us{match.group(1)}"
        content = path.read_text(encoding="utf-8")
        assert f"pytest.mark.{story}" in content, (
            f"{path.name} must declare pytestmark = pytest.mark.{story}"
        )


def test_installed_langgraph_matches_documented_pin() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    min_version, max_exclusive = _parse_langgraph_pin(pyproject_text)
    documented_version = _parse_readme_langgraph_version(readme_text)
    installed_version = pkg_version("langgraph")

    assert documented_version == min_version, (
        "README compatibility matrix must document the minimum pinned langgraph version"
    )
    assert _version_at_least(installed_version, min_version), (
        f"Installed langgraph {installed_version} is below pin {min_version}"
    )
    if max_exclusive is not None:
        assert _version_below(installed_version, max_exclusive), (
            f"Installed langgraph {installed_version} violates upper bound <{max_exclusive}"
        )


class _NoNetworkModel(BaseChatModel):
    """Deterministic model used to prove tests do not require live LLM APIs."""

    @property
    def _llm_type(self) -> str:
        return "no-network-model"

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
                    "args": {"report": "ok"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


class _HarnessSmokeGraph(ReactGraph):
    name = "harness_smoke"
    description = "US-05 compatibility harness smoke graph"


def test_react_graph_compiles_without_llm_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    graph = _HarnessSmokeGraph(state_schema=BaseState, context_schema=BaseContext)
    compiled = graph.compile_graph()
    context = BaseContext(model=_NoNetworkModel())

    assert compiled.name == "harness_smoke"
    assert compiled._compiled is not None

    result = compiled.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=25),
        context=context,
    )
    assert result["current_agent_report"] == "ok"
