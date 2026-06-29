"""US-06 IRS multi-stage hierarchy tests."""

from __future__ import annotations

import pytest
from irs_hierarchy.agents import (
    ARTIFACT_POLICY,
    EvidenceOrchestrator,
    IRSState,
    compile_root,
)
from irs_hierarchy.data import build_evidence_artifact
from irs_hierarchy.hierarchy import build_context
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.schema import create_base_state_defaults

pytestmark = pytest.mark.us06

TEST_POSITIONS = 3


def test_full_hierarchy_runs_end_to_end() -> None:
    root = compile_root(num_positions=TEST_POSITIONS)
    context = build_context(num_positions=TEST_POSITIONS)
    recursion_limit = TEST_POSITIONS * 8 + 100

    result = root.invoke(
        create_base_state_defaults(),
        config=RunnableConfig(recursion_limit=recursion_limit),
        context=context,
    )

    report = result["current_agent_report"]
    assert "IRS REPORT COMPLETE" in report
    assert result["is_finished"] is True


def test_subagent_policy_clears_messages_but_merges_artifact() -> None:
    evidence = EvidenceOrchestrator(
        state_schema=IRSState,
        context_schema=BaseContext,
        subagent_policy=ARTIFACT_POLICY,
    ).compile_graph()
    fetcher = next(
        child
        for child in evidence.compiled_subgraphs
        if child.name == "document_fetcher"
    )

    parent_state = create_base_state_defaults()
    parent_state["messages"] = [HumanMessage(content="upstream history must not leak")]
    parent_state["pipeline_artifact"] = "prior-artifact"

    child_input = {
        **parent_state,
        "current_agent_args": {"task": "fetch", "task_scope": "fetch only"},
        "current_tool_call": {
            "name": "document_fetcher",
            "args": {},
            "id": "call-fetch",
            "type": "tool_call",
        },
    }

    entered = fetcher.entry_hook(child_input)
    assert entered["messages"] == []

    child_output = fetcher.invoke(entered)
    restored = fetcher.exit_hook(child_output)

    assert restored["messages"][0].content == "upstream history must not leak"
    assert restored["pipeline_artifact"].startswith("FETCHED:")


def test_evidence_stage_runs_in_isolation_with_fixture_artifact() -> None:
    orchestrator = EvidenceOrchestrator(
        state_schema=IRSState,
        context_schema=BaseContext,
    ).compile_graph()

    from irs_hierarchy.model import RuleBasedModel, build_all_responses

    responses = build_all_responses(TEST_POSITIONS)
    evidence_responses = responses[2:6]
    context = BaseContext(model=RuleBasedModel(responses=evidence_responses))

    state = create_base_state_defaults()
    state["pipeline_artifact"] = "seed-input"

    result = orchestrator.invoke(
        state,
        config=RunnableConfig(recursion_limit=50),
        context=context,
    )

    assert result["current_agent_report"] == build_evidence_artifact()
