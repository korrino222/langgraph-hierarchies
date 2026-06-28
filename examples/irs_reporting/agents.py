"""Graph classes for the IRS reporting hierarchy example."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from examples.irs_reporting.data import (
    NUM_POSITIONS,
    SOURCE_DOCUMENTS,
    STAGE_NAMES,
    build_evidence_artifact,
    build_extraction_artifact,
    build_final_report,
    build_reconciliation_artifact,
)
from langgraph_hierarchies.graphs.compiled import SubchainPolicy
from langgraph_hierarchies.graphs.react import ReactGraph
from langgraph_hierarchies.graphs.simple import SimpleGraph
from langgraph_hierarchies.graphs.todo import TodoGraph
from langgraph_hierarchies.state.context import BaseContext
from langgraph_hierarchies.state.reducers import reduce_current_agent_report
from langgraph_hierarchies.state.schema import BaseState, create_base_state_defaults

ARTIFACT_POLICY = SubchainPolicy(
    clear_messages=True, merge_fields=["pipeline_artifact"]
)


def root_stage_todo_check(state: dict) -> bool:
    todo_list = state.get("todo_list") or {}
    stages = {name: todo_list.get(name, False) for name in STAGE_NAMES}
    return bool(stages) and all(stages.values())


def matching_position_todo_check(state: dict) -> bool:
    todo_list = state.get("todo_list") or {}
    positions = {
        key: done for key, done in todo_list.items() if key.startswith("position-")
    }
    return bool(positions) and all(positions.values())


class ArtifactReactGraph(ReactGraph):
    """ReactGraph that copies reports into pipeline_artifact for stage handoff."""

    def empty_back(self, state: dict) -> Command:
        command = super().empty_back(state)
        report = command.update.get("current_agent_report", "")
        if report:
            command.update["pipeline_artifact"] = report
        return command


class IRSState(BaseState):
    """Extended state carrying the artifact passed between pipeline stages."""

    pipeline_artifact: Annotated[str, reduce_current_agent_report]


def _artifact_from_state(state: dict) -> str:
    return state.get("pipeline_artifact") or state.get("current_agent_report") or ""


class DocumentFetcher(SimpleGraph):
    name = "document_fetcher"
    description = "Fetch source documents for the evidence stage"

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        doc_ids = ", ".join(document["id"] for document in SOURCE_DOCUMENTS)
        artifact = f"FETCHED:documents=[{doc_ids}]"
        return {
            "current_agent_report": artifact,
            "pipeline_artifact": artifact,
        }


class DocumentValidator(ArtifactReactGraph):
    name = "document_validator"
    description = "Validate fetched document structure"


class OCREngine(SimpleGraph):
    name = "ocr_engine"
    description = "Run OCR over validated documents"

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        prior = _artifact_from_state(state)
        artifact = build_extraction_artifact(prior or build_evidence_artifact())
        return {
            "current_agent_report": artifact,
            "pipeline_artifact": artifact,
        }


class PositionMatcher(ArtifactReactGraph):
    name = "position_matcher"
    description = "Match a single bank position"


class TaxCalculator(SimpleGraph):
    name = "tax_calculator"
    description = "Calculate tax due from matched positions"

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        prior = _artifact_from_state(state)
        artifact = build_reconciliation_artifact(prior)
        return {
            "current_agent_report": artifact,
            "pipeline_artifact": artifact,
        }


class ReportingOrchestrator(SimpleGraph):
    name = "reporting_orchestrator"
    description = "Emit the final IRS filing report"

    def computation_node(
        self, state: dict, config: RunnableConfig | None = None
    ) -> dict:
        prior = _artifact_from_state(state)
        report = build_final_report(prior)
        return {
            "current_agent_report": report,
            "pipeline_artifact": report,
            "is_finished": True,
        }


class EvidenceOrchestrator(ArtifactReactGraph):
    name = "evidence_orchestrator"
    description = "Collect and validate source documents"

    def compile_graph(self, *args: Any, **kwargs: Any):
        fetcher = DocumentFetcher(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        validator = DocumentValidator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(
            *args,
            compiled_subgraphs=[fetcher, validator],
            **kwargs,
        )


class ExtractionOrchestrator(ArtifactReactGraph):
    name = "extraction_orchestrator"
    description = "Extract text from validated evidence"

    def compile_graph(self, *args: Any, **kwargs: Any):
        ocr = OCREngine(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(*args, compiled_subgraphs=[ocr], **kwargs)


class MatchingOrchestrator(TodoGraph):
    name = "matching_orchestrator"
    description = "Match bank positions with flat TODO-tracked context"

    def __init__(self, *args: Any, num_positions: int = NUM_POSITIONS, **kwargs: Any):
        self.matching_num_positions = num_positions
        kwargs.setdefault("todo_check", matching_position_todo_check)
        kwargs.setdefault("max_iterations", num_positions * 3 + 20)
        super().__init__(*args, **kwargs)

    def entry_hook(self, graph, state, config=None, **kwargs):
        state["todo_list"] = {}
        return state

    def empty_back(self, state: dict) -> Command:
        command = super().empty_back(state)
        report = command.update.get("current_agent_report", "")
        if report:
            command.update["pipeline_artifact"] = report
        return command

    def compile_graph(
        self, *args: Any, num_positions: int = NUM_POSITIONS, **kwargs: Any
    ):
        self.matching_num_positions = num_positions
        self.todo_check = matching_position_todo_check
        self.max_iterations = num_positions * 3 + 20
        matcher = PositionMatcher(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(
            *args,
            compiled_subgraphs=[matcher],
            **kwargs,
        )


class ReconciliationOrchestrator(ArtifactReactGraph):
    name = "reconciliation_orchestrator"
    description = "Calculate tax due from matched positions"

    def compile_graph(self, *args: Any, **kwargs: Any):
        calculator = TaxCalculator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(
            *args,
            compiled_subgraphs=[calculator],
            **kwargs,
        )


class IRSReportingRoot(TodoGraph):
    name = "irs_reporting_root"
    description = "Five-stage IRS reporting pipeline root"

    def __init__(self, *args: Any, num_positions: int = NUM_POSITIONS, **kwargs: Any):
        self.num_positions = num_positions
        kwargs.setdefault("todo_check", root_stage_todo_check)
        super().__init__(*args, **kwargs)

    def compile_graph(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        num_positions = kwargs.pop("num_positions", self.num_positions)
        reporting = ReportingOrchestrator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        evidence = EvidenceOrchestrator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        extraction = ExtractionOrchestrator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        matching = MatchingOrchestrator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph(num_positions=num_positions)
        reconciliation = ReconciliationOrchestrator(
            state_schema=IRSState,
            context_schema=BaseContext,
            subchain_policy=ARTIFACT_POLICY,
        ).compile_graph()
        return super().compile_graph(
            *args,
            compiled_subgraphs=[
                evidence,
                extraction,
                matching,
                reconciliation,
                reporting,
            ],
            **kwargs,
        )


def compile_root(*, num_positions: int = NUM_POSITIONS):
    """Compile the IRS reporting root graph."""
    root = IRSReportingRoot(
        state_schema=IRSState,
        context_schema=BaseContext,
        reports_to_supervisor=False,
        num_positions=num_positions,
    )
    return root.compile_as_root(state_defaults=create_base_state_defaults())
