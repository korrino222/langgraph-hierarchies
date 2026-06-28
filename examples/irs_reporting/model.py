"""Rule-based scripted model for deterministic IRS hierarchy execution."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from examples.irs_reporting.data import (
    STAGE_NAMES,
    build_final_report,
    build_reconciliation_artifact,
    position_ids,
)


def _tool_call(
    name: str, args: dict[str, Any], *, call_id: str | None = None
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id or str(uuid4()),
                "type": "tool_call",
            }
        ],
    )


def _delegate_args(task: str, *, artifact: str = "") -> dict[str, Any]:
    payload = task
    if artifact:
        payload = f"{task}\nPrior artifact:\n{artifact}"
    return {
        "task": payload,
        "task_scope": "Stay within assigned stage responsibilities.",
        "task_iterations": 0,
    }


def build_all_responses(num_positions: int) -> list[AIMessage]:
    """Build model responses in depth-first execution order."""
    positions = position_ids(num_positions)
    evidence_artifact = "EVIDENCE:validated documents=[doc-001, doc-002, doc-003]"
    extraction_artifact = f"EXTRACTION:ocr_text from ({evidence_artifact})"
    matching_artifact = (
        f"MATCHING:matched {len(positions)} positions from ({extraction_artifact}); "
        f"sample={positions[0]}..{positions[-1]}"
    )
    reconciliation_artifact = build_reconciliation_artifact(matching_artifact)
    final_report = build_final_report(reconciliation_artifact)

    responses: list[AIMessage] = []

    responses.append(_tool_call("todo_write", {"items": list(STAGE_NAMES)}))

    responses.append(
        _tool_call("evidence_orchestrator", _delegate_args("Run the evidence stage"))
    )
    responses.append(
        _tool_call("document_fetcher", _delegate_args("Fetch source documents"))
    )
    responses.append(
        _tool_call("document_validator", _delegate_args("Validate fetched documents"))
    )
    responses.append(
        _tool_call(
            "report_to_supervisor",
            {"report": "documents structurally valid"},
        )
    )
    responses.append(
        _tool_call(
            "report_to_supervisor",
            {"report": evidence_artifact},
        )
    )
    responses.append(_tool_call("todo_complete", {"item": "evidence"}))

    responses.append(
        _tool_call(
            "extraction_orchestrator",
            _delegate_args("Run the extraction stage"),
        )
    )
    responses.append(
        _tool_call(
            "ocr_engine", _delegate_args("Extract text from validated documents")
        )
    )
    responses.append(
        _tool_call(
            "report_to_supervisor",
            {"report": extraction_artifact},
        )
    )
    responses.append(_tool_call("todo_complete", {"item": "extraction"}))

    responses.append(
        _tool_call("matching_orchestrator", _delegate_args("Run the matching stage"))
    )
    responses.append(_tool_call("todo_write", {"items": positions}))
    for position in positions:
        responses.append(
            _tool_call(
                "position_matcher",
                _delegate_args(f"Match bank position {position}"),
            )
        )
        responses.append(
            _tool_call(
                "report_to_supervisor",
                {"report": "position matched"},
            )
        )
        responses.append(_tool_call("todo_complete", {"item": position}))
    responses.append(
        _tool_call(
            "report_to_supervisor",
            {"report": matching_artifact},
        )
    )
    responses.append(_tool_call("todo_complete", {"item": "matching"}))

    responses.append(
        _tool_call(
            "reconciliation_orchestrator",
            _delegate_args("Run the reconciliation stage"),
        )
    )
    responses.append(
        _tool_call(
            "tax_calculator",
            _delegate_args("Calculate tax due from matched positions"),
        )
    )
    responses.append(
        _tool_call(
            "report_to_supervisor",
            {"report": reconciliation_artifact},
        )
    )
    responses.append(_tool_call("todo_complete", {"item": "reconciliation"}))

    responses.append(
        _tool_call(
            "reporting_orchestrator",
            _delegate_args("Run the reporting stage"),
        )
    )
    responses.append(_tool_call("todo_complete", {"item": "reporting"}))
    responses.append(_tool_call("finish_task", {"result": final_report}))

    return responses


class RuleBasedModel(BaseChatModel):
    """Deterministic model that pops pre-built tool-call responses."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "rule-based-model"

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            msg = "RuleBasedModel response queue exhausted"
            raise RuntimeError(msg)
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    @classmethod
    def for_pipeline(cls, num_positions: int) -> RuleBasedModel:
        return cls(responses=build_all_responses(num_positions))
