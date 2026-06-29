"""Deterministic fixture data for the IRS reporting hierarchy tests."""

from __future__ import annotations

NUM_POSITIONS = 200

STAGE_NAMES = [
    "evidence",
    "extraction",
    "matching",
    "reconciliation",
    "reporting",
]

SOURCE_DOCUMENTS = [
    {"id": "doc-001", "type": "w2", "taxpayer_id": "TX-1001"},
    {"id": "doc-002", "type": "1099-int", "taxpayer_id": "TX-1001"},
    {"id": "doc-003", "type": "bank-statement", "taxpayer_id": "TX-1001"},
]


def position_ids(count: int | None = None) -> list[str]:
    """Return synthetic bank position identifiers."""
    total = count if count is not None else NUM_POSITIONS
    return [f"position-{index:03d}" for index in range(1, total + 1)]


def build_evidence_artifact() -> str:
    doc_ids = ", ".join(document["id"] for document in SOURCE_DOCUMENTS)
    return f"EVIDENCE:validated documents=[{doc_ids}]"


def build_extraction_artifact(evidence_artifact: str) -> str:
    return f"EXTRACTION:ocr_text from ({evidence_artifact})"


def build_matching_artifact(extraction_artifact: str, count: int | None = None) -> str:
    positions = position_ids(count)
    return (
        f"MATCHING:matched {len(positions)} positions from ({extraction_artifact}); "
        f"sample={positions[0]}..{positions[-1]}"
    )


def build_reconciliation_artifact(matching_artifact: str) -> str:
    return f"RECONCILIATION:tax_due=1250.00 derived from ({matching_artifact})"


def build_final_report(reconciliation_artifact: str) -> str:
    return f"IRS REPORT COMPLETE\nStatus: filed\nSummary: {reconciliation_artifact}"
