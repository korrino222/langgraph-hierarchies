# IRS reporting hierarchy example

Five-stage document-processing pipeline demonstrating **artifact-only handoff**
between compiled subgraphs. Each stage receives only the prior stage's artifact via
`SubagentPolicy(clear_messages=True, merge_fields=["pipeline_artifact"])`, not the
full upstream message history.

See [Decomposability in AI workflows](https://medium.com/@ishish222/decomposability-in-ai-workflows-what_it-is_and_why-you-want-it-c12c9a939565)
for the design rationale.

## Hierarchy

```
IRSReportingRoot (TodoGraph)
  ├── EvidenceOrchestrator (ReactGraph)
  │     ├── DocumentFetcher (SimpleGraph)
  │     └── DocumentValidator (ReactGraph)
  ├── ExtractionOrchestrator (ReactGraph)
  │     └── OCREngine (SimpleGraph)
  ├── MatchingOrchestrator (TodoGraph)  ← ~200 bank positions, flat context
  │     └── PositionMatcher (ReactGraph)
  ├── ReconciliationOrchestrator (ReactGraph)
  │     └── TaxCalculator (SimpleGraph)
  └── ReportingOrchestrator (SimpleGraph)
```

## Run

```bash
uv run python -m examples.irs_reporting.hierarchy
```

Uses a rule-based model (no LLM API keys). The matching stage processes 200 synthetic
bank positions by default; tests use a smaller count for speed.

## Key patterns

- **TodoGraph** gates `report_to_supervisor` / `finish_task` until all TODO items are done.
- **MatchingOrchestrator** seeds one TODO per bank position so context stays flat while
  progress is tracked in `todo_list`.
- **Artifact isolation** — `clear_messages=True` on every subagent boundary; only
  `pipeline_artifact` merges back to the parent.
