# langgraph-hierarchies

**Decomposable agent hierarchies for [LangGraph](https://github.com/langchain-ai/langgraph).**

**Status: early development (0.0.x).** API and scope are not stable. v0.1 will ship the keystone slice below.

An agentic system is **decomposable** when it can be split into finer agentic systems — each developed, run, and evaluated in isolation, with its own context — that still achieve the goal of the whole. [Decomposability in AI workflows](https://medium.com/@ishish222/decomposability-in-ai-workflows-what-it-is-and-why-you-want-it-c12c9a939565) explains why that matters: lean context per unit, benchmarkable seams, durable checkpoints, and improvements that stop colliding across stages.

**langgraph-hierarchies** is the library for building that on LangGraph: recursive hierarchies of **real compiled subgraphs**, with declarative per-subchain context isolation (`SubchainPolicy`), supervisor-controlled iteration budgets, and artifact handoff across boundaries — not one swelling monolith, not ephemeral `task`-tool isolation alone.

Flat delegation (supervisor handoffs, Deep Agents) is a good starting point. This library targets the production wall past that: invokable subagents that nest as deep as the problem needs while preserving streamability, durability, and resumability at every level.

## How this relates to other libraries

| | [langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/) | [Deep Agents](https://pypi.org/project/deepagents/) | langgraph-hierarchies |
| --- | --- | --- | --- |
| Model | Supervisor → workers (handoff tools) | Harness + subagents (`task` / programmatic `task()`) | Class-as-factory graphs, phased compile, `SubchainPolicy` |
| Context isolation | Shared parent history | Ephemeral subagent context | Declarative clear/merge/discard per subchain boundary |
| Nesting | Multi-level supervisors | Fan-out / data recursion (partial); not stateful deep trees | Recursive compiled subgraphs + explicit state policy |
| Best for | Quick hierarchical routing | General long-horizon agents | Production decomposable hierarchies — benchmarkable units, lean checkpoints |

Deep Agents already covers fan-out, parallel orchestration, and RLM-style recursion over data ([programmatic subagents](https://docs.langchain.com/oss/python/deepagents/programmatic-subagents), June 2026). This project does **not** try to replace that.

See [`examples/irs_reporting/`](examples/irs_reporting/) for a five-stage pipeline with artifact-only handoff between compiled subgraphs — the IRS workflow from the article, decomposed.

## Compatibility

| langgraph-hierarchies | langgraph |
| --- | --- |
| 0.0.x | 1.2.6 (pinned; see [releases](https://github.com/korrino222/langgraph-hierarchies/releases)) |

Regression tests gate version bumps. A pair is listed here only once the full suite is green for it.

### Bumping the langgraph pin

1. Update the `langgraph` constraint in `pyproject.toml`.
2. Refresh the lockfile: `uv sync`.
3. Run the full suite: `uv run pytest`.
4. To isolate a failing story, run e.g. `uv run pytest -m us04`.
5. When the suite is green, update the compatibility matrix above and cut a release.

CI runs `ruff` and `pytest` on every push/PR (Python 3.10–3.13). Tests use scripted models only — no LLM API keys required.

## What's in 0.0.2

The mechanics behind decomposable hierarchies:

- `BaseGraph` / `CompiledGraph` + phased compilation — each unit is a real invokable subgraph
- `SubchainPolicy` — entry snapshot, clear/merge/discard, exit restore; isolate context at every seam
- `ReactGraph` + iteration safety — per-agent limits, supervisor `task_iterations`, forced-exit report
- Root compile (`compile_as_root`) and unified invocation — stream and checkpoint through the full tree
- Compatibility harness (per-story pytest markers, CI) — benchmark units in isolation
- `TodoGraph` + todo toolkit — batch processing with flat context; IRS hierarchy example (`examples/irs_reporting/`)

Planner/Executor, progress tracking, HITL: follow-on after v0.1.

## Install

```bash
pip install langgraph-hierarchies
```

Requires Python ≥3.10 (matches [LangGraph](https://pypi.org/project/langgraph/)).

### Development

Uses [uv](https://docs.astral.sh/uv/) (same toolchain as upstream LangGraph):

```bash
uv sync
uv run pytest
uv build   # sdist + wheel for PyPI
```

## Disclaimer

This project is **not affiliated with, endorsed by, or maintained by** LangChain or the LangGraph team. “LangGraph” is a trademark of LangChain; this package name indicates compatibility with the LangGraph runtime, not official status.

## Contact

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/korrino222/langgraph-hierarchies/issues)

## License

MIT — see [LICENSE](https://github.com/korrino222/langgraph-hierarchies/blob/main/LICENSE).
