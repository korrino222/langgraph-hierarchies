# langgraph-hierarchies

**Status: early development (0.0.x).** API and scope are not stable. v0.1 will ship the keystone slice below.

Stateful, deeply-nested **compiled-subgraph** agent hierarchies on [LangGraph](https://github.com/langchain-ai/langgraph) — with declarative per-subchain state isolation and supervisor-controlled iteration safety.

Flat delegation (Deep Agents, supervisor handoffs) is a good starting point. This library targets the production wall past that: recursive hierarchies where subagents are **real compiled subgraphs**, with explicit clear/merge/discard policy across satte fields — not ephemeral `task`-tool isolation alone.

## How this relates to other libraries

| | [langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/) | [Deep Agents](https://pypi.org/project/deepagents/) | langgraph-hierarchies |
| --- | --- | --- | --- |
| Model | Supervisor → workers (handoff tools) | Harness + subagents (`task` / programmatic `task()`) | Class-as-factory graphs, phased compile, `SubchainPolicy` |
| Nesting | Multi-level supervisors | Fan-out / data recursion (partial); not stateful deep trees | Recursive compiled subgraphs + declarative state policy |
| Best for | Quick hierarchical routing | General long-horizon agents | Production-grade deep hierarchies with accountability |

Deep Agents already covers fan-out, parallel orchestration, and RLM-style recursion over data ([programmatic subagents](https://docs.langchain.com/oss/python/deepagents/programmatic-subagents), June 2026). This project does **not** try to replace that.

## Compatibility

| langgraph-hierarchies | langgraph |
| --- | --- |
| 0.0.x | 1.2.6 (pinned; see [releases](https://github.com/korrino222/langgraph-hierarchies/releases)) |

Regression tests will gate version bumps. A pair is listed here only once the test suite is green for it.

## Roadmap (v0.1)

- [ ] `BaseGraph` / `CompiledGraph` + phased compilation
- [ ] `SubchainPolicy` (entry snapshot, clear/merge/discard, exit restore)
- [ ] `ReactGraph` + iteration safety (per-agent limits, supervisor `task_iterations`, forced-exit report)
- [ ] Minimal runnable hierarchy example

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
