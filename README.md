# langgraph-hierarchies

**Decomposable agent hierarchies for [LangGraph](https://github.com/langchain-ai/langgraph).**

**Documentation:** [https://korrino222.github.io/langgraph-hierarchies/](https://korrino222.github.io/langgraph-hierarchies/)

**Status: early development (0.0.x).** API and scope are not stable. v0.1 will ship the keystone slice below.

An agentic system is **decomposable** when it can be split into finer agentic systems — each developed, run, and evaluated in isolation, with its own context — that still achieve the goal of the whole. [Decomposability in AI workflows](https://medium.com/@ishish222/decomposability-in-ai-workflows-what-it-is-and-why-you-want-it-c12c9a939565) explains why that matters: lean context per unit, benchmarkable seams, durable checkpoints, and improvements that stop colliding across stages.

**langgraph-hierarchies** is the library for building that on LangGraph: recursive hierarchies of **real compiled subgraphs**, with declarative per-subagent context isolation (`SubagentPolicy`), supervisor-controlled iteration budgets, and artifact handoff across boundaries — not one swelling monolith, not ephemeral `task`-tool isolation alone.

Flat delegation (supervisor handoffs, Deep Agents) is a good starting point. This library targets the production wall past that: invokable subagents that nest as deep as the problem needs while preserving streamability, durability, and resumability at every level.

## How this relates to other libraries

| | [langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/) | [Deep Agents](https://pypi.org/project/deepagents/) | langgraph-hierarchies |
| --- | --- | --- | --- |
| Model | Supervisor → workers (handoff tools) | Harness + subagents (`task` / programmatic `task()`) | Class-as-factory graphs, phased compile, `SubagentPolicy` |
| Context isolation | Shared parent history | Ephemeral subagent context | Declarative clear/merge/discard per subagent boundary |
| Nesting | Multi-level supervisors | Fan-out / data recursion (partial); not stateful deep trees | Recursive compiled subgraphs + explicit state policy |
| Best for | Quick hierarchical routing | General long-horizon agents | Production decomposable hierarchies — benchmarkable units, lean checkpoints |

Deep Agents already covers fan-out, parallel orchestration, and RLM-style recursion over data ([programmatic subagents](https://docs.langchain.com/oss/python/deepagents/programmatic-subagents), June 2026). This project does **not** try to replace that.

See [langgraph-hierarchies-examples](https://github.com/korrino222/langgraph-hierarchies-examples) for worked examples demonstrating the before/after decomposability arc.

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

## What's in 0.0.7

- **computation_node config annotation** — fix `UserWarning` emitted by LangGraph when `SimpleGraph` subclasses use `from __future__ import annotations`; `config` is now typed as `RunnableConfig` (always provided at runtime)

## What's in 0.0.6

- **Managed state in Send payloads** — `ReactGraph` strips LangGraph-managed channels (e.g. `remaining_steps`) from `Send` state to avoid pregel warnings
- **Concepts docs** — Mintlify pages for state, graph factories, compilation, `CompiledGraph`, and subagents

## What's in 0.0.5

- **Parallel tool calls** — `ReactGraph` dispatches multiple flat-tool calls in parallel; parallel batches that include a subagent are blocked with one error `ToolMessage` per `tool_call_id` so chat history stays valid for the next LLM turn

## What's in 0.0.4

- **Breaking rename** — `SubchainPolicy` → `SubagentPolicy`, `subchain_policy` → `subagent_policy`, `__subchain_stack__` → `__subagent_stack__` (no aliases; update imports and checkpoint state keys)

## What's in 0.0.3

- **LangSmith threads** — automatic `thread_id` normalization into `RunnableConfig` metadata at invoke time (see below)

## What's in 0.0.2

The mechanics behind decomposable hierarchies:

- `BaseGraph` / `CompiledGraph` + phased compilation — each unit is a real invokable subgraph
- `SubagentPolicy` — entry snapshot, clear/merge/discard, exit restore; isolate context at every seam
- `ReactGraph` + iteration safety — per-agent limits, supervisor `task_iterations`, forced-exit report
- Root compile (`compile_as_root`) and unified invocation — stream and checkpoint through the full tree
- Compatibility harness (per-story pytest markers, CI) — benchmark units in isolation
- `TodoGraph` + todo toolkit — batch processing with flat context; IRS hierarchy test fixture (`tests/irs_hierarchy/`)

Planner/Executor, progress tracking, HITL: follow-on after v0.1.

### LangSmith threads

LangSmith groups multi-turn conversations when runs share `metadata.thread_id`
(or `session_id`). The library normalizes thread IDs automatically at invoke time.
Provide **any one** of:

- `config["configurable"]["thread_id"]` — LangGraph checkpoint convention
- `config["metadata"]["thread_id"]` — explicit LangSmith thread
- `context.thread_id` on your context dataclass (e.g. `BaseContext`)

All three are backfilled when missing. Explicit `metadata.thread_id` is never overwritten.

```python
from langgraph_hierarchies import BaseContext, build_invoke_config

config = build_invoke_config(thread_id="demo-run-1", recursion_limit=200)
result = root.invoke(state, config=config, context=BaseContext(model=model))
```

With `LANGCHAIN_TRACING_V2=true`, open your project → **Threads** tab and filter by the thread ID.
Child subgraph spans inherit the same config via LangGraph propagation.

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
