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
4. To isolate a failing story, run e.g. `uv run pytest -m us04` (see test markers below).
5. When the suite is green, update the compatibility matrix above, the [docs compatibility page](docs/compatibility.mdx), and cut a release.

CI runs `ruff` and `pytest` on every push/PR (Python 3.10–3.13). Tests use scripted models only — no LLM API keys required.

### Test markers

Stories map to pytest markers for targeted regression:

| Marker | Scope |
| --- | --- |
| `us01` | Foundational layer |
| `us02` | Subagent state isolation |
| `us03` | Iteration budget enforcement |
| `us04` | Root compile and unified invocation |
| `us05` | LangGraph compatibility harness |
| `us06` | IRS multi-stage hierarchy |
| `us07` | Responsibility boundary violations |

## Changelog

- **0.0.8** — lean sdist: exclude `docs-site/`, `docs/`, `tests/`, and config dirs; sdist drops ~55 MB → ~120 KB
- **0.0.5–0.0.7** — parallel flat-tool calls in `ReactGraph`; managed-channel stripping in `Send` payloads; concepts docs; `computation_node` config-annotation warning fix
- **0.0.4** — **breaking rename** `SubchainPolicy` → `SubagentPolicy` (`subchain_policy` → `subagent_policy`, `__subchain_stack__` → `__subagent_stack__`; no aliases — update imports and checkpoint state keys)
- **0.0.3** — LangSmith threads: automatic `thread_id` normalization into `RunnableConfig` metadata at invoke time ([see below](#langsmith-threads))
- **0.0.2** — foundational mechanics for decomposable hierarchies: `BaseGraph`/`CompiledGraph` phased compilation, `SubagentPolicy` context isolation, `ReactGraph` iteration safety, root compile (`compile_as_root`) + unified invocation, compatibility harness, `TodoGraph` + todo toolkit (`tests/irs_hierarchy/`)

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
uv run pre-commit install   # optional: auto-run ruff before each commit
uv build   # sdist + wheel for PyPI
```

Lint/format (also enforced in CI):

```bash
uv run ruff check .
uv run ruff format --check .
```

## Disclaimer

This project is **not affiliated with, endorsed by, or maintained by** LangChain or the LangGraph team. “LangGraph” is a trademark of LangChain; this package name indicates compatibility with the LangGraph runtime, not official status.

## Contact

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/korrino222/langgraph-hierarchies/issues)

## License

MIT — see [LICENSE](https://github.com/korrino222/langgraph-hierarchies/blob/main/LICENSE).
