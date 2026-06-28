# Minimal hierarchy example

Demonstrates US-04 root compilation and unified invocation:

- `compile_as_root()` for the top-level orchestrator
- `context=` injection via `BaseContext(model=...)`
- A child `ReactGraph` wired through `compiled_subgraphs`
- Automatic `ToolMessage` generation when the child reports upward

## Run

From the repository root:

```bash
uv run python -m examples.minimal.hierarchy
```

Expected output:

```text
orchestrator complete
```

No LLM API key is required; the example uses a deterministic scripted model.
