"""RunnableConfig helpers for LangSmith thread grouping."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

THREAD_METADATA_KEYS = ("thread_id", "session_id")
CONFIGURABLE_THREAD_KEY = "thread_id"


def _is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _coerce_thread_id(value: Any) -> str | None:
    if not _is_set(value):
        return None
    return str(value).strip()


def ensure_runnable_config_dict(
    config: RunnableConfig | dict | None,
) -> dict:
    """Return a shallow copy of config as a mutable dict."""
    return dict(config or {})


def resolve_thread_id(
    config: RunnableConfig | dict | None,
    context: Any = None,
) -> str | None:
    """Resolve the effective thread ID using first non-empty wins."""
    cfg = config or {}
    metadata = cfg.get("metadata") or {}
    configurable = cfg.get("configurable") or {}

    if _is_set(metadata.get("thread_id")) and _is_set(metadata.get("session_id")):
        thread_id = _coerce_thread_id(metadata.get("thread_id"))
        session_id = _coerce_thread_id(metadata.get("session_id"))
        if thread_id is not None and session_id is not None and thread_id != session_id:
            logger.debug(
                "Conflicting thread metadata: metadata.thread_id=%r and "
                "metadata.session_id=%r differ; leaving caller values unchanged",
                thread_id,
                session_id,
            )

    for source in (
        metadata.get("thread_id"),
        metadata.get("session_id"),
        configurable.get("thread_id"),
        getattr(context, "thread_id", None) if context is not None else None,
    ):
        tid = _coerce_thread_id(source)
        if tid is not None:
            return tid
    return None


def normalize_thread_config(
    config: RunnableConfig | dict | None,
    context: Any = None,
) -> RunnableConfig:
    """Backfill thread metadata and configurable slots from any available source."""
    cfg = ensure_runnable_config_dict(config)
    tid = resolve_thread_id(cfg, context)
    if tid is None:
        return cfg

    metadata = dict(cfg.get("metadata") or {})
    configurable = dict(cfg.get("configurable") or {})

    if not _is_set(metadata.get("thread_id")):
        metadata["thread_id"] = tid
    if not _is_set(metadata.get("session_id")):
        metadata["session_id"] = tid
    if not _is_set(configurable.get("thread_id")):
        configurable["thread_id"] = tid

    cfg["metadata"] = metadata
    cfg["configurable"] = configurable
    return cfg


def build_invoke_config(
    *,
    thread_id: str,
    run_name: str | None = None,
    tags: list[str] | None = None,
    recursion_limit: int = 50,
    **extra: Any,
) -> RunnableConfig:
    """Return a RunnableConfig pre-populated for LangSmith threads and checkpointing."""
    config: dict[str, Any] = {
        "recursion_limit": recursion_limit,
        **extra,
    }
    if run_name is not None:
        config["run_name"] = run_name
    if tags is not None:
        config["tags"] = tags
    return normalize_thread_config(
        {
            **config,
            "configurable": {"thread_id": thread_id},
        }
    )
