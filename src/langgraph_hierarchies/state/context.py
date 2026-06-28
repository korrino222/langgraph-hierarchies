"""Runtime context schema for hierarchy graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel


@dataclass
class BaseContext:
    """Immutable runtime configuration injected via LangGraph Runtime.context."""

    thread_id: str = ""
    created_at: datetime | None = None
    debug: bool = False
    model: BaseChatModel | None = None
    model_visual: BaseChatModel | None = None
    root_agent: Any | None = None
    file_store: Any | None = None
