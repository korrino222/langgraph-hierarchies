"""Shared value types."""

from pydantic import BaseModel


class Progress(BaseModel):
    """Tracks scheduled and finished executions for one agent slot."""

    scheduled_executions: int = 0
    finished_executions: int = 0
