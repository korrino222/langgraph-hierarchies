"""Hierarchy-specific exceptions."""


class HierarchyError(Exception):
    """Base exception for hierarchy errors."""


class AgentStuckError(HierarchyError):
    """Raised when an agent cannot proceed and reports a blocker."""

    def __init__(
        self,
        message: str,
        *,
        reporting_agent: str = "",
        subject_agent: str = "",
    ) -> None:
        super().__init__(message)
        self.reporting_agent = reporting_agent
        self.subject_agent = subject_agent


class InfrastructureBlocker(HierarchyError):
    """Raised for internal infrastructure failures that must not leak upstream."""
