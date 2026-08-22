"""Enumerations used across the agent."""

from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    CONTEXT = "context"


class RiskCategory(str, Enum):
    SECURITY = "security"
    COMPLEXITY = "complexity"
    BUG_RISK = "bug_risk"
    STYLE = "style"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"


class AgentPhase(str, Enum):
    PLANNING = "planning"
    TOOL_ROUTING = "tool_routing"
    JUDGING = "judging"
    REFLECTING = "reflecting"
    REPORTING = "reporting"
    DONE = "done"
