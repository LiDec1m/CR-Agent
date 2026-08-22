"""Core data models: HunkInfo, Evidence, RiskItem, RiskReport, AgentState."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .enums import AgentPhase, ChangeType, RiskCategory, Severity


class DiffLine(BaseModel):
    content: str
    change_type: ChangeType
    old_line_no: Optional[int] = None
    new_line_no: Optional[int] = None


class HunkInfo(BaseModel):
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section_header: str = ""
    lines: list[DiffLine] = Field(default_factory=list)

    @property
    def added_lines(self) -> list[DiffLine]:
        return [l for l in self.lines if l.change_type == ChangeType.ADDED]

    @property
    def removed_lines(self) -> list[DiffLine]:
        return [l for l in self.lines if l.change_type == ChangeType.REMOVED]

    @property
    def added_code(self) -> str:
        return "\n".join(l.content for l in self.added_lines)

    @property
    def language(self) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".tsx": "typescript", ".jsx": "javascript",
            ".java": "java", ".go": "go", ".rs": "rust",
            ".cpp": "cpp", ".c": "c",
        }
        _, ext = os.path.splitext(self.file_path)
        return ext_map.get(ext, "unknown")


class Evidence(BaseModel):
    source: str
    rule_id: Optional[str] = None
    category: RiskCategory
    severity: Severity
    message: str
    line_range: Optional[tuple[int, int]] = None
    snippet: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_type: str = "deterministic"
    # File the evidence was found in. Attached centrally by ToolRouter
    # (rules only see one hunk at a time; Judge needs file attribution
    # to map evidence to the right file in multi-file diffs).
    file_path: Optional[str] = None


class RiskItem(BaseModel):
    title: str
    category: RiskCategory
    severity: Severity
    description: str
    evidence_chain: list[Evidence] = Field(default_factory=list)
    suggestion: Optional[str] = None
    file_path: Optional[str] = None
    line_range: Optional[tuple[int, int]] = None
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RuleOutcomeStatus(str, Enum):
    """Credibility of one rule execution against one diff hunk."""

    EVIDENCE_PRODUCED = "evidence_produced"
    CLEAN = "clean"
    DEGRADED = "degraded"
    FAILED = "failed"


class RuleOutcome(BaseModel):
    """Auditable outcome of running one rule on one hunk.

    Unlike Evidence, this records successful negative checks and execution
    failures too, so Reflection can distinguish a clean hunk from one that
    was never conclusively assessed.
    """

    hunk_key: str
    rule: str
    status: RuleOutcomeStatus
    detail: Optional[str] = None


class DismissedEvidence(BaseModel):
    """An evidence item that the Judge rejected as false-positive or
    irrelevant, with a human-readable reason.

    Making the dismissal explicit (rather than silently dropping the
    evidence) keeps the rejection auditable: a reviewer can see *why*
    the judge dismissed something and flag false-negative if the
    reasoning is wrong.
    """

    evidence: Evidence
    reason: str


class RiskReport(BaseModel):
    repo: str = ""
    commit_sha: Optional[str] = None
    summary: str = ""
    risks: list[RiskItem] = Field(default_factory=list)
    dismissed_evidence: list[DismissedEvidence] = Field(default_factory=list)
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    files_scanned: list[str] = Field(default_factory=list)
    total_hunks: int = 0
    rules_executed: list[str] = Field(default_factory=list)
    coverage_limited_hunks: int = 0
    reflection_rounds: int = 0

    def to_text(self) -> str:
        lines = [
            "=== Code Change Risk Report ===",
            f"Repository: {self.repo}",
            f"Commit: {self.commit_sha or 'N/A'}",
            f"Files scanned: {len(self.files_scanned)}",
            f"Total hunks: {self.total_hunks}",
            f"Conclusive coverage: {self.total_hunks - self.coverage_limited_hunks}/{self.total_hunks} hunks",
            f"Coverage-limited hunks: {self.coverage_limited_hunks}",
            f"Overall risk score: {self.overall_risk_score:.2f}",
            f"Reflection rounds: {self.reflection_rounds}",
            "",
            f"Summary: {self.summary}",
            "",
        ]
        if not self.risks:
            lines.append("✅ No significant risks detected.")
        else:
            lines.append(f"Detected {len(self.risks)} risk(s):")
            lines.append("")
            for i, r in enumerate(self.risks, 1):
                lines.append(f"  {i}. [{r.severity.value.upper()}] {r.title}")
                lines.append(f"     Category: {r.category.value}")
                lines.append(f"     Score: {r.risk_score:.2f}")
                if r.file_path:
                    loc = r.file_path
                    if r.line_range:
                        loc += f":{r.line_range[0]}-{r.line_range[1]}"
                    lines.append(f"     Location: {loc}")
                lines.append(f"     Description: {r.description}")
                lines.append(f"     Evidence chain ({len(r.evidence_chain)} item(s)):")
                for ev in r.evidence_chain:
                    lines.append(
                        f"       - [{ev.source_type}] {ev.source} "
                        f"({ev.severity.value}, conf={ev.confidence:.2f}): "
                        f"{ev.message}"
                    )
                if r.suggestion:
                    lines.append(f"     Suggestion: {r.suggestion}")
                lines.append("")
        return "\n".join(lines)


class AgentState(BaseModel):
    repo: str = ""
    commit_sha: Optional[str] = None
    raw_diff: str = ""
    hunks: list[HunkInfo] = Field(default_factory=list)
    # Queue for the CURRENT ToolRouter round, keyed by ``file_path:new_start``.
    # Planner and Reflection write it; ToolRouter consumes and clears it.
    pending_tools_by_hunk: dict[str, list[str]] = Field(default_factory=dict)
    # Why each hunk was scheduled as it was, keyed by ``file_path:new_start``.
    # Written once by Planner; consumed by Reflection's coverage digest so
    # the scheduling rationale stays auditable across rounds.
    planning_reasons: dict[str, str] = Field(default_factory=dict)
    # One durable outcome per rule execution, including clean, degraded and
    # failed checks. Replaced as a whole by ToolRouter to avoid duplicates.
    # This is the ONLY execution-history source: which rules conclusively
    # ran where is derived from it (CLEAN / EVIDENCE_PRODUCED entries).
    rule_outcomes: list[RuleOutcome] = Field(default_factory=list)
    phase: AgentPhase = AgentPhase.PLANNING
    evidence_pool: list[Evidence] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    reflection_round: int = 0
    reflection_notes: list[str] = Field(default_factory=list)
    dismissed_evidence: list[DismissedEvidence] = Field(default_factory=list)
    needs_more_analysis: bool = False
    rag_context: dict = Field(default_factory=dict)
    report: Optional[RiskReport] = None

    model_config = {"arbitrary_types_allowed": True}
