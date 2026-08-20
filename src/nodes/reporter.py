"""Reporter node: build the final RiskReport from the agent state."""

from __future__ import annotations

from src.models import AgentPhase, AgentState, RiskReport, RuleOutcomeStatus
from src.nodes.tool_router import hunk_key


class ReporterNode:
    """Assemble the final RiskReport from the current state."""

    def __call__(self, state: AgentState) -> dict:
        return {"phase": AgentPhase.DONE, "report": self._build_report(state)}

    @staticmethod
    def _build_report(state: AgentState) -> RiskReport:
        """Build the report and derive global summaries from hunk-level state."""
        files_scanned = list(dict.fromkeys(hunk.file_path for hunk in state.hunks))
        rule_names = sorted({
            rule for rules in state.executed_tools_by_hunk.values() for rule in rules
        })
        outcomes = {(outcome.hunk_key, outcome.rule): outcome.status
                    for outcome in state.rule_outcomes}
        conclusive = 0
        limited = 0
        for hunk in state.hunks:
            statuses = [
                status for (key, _), status in outcomes.items()
                if key == hunk_key(hunk)
            ]
            if any(status in {
                RuleOutcomeStatus.CLEAN, RuleOutcomeStatus.EVIDENCE_PRODUCED,
            } for status in statuses):
                conclusive += 1
            else:
                limited += 1
        overall_risk_score = max((risk.risk_score for risk in state.risks), default=0.0)
        summary = (
            f"Detected {len(state.risks)} risk(s)."
            if state.risks else "No significant risks detected."
        )
        return RiskReport(
            repo=state.repo, commit_sha=state.commit_sha, summary=summary,
            risks=state.risks, dismissed_evidence=state.dismissed_evidence,
            overall_risk_score=overall_risk_score, files_scanned=files_scanned,
            total_hunks=len(state.hunks), rules_executed=rule_names,
            conclusively_examined_hunks=conclusive,
            coverage_limited_hunks=limited,
            reflection_rounds=state.reflection_round,
            long_term_feedback_applied=state.long_term_feedback,
        )
