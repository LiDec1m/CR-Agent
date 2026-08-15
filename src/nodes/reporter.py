"""Reporter node: build the final RiskReport from the agent state.

Pure state aggregation — no LLM calls, no side effects. Split out of
ReflectionNode so that:

- Reflection only decides *whether* to loop (coverage assessment),
- Reporter owns *finalization* (report construction).

Structural guarantee: the conditional edge after reflection routes to
``reporter`` on END instead of a bare END, so the graph can never finish
without producing a report — the class of bug where reflection returned
``needs_more_analysis=True`` at the round cap (and the router went to END
anyway, skipping report generation) is now impossible by construction.
"""

from __future__ import annotations

from src.models import AgentPhase, AgentState, RiskReport


class ReporterNode:
    """Assemble the final RiskReport from the current state."""

    def __call__(self, state: AgentState) -> dict:
        report = self._build_report(state)
        return {
            "phase": AgentPhase.DONE,
            "needs_more_analysis": False,
            "report": report,
        }

    @staticmethod
    def _build_report(state: AgentState) -> RiskReport:
        """Build the final report entirely from the current agent state."""
        files_scanned = list(dict.fromkeys(hunk.file_path for hunk in state.hunks))
        overall_risk_score = max(
            (risk.risk_score for risk in state.risks), default=0.0
        )
        summary = (
            f"Detected {len(state.risks)} risk(s)."
            if state.risks
            else "No significant risks detected."
        )
        return RiskReport(
            repo=state.repo,
            commit_sha=state.commit_sha,
            summary=summary,
            risks=state.risks,
            overall_risk_score=overall_risk_score,
            files_scanned=files_scanned,
            total_hunks=len(state.hunks),
            rules_executed=state.rules_executed,
            reflection_rounds=state.reflection_round,
            long_term_feedback_applied=state.long_term_feedback,
        )
