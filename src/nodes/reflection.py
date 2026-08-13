"""Reflection node and final-report builder."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.models import AgentPhase, AgentState, RiskReport
from src.rules import registry


class ReflectionNode:
    """Decide whether more rules are needed or finalize the risk report."""

    def __init__(self, llm: LLMClient, max_rounds: int = 3) -> None:
        self.llm = llm
        self.max_rounds = max_rounds

    def __call__(self, state: AgentState) -> dict:
        new_round = state.reflection_round + 1
        if new_round > self.max_rounds:
            return {
                "needs_more_analysis": False,
                "phase": AgentPhase.DONE,
                "reflection_round": new_round,
                "report": self._build_report(state, new_round),
            }

        prompt = (
            "Review analysis coverage and determine whether more deterministic "
            "rules are required. Return JSON only: {\"needs_more_analysis\": bool, "
            "\"additional_tools_needed\": [str], \"reason\": str, "
            "\"coverage_assessment\": str}.\n\n"
            f"Executed rules: {json.dumps(state.rules_executed)}\n"
            f"Risks: {json.dumps([risk.model_dump(mode='json') for risk in state.risks])}\n"
            f"Available rules: {json.dumps(registry.list_all())}"
        )
        try:
            response = json.loads(self.llm.chat("You are a code-risk reviewer.", prompt))
            needs_more_analysis = bool(response.get("needs_more_analysis", False))
            additional_tools_needed = response.get("additional_tools_needed", [])
            reason = response.get("reason", "")
            coverage_assessment = response.get("coverage_assessment", "")
        except Exception:
            needs_more_analysis = False
            additional_tools_needed = []
            reason = "Unable to parse reflection response."
            coverage_assessment = "unknown"

        notes = list(state.reflection_notes)
        notes.append(
            f"Round {new_round}: {coverage_assessment}; {reason}"
        )
        if not needs_more_analysis:
            return {
                "needs_more_analysis": False,
                "phase": AgentPhase.DONE,
                "reflection_round": new_round,
                "reflection_notes": notes,
                "report": self._build_report(state, new_round),
            }
        return {
            "needs_more_analysis": True,
            "additional_tools_needed": additional_tools_needed,
            "phase": AgentPhase.TOOL_ROUTING,
            "reflection_round": new_round,
            "reflection_notes": notes,
        }

    def _build_report(self, state: AgentState, reflection_round: int) -> RiskReport:
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
            reflection_rounds=reflection_round,
            long_term_feedback_applied=state.long_term_feedback,
        )
