"""Reflection node: decide whether more analysis rounds are needed.

Pure decision node — it assesses coverage (with cross-file feedback from
long-term memory) and returns ``needs_more_analysis`` plus any additional
rules to run. It never builds the report: that responsibility belongs to
ReporterNode, which the graph always routes through on the way to END.
"""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
from src.models import AgentPhase, AgentState
from src.rules import registry


class ReflectionNode:
    """Decide whether more rules are needed or the analysis is complete."""

    def __init__(
        self,
        llm: LLMClient,
        ltm: LongTermMemory | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.ltm = ltm
        self.max_rounds = max_rounds

    def __call__(self, state: AgentState) -> dict:
        new_round = state.reflection_round + 1
        if new_round > self.max_rounds:
            return {
                "needs_more_analysis": False,
                "phase": AgentPhase.DONE,
                "reflection_round": new_round,
            }

        # Fetch cross-file feedback for rules that were ACTUALLY triggered
        # in this analysis. This tells the LLM: "the same rules that fired
        # here were marked as false_positive / missing / severity_adjust
        # in other files" — useful context for deciding whether to run
        # more rules or finalize.
        cross_file_feedback: list[str] = []
        if self.ltm and state.rules_executed:
            diff_files = {hunk.file_path for hunk in state.hunks}
            try:
                raw_feedback = self.ltm.get_feedback_by_rule_across_files(
                    state.rules_executed,
                )
                for item in raw_feedback:
                    file_pat = item.get("file_pattern", "?")
                    # Skip feedback from the current diff files (already
                    # covered by Planner's file-specific feedback)
                    if any(file_pat.startswith(fp) or fp.startswith(file_pat)
                           for fp in diff_files):
                        continue
                    rule_id = item.get("rule_id") or "general"
                    fb_type = item.get("feedback_type", "")
                    content = item.get("feedback_content", str(item))
                    cross_file_feedback.append(
                        f"[{rule_id}/{fb_type}] (from {file_pat}): {content}"
                    )
            except Exception:
                pass

        prompt = (
            "Review analysis coverage and determine whether more deterministic "
            "rules are required. Return JSON only: {\"needs_more_analysis\": bool, "
            "\"additional_tools_needed\": [str], \"reason\": str, "
            "\"coverage_assessment\": str}.\n\n"
            "Consider: are there risk patterns in the code that the executed "
            "rules did NOT cover? (e.g. logic errors, race conditions, "
            "resource leaks, missing input validation). If so, you can "
            "suggest \"llm_assisted\" in additional_tools_needed to trigger "
            "LLM-based analysis for risks that deterministic rules miss.\n\n"
            f"Executed rules: {json.dumps(state.rules_executed)}\n"
            f"Risks: {json.dumps([risk.model_dump(mode='json') for risk in state.risks])}\n"
            f"Available rules: {json.dumps(registry.list_all())}"
        )
        if cross_file_feedback:
            prompt += (
                f"\n\nCross-file feedback (same rules were flagged in "
                f"other files):\n{json.dumps(cross_file_feedback)}"
            )
        try:
            response = self.llm.chat_json(
                "You are a code-risk reviewer.", prompt
            )
            needs_more_analysis = bool(
                response.get("needs_more_analysis", False)
            ) if response else False
            additional_tools_needed = (
                response.get("additional_tools_needed", []) if response else []
            )
            reason = response.get("reason", "") if response else ""
            coverage_assessment = (
                response.get("coverage_assessment", "") if response else ""
            )
        except Exception:
            needs_more_analysis = False
            additional_tools_needed = []
            reason = "Unable to parse reflection response."
            coverage_assessment = "unknown"

        # Boundary guard: this is the last allowed round. Even if the LLM
        # wants more analysis, the routing condition
        # (reflection_round < max_rounds) will not send us back to
        # tool_router — so report the intention to stop instead of
        # requesting a loop that cannot happen.
        if new_round >= self.max_rounds and needs_more_analysis:
            needs_more_analysis = False
            reason = (
                f"Max reflection rounds ({self.max_rounds}) reached; "
                f"finalizing. Last assessment: {reason}"
            )

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
            }
        return {
            "needs_more_analysis": True,
            "additional_tools_needed": additional_tools_needed,
            "phase": AgentPhase.TOOL_ROUTING,
            "reflection_round": new_round,
            "reflection_notes": notes,
        }
