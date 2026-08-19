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

    def _build_coverage_digest(self, state: AgentState) -> str:
        """Process-level coverage signals for the reflection prompt.

        Summary only, never raw material: per-hunk evidence/risk density
        (zero-evidence hunks are explicit coverage gaps) plus failed-rule
        signals (error evidence means that rule's detection scope was a
        blind spot this round). Feeding material (diff text, codebase
        context, full evidence) here would drift this node into detection.
        """
        if not state.hunks:
            return ""
        lines: list[str] = []
        for hunk in state.hunks:
            n_ev = sum(
                1 for e in state.evidence_pool
                if e.file_path == hunk.file_path
            )
            n_risk = sum(
                1 for r in state.risks
                if r.file_path == hunk.file_path
            )
            lines.append(
                f"- {hunk.file_path} hunk@{hunk.new_start} "
                f"(+{hunk.new_count}/-{hunk.old_count}): "
                f"{n_ev} evidences, {n_risk} risks"
            )
        digest = "Per-hunk coverage:\n" + "\n".join(lines)

        failed = [
            f"{e.source} on {e.file_path}"
            for e in state.evidence_pool
            if e.source_type == "error"
        ]
        if failed:
            digest += "\nFailed rules (blind spots this round): " + ", ".join(failed)
        return digest

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
        digest = self._build_coverage_digest(state)
        if digest:
            prompt += (
                "\n\n" + digest + "\n"
                "Hunks with 0 evidences are potential coverage gaps: "
                "consider whether an available rule (or llm_assisted) "
                "should have produced signals there."
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

        # Anti-idle-loop validation: a round-trip to tool_router is only
        # useful if it will execute at least one rule that has NOT run
        # yet (valid name + not already executed). Otherwise the next
        # round would collect zero new evidence and burn an LLM round
        # for nothing — finalize instead.
        if needs_more_analysis:
            available = set(registry.list_all())
            new_tools = [
                t for t in (additional_tools_needed or [])
                if t in available and t not in state.rules_executed
            ]
            if not new_tools:
                needs_more_analysis = False
                additional_tools_needed = []
                reason = (
                    f"Finalizing: suggested tools contain no new rules to "
                    f"execute (already run or unknown). Last assessment: "
                    f"{reason}"
                )
            else:
                additional_tools_needed = new_tools
                # NOTE: at the round cap (new_round >= max_rounds) the LLM's
                # true verdict is preserved (needs_more_analysis stays
                # True). The routing condition (reflection_round <
                # max_rounds) still sends the graph to reporter, and the
                # terminal needs_more_analysis=True is itself the
                # observability signal that this diff was under-analysed
                # at the cap.

        # NOTE: GraphState.reflection_notes uses operator.add (delta
        # accumulation), so we must return ONLY the new note here —
        # returning the full list would duplicate every earlier note
        # on each round.
        new_note = f"Round {new_round}: {coverage_assessment}; {reason}"
        if not needs_more_analysis:
            return {
                "needs_more_analysis": False,
                "pending_tools": [],
                "phase": AgentPhase.DONE,
                "reflection_round": new_round,
                "reflection_notes": [new_note],
            }
        # Wire key "additional_tools_needed" (LLM JSON contract) maps
        # to the unified pending_tools queue consumed by ToolRouter.
        return {
            "needs_more_analysis": True,
            "pending_tools": additional_tools_needed,
            "phase": AgentPhase.TOOL_ROUTING,
            "reflection_round": new_round,
            "reflection_notes": [new_note],
        }
