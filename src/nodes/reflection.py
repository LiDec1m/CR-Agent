"""Reflection node: assess hunk-level coverage and schedule targeted checks."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.models import AgentPhase, AgentState, RuleOutcomeStatus
from src.nodes.tool_router import hunk_key
from src.rules import registry


class ReflectionNode:
    """Decide whether uncovered hunks need additional rule execution."""

    def __init__(
        self, llm: LLMClient, max_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds

    def _build_coverage_digest(self, state: AgentState) -> str:
        """Summarize reliable, degraded and absent hunk-level assessments.

        Each hunk line also surfaces the Planner's reason for the rules
        originally chosen there, so the LLM can judge whether a
        re-analysis target is genuinely uncovered or already covered
        by design.
        """
        if not state.hunks:
            return ""
        by_hunk: dict[str, list] = {}
        for outcome in state.rule_outcomes:
            by_hunk.setdefault(outcome.hunk_key, []).append(outcome)

        lines: list[str] = []
        for hunk in state.hunks:
            key = hunk_key(hunk)
            outcomes = by_hunk.get(key, [])
            n_evidence = sum(
                1 for evidence in state.evidence_pool
                if evidence.file_path == hunk.file_path
            )
            n_risks = sum(
                1 for risk in state.risks if risk.file_path == hunk.file_path
            )
            statuses = ", ".join(
                f"{outcome.rule}={outcome.status.value}"
                for outcome in outcomes
            ) or "unexamined"
            reason = state.planning_reasons.get(key, "")
            planned = f"; planned: {reason}" if reason else ""
            lines.append(
                f"- {key} (+{hunk.new_count}/-{hunk.old_count}): "
                f"{n_evidence} evidences, {n_risks} risks; checks: {statuses}"
                f"{planned}"
            )
        return "Per-hunk assessment coverage:\n" + "\n".join(lines)

    def __call__(self, state: AgentState) -> dict:
        new_round = state.reflection_round + 1
        if new_round > self.max_rounds:
            return {
                "needs_more_analysis": False, "phase": AgentPhase.REPORTING,
                "reflection_round": new_round,
            }

        prompt = (
            "Review per-hunk analysis coverage and decide whether targeted rules "
            "are required. Return JSON only: {\"needs_more_analysis\": bool, "
            "\"additional_tools_by_hunk\": {\"file_path:new_start\": [str]}, "
            "\"reason\": str}.\n\n"
            "Interpretation: clean and evidence_produced are conclusive completed "
            "checks. degraded means a rule attempted but did not yield a reliable "
            "conclusion and may be retried. failed means the rule is broken and must "
            "not be retried. unexamined means no rule completed for that hunk. "
            "Suggest a rule only for supplied hunk keys.\n\n"
            f"Risks: {json.dumps([risk.model_dump(mode='json') for risk in state.risks])}\n"
            f"Available rules: {json.dumps(registry.list_all())}\n\n"
            f"{self._build_coverage_digest(state)}"
        )
        try:
            response = self.llm.chat_json("You are a code-risk reviewer.", prompt)
            needs_more = bool(response.get("needs_more_analysis", False)) if response else False
            raw_suggestions = response.get("additional_tools_by_hunk", {}) if response else {}
            reason = response.get("reason", "") if response else ""
        except Exception:
            needs_more, raw_suggestions = False, {}
            reason = "Unable to parse reflection response."

        pending = self._new_assignments(state, raw_suggestions)
        if needs_more and not pending:
            needs_more = False
            reason = (
                "Finalizing: suggested rules have already completed conclusively, "
                "failed previously, are unknown, or target unknown hunks. Last "
                f"assessment: {reason}"
            )
        note = f"Round {new_round}: {reason}"
        return {
            "needs_more_analysis": needs_more,
            "pending_tools_by_hunk": pending if needs_more else {},
            "phase": AgentPhase.TOOL_ROUTING if needs_more else AgentPhase.REPORTING,
            "reflection_round": new_round,
            "reflection_notes": [note],
        }

    @staticmethod
    def _new_assignments(
        state: AgentState, raw: object,
    ) -> dict[str, list[str]]:
        """Keep only valid hunk-rule work not already conclusive or failed.

        A degraded assignment remains retryable because it did not establish a
        reliable result. The retry budget is the graph's reflection-round cap.
        """
        if not isinstance(raw, dict):
            return {}
        valid_keys = {hunk_key(hunk) for hunk in state.hunks}
        available = set(registry.list_all())
        outcome_status = {
            (outcome.hunk_key, outcome.rule): outcome.status
            for outcome in state.rule_outcomes
        }
        pending: dict[str, list[str]] = {}
        for key, rules in raw.items():
            if not isinstance(key, str) or key not in valid_keys or not isinstance(rules, list):
                continue
            accepted: list[str] = []
            for rule in rules:
                if not isinstance(rule, str) or rule not in available:
                    continue
                status = outcome_status.get((key, rule))
                if status in {
                    RuleOutcomeStatus.CLEAN,
                    RuleOutcomeStatus.EVIDENCE_PRODUCED,
                    RuleOutcomeStatus.FAILED,
                }:
                    continue
                if rule not in accepted:
                    accepted.append(rule)
            if accepted:
                pending[key] = accepted
        return pending
