"""LLM-driven planner that schedules deterministic rules per diff hunk."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.models import AgentPhase, AgentState, HunkInfo
from src.rag.retriever import RAGRetriever
from src.rules import registry
from src.nodes.tool_router import hunk_key


class PlannerNode:
    """Load historical-risk context and choose deterministic checks per hunk."""

    def __init__(self, llm: LLMClient, rag: RAGRetriever) -> None:
        self.llm = llm
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        history: list[dict] = []
        for hunk in state.hunks:
            try:
                history.extend(
                    self.rag.search_history(hunk.added_code, hunk.file_path)
                )
            except Exception:
                pass

        # llm_assisted is excluded from the initial plan by construction:
        # it never appears in the available list, and _normalize_plan only
        # accepts names from that list — no prompt-level prohibition needed.
        available = [rule for rule in registry.list_all() if rule != "llm_assisted"]
        changed = "\n\n".join(
            f"Hunk key: {hunk_key(hunk)}\nFile: {hunk.file_path}\n"
            f"Added code:\n{hunk.added_code or '(none)'}"
            for hunk in state.hunks
        )
        prompt = (
            "Analyze each diff hunk and select relevant deterministic analysis "
            "rules. Return JSON only: {\"plan_by_hunk\": "
            "{\"file_path:new_start\": {\"tools\": [str], \"reason\": str}}}. "
            "Every plan_by_hunk key must be one supplied Hunk key. \"tools\" "
            "must only contain rule names from the available rules list. "
            "\"reason\" is a short justification of why these tools were "
            "selected for this hunk.\n\n"
            f"Changed hunks:\n{changed}\n\n"
            f"Historical risks:\n{json.dumps(history)}\n\n"
            f"Available deterministic rules:\n{json.dumps(available)}"
        )
        try:
            response = self.llm.chat_json("You are a code risk planner.", prompt)
            raw_plan = response.get("plan_by_hunk", {}) if response else {}
        except Exception:
            raw_plan = {}

        valid_keys = {hunk_key(hunk) for hunk in state.hunks}
        pending, reasons = self._normalize_plan(
            raw_plan, valid_keys, set(available)
        )
        return {
            "pending_tools_by_hunk": pending,
            "planning_reasons": reasons,
            "phase": AgentPhase.TOOL_ROUTING,
            "rag_context": {"history": history},
        }

    @staticmethod
    def _normalize_plan(
        raw_plan: object, valid_keys: set[str], available: set[str],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Validate hunk assignments; return (plan, reasons) pair.

        Drops malformed entries, unknown hunk keys, non-string rule names,
        rules not in the available list, and duplicate assignments. The
        reason is kept as an empty string when the LLM did not provide one.
        """
        if not isinstance(raw_plan, dict):
            return {}, {}
        plan: dict[str, list[str]] = {}
        reasons: dict[str, str] = {}
        for key, entry in raw_plan.items():
            if not isinstance(key, str) or key not in valid_keys:
                continue
            if isinstance(entry, list):
                # Tolerate the legacy bare-list form.
                tools_raw: list = entry
                reason = ""
            elif isinstance(entry, dict):
                tools_raw = entry.get("tools", [])
                reason = entry.get("reason", "")
                if not isinstance(reason, str):
                    reason = ""
            else:
                continue
            if not isinstance(tools_raw, list):
                continue
            accepted = list(dict.fromkeys(
                rule for rule in tools_raw
                if isinstance(rule, str) and rule in available
            ))
            if accepted:
                plan[key] = accepted
                reasons[key] = reason
        return plan, reasons
