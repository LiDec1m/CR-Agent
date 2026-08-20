"""LLM-driven planner that schedules deterministic rules per diff hunk."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
from src.models import AgentPhase, AgentState, HunkInfo
from src.rag.retriever import RAGRetriever
from src.rules import registry
from src.nodes.tool_router import hunk_key


class PlannerNode:
    """Load context and choose initial deterministic checks per hunk."""

    def __init__(self, llm: LLMClient, rag: RAGRetriever, ltm: LongTermMemory) -> None:
        self.llm = llm
        self.rag = rag
        self.ltm = ltm

    def __call__(self, state: AgentState) -> dict:
        feedback: list[str] = []
        history: list[dict] = []
        for hunk in state.hunks:
            feedback.extend(self._file_feedback(hunk))
            try:
                history.extend(self.rag.search_history(hunk.added_code, hunk.file_path))
            except Exception:
                pass

        available = [rule for rule in registry.list_all() if rule != "llm_assisted"]
        changed = "\n\n".join(
            f"Hunk key: {hunk_key(hunk)}\nFile: {hunk.file_path}\n"
            f"Added code:\n{hunk.added_code or '(none)'}"
            for hunk in state.hunks
        )
        prompt = (
            "Analyze each diff hunk and select relevant deterministic analysis rules. "
            "Return JSON only: {\"summary\": str, \"plan_by_hunk\": "
            "{\"file_path:new_start\": [str]}, \"risk_areas\": [str]}. "
            "Every plan_by_hunk key must be one supplied Hunk key. Do not select "
            "llm_assisted in this initial plan.\n\n"
            f"Changed hunks:\n{changed}\n\n"
            f"Long-term feedback:\n{json.dumps(feedback)}\n\n"
            f"Historical risks:\n{json.dumps(history)}\n\n"
            f"Available deterministic rules:\n{json.dumps(available)}"
        )
        try:
            response = self.llm.chat_json("You are a code risk planner.", prompt)
            raw_plan = response.get("plan_by_hunk", {}) if response else {}
        except Exception:
            raw_plan = {}

        valid_keys = {hunk_key(hunk) for hunk in state.hunks}
        pending = self._normalize_plan(raw_plan, valid_keys, set(available))
        return {
            "pending_tools_by_hunk": pending,
            "phase": AgentPhase.TOOL_ROUTING,
            "long_term_feedback": feedback,
            "rag_context": {"history": history},
        }

    def _file_feedback(self, hunk: HunkInfo) -> list[str]:
        try:
            items = self.ltm.get_feedback(hunk.file_path)
        except Exception:
            return []
        return [
            f"[{item.get('rule_id') or 'general'}/{item.get('feedback_type', '')}] "
            f"{hunk.file_path}: {item.get('feedback_content', str(item))}"
            for item in items
        ]

    @staticmethod
    def _normalize_plan(
        raw_plan: object, valid_keys: set[str], available: set[str],
    ) -> dict[str, list[str]]:
        """Drop malformed, unknown and duplicate hunk-rule assignments."""
        if not isinstance(raw_plan, dict):
            return {}
        result: dict[str, list[str]] = {}
        for key, rules in raw_plan.items():
            if not isinstance(key, str) or key not in valid_keys or not isinstance(rules, list):
                continue
            accepted = list(dict.fromkeys(
                rule for rule in rules
                if isinstance(rule, str) and rule in available
            ))
            if accepted:
                result[key] = accepted
        return result
