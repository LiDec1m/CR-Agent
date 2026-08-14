"""LLM-driven planning node."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.memory.long_term import LongTermMemory
from src.models import AgentPhase, AgentState
from src.rag.retriever import RAGRetriever
from src.rules import registry


class PlannerNode:
    """Load context and choose deterministic rules for changed code."""

    def __init__(self, llm: LLMClient, rag: RAGRetriever, ltm: LongTermMemory) -> None:
        self.llm = llm
        self.rag = rag
        self.ltm = ltm

    def __call__(self, state: AgentState) -> dict:
        feedback: list[str] = []
        history: list[dict] = []

        for hunk in state.hunks:
            try:
                file_feedback = self.ltm.get_feedback(hunk.file_path)
                for item in file_feedback:
                    rule_id = item.get("rule_id") or "general"
                    fb_type = item.get("feedback_type", "")
                    content = item.get("feedback_content", str(item))
                    feedback.append(
                        f"[{rule_id}/{fb_type}] {hunk.file_path}: {content}"
                    )
            except Exception:
                pass
            try:
                history.extend(self.rag.search_history(hunk.added_code, hunk.file_path))
            except Exception:
                pass

        # Note: Cross-file feedback (feedback from OTHER files for the
        # same rules) is fetched in the Reflection node, not here, because
        # at the Planner stage we don't yet know which rules will be
        # triggered. Reflection has state.rules_executed to filter by.

        changed_code = "\n\n".join(
            f"File: {hunk.file_path}\n{hunk.added_code}" for hunk in state.hunks
        )
        # Exclude llm_assisted from the initial plan — it should only
        # be triggered by Reflection when deterministic coverage is
        # insufficient, never in the first round.
        available_rules = [r for r in registry.list_all() if r != "llm_assisted"]

        prompt = (
            "Analyze this code change and select relevant analysis rules. "
            "Return JSON only: {\"summary\": str, \"plan\": [str], "
            "\"risk_areas\": [str]}.\n\n"
            f"Changed code:\n{changed_code}\n\n"
            f"Long-term feedback:\n{json.dumps(feedback)}\n\n"
            f"Historical risks:\n{json.dumps(history)}\n\n"
            f"Available rules:\n{json.dumps(available_rules)}"
        )
        try:
            response = json.loads(self.llm.chat("You are a code risk planner.", prompt))
            plan = response.get("plan", [])
        except Exception:
            plan = []

        return {
            "plan": plan,
            "phase": AgentPhase.TOOL_ROUTING,
            "long_term_feedback": feedback,
            "rag_context": {"history": history},
        }
