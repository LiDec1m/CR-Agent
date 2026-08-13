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
                for item in self.ltm.get_feedback(hunk.file_path):
                    content = item.get("feedback_content", str(item))
                    feedback.append(f"{hunk.file_path}: {content}")
            except Exception:
                pass
            try:
                history.extend(self.rag.search_history(hunk.added_code, hunk.file_path))
            except Exception:
                pass

        changed_code = "\n\n".join(
            f"File: {hunk.file_path}\n{hunk.added_code}" for hunk in state.hunks
        )
        prompt = (
            "Analyze this code change and select relevant analysis rules. "
            "Return JSON only: {\"summary\": str, \"plan\": [str], "
            "\"risk_areas\": [str]}.\n\n"
            f"Changed code:\n{changed_code}\n\n"
            f"Long-term feedback:\n{json.dumps(feedback)}\n\n"
            f"Historical risks:\n{json.dumps(history)}\n\n"
            f"Available rules:\n{json.dumps(registry.list_all())}"
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
