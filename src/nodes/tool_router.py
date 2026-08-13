"""Deterministic rule-routing node."""

from __future__ import annotations

from src.models import AgentPhase, AgentState, Evidence
from src.rag.retriever import RAGRetriever
from src.rules.registry import ToolRegistry


class ToolRouterNode:
    """Run selected rules against every hunk and retrieve code context."""

    def __init__(self, registry: ToolRegistry, rag: RAGRetriever) -> None:
        self.registry = registry
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        selected_tools = (
            state.additional_tools_needed
            if state.needs_more_analysis and state.additional_tools_needed
            else state.plan
        )
        evidence_pool: list[Evidence] = []
        rules_executed: list[str] = []
        for rule_name in selected_tools:
            if rule_name not in self.registry.list_all():
                continue
            rules_executed.append(rule_name)
            for hunk in state.hunks:
                try:
                    evidence_pool.extend(self.registry.execute(rule_name, hunk))
                except Exception:
                    pass

        codebase: dict[str, list[dict]] = {}
        for hunk in state.hunks:
            try:
                codebase[hunk.file_path] = self.rag.search_codebase(hunk.file_path)
            except Exception:
                codebase[hunk.file_path] = []

        rag_context = dict(state.rag_context)
        rag_context["codebase"] = codebase
        return {
            "evidence_pool": evidence_pool,
            "rules_executed": rules_executed,
            "selected_tools": selected_tools,
            "phase": AgentPhase.JUDGING,
            "needs_more_analysis": False,
            "additional_tools_needed": [],
            "rag_context": rag_context,
        }
