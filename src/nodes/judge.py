"""LLM-driven risk-judgment node."""

from __future__ import annotations

import json

from src.llm.client import LLMClient
from src.models import AgentPhase, AgentState, RiskCategory, RiskItem, Severity
from src.rag.retriever import RAGRetriever


class JudgeNode:
    """Convert evidence and RAG context into structured risk items."""

    def __init__(self, llm: LLMClient, rag: RAGRetriever) -> None:
        self.llm = llm
        self.rag = rag

    def __call__(self, state: AgentState) -> dict:
        rule_ids = list({item.rule_id for item in state.evidence_pool if item.rule_id})
        try:
            security = self.rag.search_security(
                " ".join(item.message for item in state.evidence_pool), rule_ids
            )
        except Exception:
            security = []

        evidence = [item.model_dump(mode="json") for item in state.evidence_pool]
        prompt = (
            "Judge these code-risk evidences. Return JSON only: "
            "{\"risks\": [{\"title\": str, \"category\": str, "
            "\"severity\": str, \"description\": str, \"evidence_refs\": [int], "
            "\"suggestion\": str, \"file_path\": str, \"line_range\": [int, int], "
            "\"risk_score\": float (0.0-1.0)}], \"overall_risk_score\": float (0.0-1.0)}.\n\n"
            f"Evidence (reference by zero-based index):\n{json.dumps(evidence)}\n\n"
            f"Security knowledge:\n{json.dumps(security)}\n\n"
            f"Codebase context:\n{json.dumps(state.rag_context.get('codebase', {}))}"
        )
        try:
            response = json.loads(self.llm.chat("You are a code risk judge.", prompt))
            raw_risks = response.get("risks", [])
        except Exception:
            raw_risks = []

        risks: list[RiskItem] = []
        for item in raw_risks:
            refs = item.get("evidence_refs", [])
            chain = [
                state.evidence_pool[index]
                for index in refs
                if isinstance(index, int) and 0 <= index < len(state.evidence_pool)
            ]
            line_range = item.get("line_range")
            if isinstance(line_range, list) and len(line_range) == 2:
                line_range = tuple(line_range)
            else:
                line_range = None
            try:
                category = RiskCategory(item.get("category", "security"))
            except ValueError:
                category = RiskCategory.SECURITY
            try:
                severity = Severity(item.get("severity", "medium"))
            except ValueError:
                severity = Severity.MEDIUM
            raw_score = item.get("risk_score", 0.0)
            if isinstance(raw_score, (int, float)) and raw_score > 1.0:
                raw_score = min(raw_score / 10.0, 1.0)
            risks.append(RiskItem(
                title=item.get("title", "Unknown risk"),
                category=category,
                severity=severity,
                description=item.get("description", ""),
                evidence_chain=chain,
                suggestion=item.get("suggestion"),
                file_path=item.get("file_path"),
                line_range=line_range,
                risk_score=raw_score,
            ))

        rag_context = dict(state.rag_context)
        rag_context["security"] = security
        return {
            "risks": risks,
            "phase": AgentPhase.REFLECTING,
            "rag_context": rag_context,
        }
