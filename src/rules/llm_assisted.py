"""LLM-assisted rule: uses LLM to find risks that deterministic rules miss.

This rule is registered dynamically (not at import time) because it needs
an LLMClient instance. It is only triggered when the Reflection node
decides that deterministic rules have insufficient coverage and suggests
``"llm_assisted"`` in ``additional_tools_needed``.

The Planner node explicitly excludes this rule from the initial plan so
it can never run in the first round.
"""

from __future__ import annotations

import json
from typing import Any

from src.llm.client import LLMClient
from src.models import ChangeType, Evidence, HunkInfo, RiskCategory, Severity


def create_llm_assisted_rule(llm: LLMClient) -> Any:
    """Create an LLM-assisted rule function bound to the given LLM client.

    Returns a callable matching the ``RuleFunc`` signature
    ``(HunkInfo) -> list[Evidence]``.
    """

    def llm_assisted(hunk: HunkInfo) -> list[Evidence]:
        code = hunk.added_code
        if not code.strip():
            return []

        prompt = (
            "Analyze this code change for risks that deterministic rules "
            "might miss (e.g. logic errors, race conditions, resource "
            "leaks, missing input validation, error handling gaps). "
            "Return JSON only: "
            "{\"evidences\": [{\"rule_id\": str, \"category\": str, "
            "\"severity\": str, \"message\": str, \"line_no\": int}]}\n\n"
            f"File: {hunk.file_path}\n"
            f"Added code:\n{code}"
        )

        try:
            response = json.loads(
                llm.chat("You are a code risk analyzer.", prompt)
            )
            raw_evidences = response.get("evidences", [])
        except Exception:
            return []

        results: list[Evidence] = []
        for ev in raw_evidences:
            try:
                category = RiskCategory(ev.get("category", "bug_risk"))
            except ValueError:
                category = RiskCategory.BUG_RISK
            try:
                severity = Severity(ev.get("severity", "medium"))
            except ValueError:
                severity = Severity.MEDIUM

            line_no = ev.get("line_no", 0) or 0
            results.append(
                Evidence(
                    source="llm_assisted",
                    rule_id=ev.get("rule_id", "LLM001"),
                    category=category,
                    severity=severity,
                    message=ev.get("message", ""),
                    line_range=(line_no, line_no),
                    snippet=code,
                    confidence=0.7,
                    source_type="llm",
                )
            )
        return results

    return llm_assisted
