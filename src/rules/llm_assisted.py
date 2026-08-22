"""LLM-assisted rule: uses LLM to find risks that deterministic rules miss.

This rule is registered dynamically (not at import time) because it needs
an LLMClient instance. It is only triggered when the Reflection node
decides that deterministic rules have insufficient coverage and suggests
``"llm_assisted"`` in ``additional_tools_by_hunk``.

The Planner node explicitly excludes this rule from the initial plan so
it can never run in the first round.
"""

from __future__ import annotations

from typing import Any

from src.llm.client import LLMClient
from src.models import Evidence, HunkInfo, RiskCategory, Severity

_SEVERITY_VALUES = sorted(s.value for s in Severity)


def _validate_llm_evidences(parsed) -> None:
    """Contract check run inside the chat_json retry loop.

    Raises ValueError on contract violations (non-list evidences or an
    unknown severity), so chat_json retries with a repair prompt instead
    of the caller silently degrading to defaults.
    """
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    evidences = parsed.get("evidences", [])
    if not isinstance(evidences, list):
        raise ValueError("'evidences' must be a list")
    for i, ev in enumerate(evidences):
        if not isinstance(ev, dict):
            raise ValueError(f"evidences[{i}] must be an object")
        severity = ev.get("severity")
        if severity not in _SEVERITY_VALUES:
            raise ValueError(
                f"evidences[{i}].severity must be one of {_SEVERITY_VALUES}; "
                f"got {severity!r}"
            )


class LLMAnalysisDegraded(RuntimeError):
    """The LLM rule could not produce a reliable assessment for one hunk."""


def create_llm_assisted_rule(llm: LLMClient) -> Any:
    """Create an LLM-assisted rule function bound to the given LLM client.

    Returns a callable matching the ``RuleFunc`` signature
    ``(HunkInfo) -> list[Evidence]``.
    """

    def llm_assisted(hunk: HunkInfo) -> list[Evidence]:
        code = hunk.added_code
        removed = "\n".join(l.content for l in hunk.removed_lines)
        if not code.strip() and not removed.strip():
            return []

        prompt = (
            "Analyze this code change for risks that deterministic rules "
            "might miss (e.g. logic errors, race conditions, resource "
            "leaks, missing input validation, error handling gaps). "
            "If removed lines are provided, check specifically whether the "
            "deletion drops a validation, permission check, or resource "
            "release without re-adding it elsewhere. "
            "Return JSON only: "
            "{\"evidences\": [{\"rule_id\": str, \"category\": str, "
            "\"severity\": one of \"info\", \"low\", \"medium\", "
            "\"high\", \"critical\", \"message\": str, \"line_no\": int}]}\n\n"
            f"File: {hunk.file_path}\n"
            f"Added code:\n{code or '(none)'}"
        )
        if removed.strip():
            prompt += f"\n\nRemoved code (old lines):\n{removed}"

        try:
            response = llm.chat_json(
                "You are a code risk analyzer.", prompt,
                validator=_validate_llm_evidences,
            )
        except Exception as exc:
            raise LLMAnalysisDegraded(
                f"LLM analysis failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response is None:
            raise LLMAnalysisDegraded("LLM analysis returned no parseable JSON")
        raw_evidences = response.get("evidences", [])
        if not isinstance(raw_evidences, list):
            raise LLMAnalysisDegraded("LLM analysis returned a non-list evidences field")

        results: list[Evidence] = []
        for ev in raw_evidences:
            # Severity is guaranteed valid by _validate_llm_evidences; the
            # try/except is a last-resort guard so a contract regression
            # degrades to a labeled fallback instead of crashing the round.
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
