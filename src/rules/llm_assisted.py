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

    Raises ValueError on contract violations, so chat_json retries with
    a repair prompt instead of the caller silently degrading to
    defaults. The contract is strict on purpose: a missing
    ``evidences`` key, a non-int/negative ``line_no``, or an incomplete
    evidence item is a malformed response to be repaired, not "zero
    findings".
    """
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")
    if "evidences" not in parsed:
        raise ValueError("response must contain an 'evidences' key")
    evidences = parsed["evidences"]
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
        line_no = ev.get("line_no")
        if (
            not isinstance(line_no, int) or isinstance(line_no, bool)
            or line_no < 0
        ):
            raise ValueError(
                f"evidences[{i}].line_no must be a non-negative int; "
                f"got {line_no!r}"
            )
        message = ev.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(
                f"evidences[{i}].message must be a non-empty string"
            )
        rule_id = ev.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id:
            raise ValueError(
                f"evidences[{i}].rule_id must be a non-empty string"
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
            # Severity/line_no/message/rule_id are guaranteed valid by
            # _validate_llm_evidences; the fallbacks below are a
            # last-resort guard so a contract regression degrades to a
            # labeled fallback instead of crashing the round. They also
            # catch TypeError: non-string enum inputs (None, list) raise
            # TypeError, not ValueError, from enum construction.
            try:
                category = RiskCategory(ev.get("category", "bug_risk"))
            except (ValueError, TypeError):
                category = RiskCategory.BUG_RISK
            try:
                severity = Severity(ev.get("severity", "medium"))
            except (ValueError, TypeError):
                severity = Severity.MEDIUM

            line_no = ev.get("line_no", 0)
            if (
                not isinstance(line_no, int) or isinstance(line_no, bool)
                or line_no < 0
            ):
                line_no = 0
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
