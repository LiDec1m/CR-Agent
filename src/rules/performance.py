"""Performance rules: PERF001-PERF003."""

from __future__ import annotations

import re

from src.models import Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry

_IO_IN_LOOP_RE = re.compile(
    r"\b(?:open|request|urllib|requests\.\w+)\s*\(", re.IGNORECASE,
)
_DB_IN_LOOP_RE = re.compile(
    r"\b(?:execute|query|fetch|find|select|insert|update|delete)\s*\(",
    re.IGNORECASE,
)


def _ev(
    rule_id: str, source: str, msg: str,
    severity: Severity, ln: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id, category=RiskCategory.PERFORMANCE,
        severity=severity, message=msg, line_range=(ln, ln),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def _is_inside_loop(lines: list[str], idx: int) -> bool:
    """Walk upward from *idx* — return True if we are inside a for/while block."""
    for i in range(idx - 1, -1, -1):
        if re.match(r"^\s*(for|while)\s", lines[i]):
            return True
        stripped = lines[i].strip()
        # If we hit a non-indented, non-comment statement we've left the block
        if stripped and not stripped.startswith("#") and not lines[i][0].isspace():
            return False
    return False


def io_in_loop(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    added = [l.content for l in hunk.added_lines]
    for i, content in enumerate(added):
        if _IO_IN_LOOP_RE.search(content) and _is_inside_loop(added, i):
            ln = hunk.added_lines[i].new_line_no or 0
            results.append(_ev(
                "PERF001", "io_in_loop",
                f"IO operation inside loop at line {ln}",
                Severity.HIGH, ln, content,
            ))
    return results


def n_plus_1_query(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    added = [l.content for l in hunk.added_lines]
    for i, content in enumerate(added):
        if _DB_IN_LOOP_RE.search(content) and _is_inside_loop(added, i):
            ln = hunk.added_lines[i].new_line_no or 0
            results.append(_ev(
                "PERF002", "n_plus_1_query",
                f"Potential N+1 query: DB call inside loop at line {ln}",
                Severity.HIGH, ln, content,
            ))
    return results


def string_concat_in_loop(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    concat_re = re.compile(r"\+=\s*str\(|\+=\s*['\"]")
    added = [l.content for l in hunk.added_lines]
    for i, content in enumerate(added):
        if concat_re.search(content) and _is_inside_loop(added, i):
            ln = hunk.added_lines[i].new_line_no or 0
            results.append(_ev(
                "PERF003", "string_concat_in_loop",
                f"String concatenation with += inside loop at line {ln}",
                Severity.MEDIUM, ln, content,
            ))
    return results


registry.register("io_in_loop", io_in_loop)
registry.register("n_plus_1_query", n_plus_1_query)
registry.register("string_concat_in_loop", string_concat_in_loop)
