"""Style rules: STY001-STY003."""

from __future__ import annotations

import re

from src.models import Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry

_MAGIC_NUM_RE = re.compile(r"(?<![\w.])\d{4,}(?![\w.])")
_CAMEL_CASE_RE = re.compile(r"\bdef\s+([a-z]+[A-Z]\w*)\(")
_SNAKE_VIOLATION_RE = re.compile(r"\bdef\s+([A-Z]\w*)\(")


def _ev(
    rule_id: str, source: str, msg: str,
    severity: Severity, ln: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id, category=RiskCategory.STYLE,
        severity=severity, message=msg, line_range=(ln, ln),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def naming_violation(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        ln = line.new_line_no or 0
        if _CAMEL_CASE_RE.search(line.content):
            match = _CAMEL_CASE_RE.search(line.content)
            if match:
                results.append(_ev(
                    "STY001", "naming_violation",
                    f"Function '{match.group(1)}' uses camelCase (expected snake_case)",
                    Severity.LOW, ln, line.content,
                ))
        elif _SNAKE_VIOLATION_RE.search(line.content):
            match = _SNAKE_VIOLATION_RE.search(line.content)
            if match:
                results.append(_ev(
                    "STY001", "naming_violation",
                    f"Function '{match.group(1)}' starts with uppercase (expected snake_case)",
                    Severity.LOW, ln, line.content,
                ))
    return results


def magic_number(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        ln = line.new_line_no or 0
        if _MAGIC_NUM_RE.search(line.content):
            results.append(_ev(
                "STY002", "magic_number",
                f"Magic number detected at line {ln}",
                Severity.LOW, ln, line.content,
            ))
    return results


def long_line(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        ln = line.new_line_no or 0
        if len(line.content) > 120:
            results.append(_ev(
                "STY003", "long_line",
                f"Line {ln} is {len(line.content)} characters (threshold: 120)",
                Severity.LOW, ln, line.content,
            ))
    return results


registry.register("naming_violation", naming_violation)
registry.register("magic_number", magic_number)
registry.register("long_line", long_line)
