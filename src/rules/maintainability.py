"""Maintainability rules: MAIN001-MAIN003."""

from __future__ import annotations

import ast
import re

from src.models import Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)", re.IGNORECASE)


def _ev(
    rule_id: str, source: str, msg: str,
    severity: Severity, ln: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id, category=RiskCategory.MAINTAINABILITY,
        severity=severity, message=msg, line_range=(ln, ln),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def missing_docstring(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                results.append(_ev(
                    "MAIN001", "missing_docstring",
                    f"'{node.name}' is missing a docstring",
                    Severity.LOW, node.lineno, f"def {node.name}(...):",
                ))
    return results


def duplicate_pattern(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    added = [l.content.strip() for l in hunk.added_lines if l.content.strip()]
    seen: dict[str, int] = {}
    for i, line in enumerate(added):
        if len(line) > 20 and line in seen:
            ln = hunk.added_lines[i].new_line_no or 0
            results.append(_ev(
                "MAIN002", "duplicate_pattern",
                f"Duplicate code pattern at line {ln} (also at line {seen[line]})",
                Severity.MEDIUM, ln, line,
            ))
        elif len(line) > 20:
            seen[line] = hunk.added_lines[i].new_line_no or 0
    return results


def todo_fixme(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        if _TODO_RE.search(line.content):
            ln = line.new_line_no or 0
            results.append(_ev(
                "MAIN003", "todo_fixme",
                f"TODO/FIXME/HACK marker found at line {ln}",
                Severity.INFO, ln, line.content,
            ))
    return results


registry.register("missing_docstring", missing_docstring)
registry.register("duplicate_pattern", duplicate_pattern)
registry.register("todo_fixme", todo_fixme)
