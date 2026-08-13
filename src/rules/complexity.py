"""Complexity rules: CX001-CX003."""

from __future__ import annotations

import ast

from src.models import Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry


def _ev(
    rule_id: str, source: str, msg: str,
    severity: Severity, ln: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id, category=RiskCategory.COMPLEXITY,
        severity=severity, message=msg, line_range=(ln, ln),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def function_too_long(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = node.end_lineno - node.lineno + 1 if node.end_lineno else 0
            if length > 50:
                results.append(_ev(
                    "CX001", "function_too_long",
                    f"Function '{node.name}' is {length} lines long (threshold: 50)",
                    Severity.MEDIUM, node.lineno, f"def {node.name}(...):",
                ))
    return results


def high_cyclomatic(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
            if complexity > 10:
                results.append(_ev(
                    "CX002", "high_cyclomatic",
                    f"Function '{node.name}' has cyclomatic complexity {complexity} (threshold: 10)",
                    Severity.MEDIUM, node.lineno, f"def {node.name}(...):",
                ))
    return results


def deep_nesting(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results

    def _depth(node: ast.AST, current: int = 0) -> int:
        max_d = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                d = _depth(child, current + 1)
            else:
                d = _depth(child, current)
            max_d = max(max_d, d)
        return max_d

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = _depth(node)
            if depth > 4:
                results.append(_ev(
                    "CX003", "deep_nesting",
                    f"Function '{node.name}' has nesting depth {depth} (threshold: 4)",
                    Severity.MEDIUM, node.lineno, f"def {node.name}(...):",
                ))
    return results


registry.register("function_too_long", function_too_long)
registry.register("high_cyclomatic", high_cyclomatic)
registry.register("deep_nesting", deep_nesting)
