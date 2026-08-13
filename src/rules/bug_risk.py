"""Bug risk rules: BUG001-BUG004."""

from __future__ import annotations

import ast
import re

from src.models import Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry

_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:")
_MUTABLE_DEFAULT_RE = re.compile(r"def\s+\w+\(.*=\s*(\[\]|\{\}|set\(\))", re.DOTALL)


def _ev(
    rule_id: str, source: str, msg: str,
    severity: Severity, ln: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id, category=RiskCategory.BUG_RISK,
        severity=severity, message=msg, line_range=(ln, ln),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def bare_except(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        if _BARE_EXCEPT_RE.match(line.content):
            ln = line.new_line_no or 0
            results.append(_ev(
                "BUG001", "bare_except",
                f"Bare except clause at line {ln}",
                Severity.HIGH, ln, line.content,
            ))
    return results


def mutable_default_arg(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to regex if we can't parse the AST
        for line in hunk.added_lines:
            if _MUTABLE_DEFAULT_RE.search(line.content):
                ln = line.new_line_no or 0
                results.append(_ev(
                    "BUG002", "mutable_default_arg",
                    f"Mutable default argument at line {ln}",
                    Severity.HIGH, ln, line.content,
                ))
        return results
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.defaults:
                if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                    results.append(_ev(
                        "BUG002", "mutable_default_arg",
                        f"Function '{node.name}' has mutable default argument",
                        Severity.HIGH, node.lineno, f"def {node.name}(...):",
                    ))
    return results


def unused_import(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    code = hunk.added_code
    if not code.strip():
        return results
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return results
    imports: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imports.append((alias.name, name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                imports.append((alias.name, name, node.lineno))
    for full_name, name, lineno in imports:
        if name == "*":
            continue
        # Check if the name appears anywhere outside its own import statement
        used = name in code.replace(f"import {full_name}", "").replace(f"import {name}", "")
        if not used:
            results.append(_ev(
                "BUG003", "unused_import",
                f"Imported '{full_name}' appears unused",
                Severity.LOW, lineno, f"import {full_name}",
            ))
    return results


def none_unsafe_access(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    optional_re = re.compile(r":\s*Optional\[(\w+)\]|:\s*(\w+)\s*\|\s*None")
    for line in hunk.added_lines:
        match = optional_re.search(line.content)
        if match and ".strip()" not in line.content:
            ln = line.new_line_no or 0
            var_name = match.group(1) or match.group(2)
            if var_name:
                results.append(_ev(
                    "BUG004", "none_unsafe_access",
                    f"Variable '{var_name}' is Optional but may be accessed without None check",
                    Severity.MEDIUM, ln, line.content,
                ))
    return results


registry.register("bare_except", bare_except)
registry.register("mutable_default_arg", mutable_default_arg)
registry.register("unused_import", unused_import)
registry.register("none_unsafe_access", none_unsafe_access)
