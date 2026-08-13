"""Security rules: SEC001-SEC004."""

from __future__ import annotations

import re

from src.models import ChangeType, Evidence, HunkInfo, RiskCategory, Severity
from src.rules.registry import registry

_SQL_CONCAT_RE = re.compile(
    r"""['"](?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|SQL)\b""",
    re.IGNORECASE,
)
_STRING_CONCAT_RE = re.compile(r"""['"][^'"]*['"]\s*\+""")
_SECRET_RE = re.compile(
    r"""(?:api[_-]?key|password|secret|token|passwd|pwd)\s*=\s*['"][^'"]{8,}['"]""",
    re.IGNORECASE,
)
_OS_SYSTEM_RE = re.compile(r"os\.system\s*\(")
_SHELL_TRUE_RE = re.compile(r"shell\s*=\s*True")
_EVAL_RE = re.compile(r"\beval\s*\(")
_EXEC_RE = re.compile(r"\bexec\s*\(")
_PICKLE_RE = re.compile(r"pickle\.loads?\s*\(")


def _make_evidence(
    rule_id: str, source: str, message: str,
    severity: Severity, line_no: int, snippet: str,
) -> Evidence:
    return Evidence(
        source=source, rule_id=rule_id,
        category=RiskCategory.SECURITY, severity=severity,
        message=message, line_range=(line_no, line_no),
        snippet=snippet, confidence=1.0, source_type="deterministic",
    )


def sql_injection(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        if line.change_type != ChangeType.ADDED:
            continue
        if _SQL_CONCAT_RE.search(line.content) and _STRING_CONCAT_RE.search(
            line.content
        ):
            ln = line.new_line_no or 0
            results.append(
                _make_evidence(
                    "SEC001", "sql_injection",
                    f"Potential SQL injection: string concatenation in SQL query at line {ln}",
                    Severity.HIGH, ln, line.content,
                )
            )
    return results


def hardcoded_secret(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        if _SECRET_RE.search(line.content):
            ln = line.new_line_no or 0
            results.append(
                _make_evidence(
                    "SEC002", "hardcoded_secret",
                    f"Hardcoded secret/credential detected at line {ln}",
                    Severity.CRITICAL, ln, line.content,
                )
            )
    return results


def command_injection(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        ln = line.new_line_no or 0
        if _OS_SYSTEM_RE.search(line.content):
            results.append(
                _make_evidence(
                    "SEC003", "command_injection",
                    f"Command injection risk: os.system() at line {ln}",
                    Severity.HIGH, ln, line.content,
                )
            )
        if _SHELL_TRUE_RE.search(line.content):
            results.append(
                _make_evidence(
                    "SEC003", "command_injection",
                    f"Command injection risk: shell=True at line {ln}",
                    Severity.HIGH, ln, line.content,
                )
            )
    return results


def unsafe_deserialize(hunk: HunkInfo) -> list[Evidence]:
    results: list[Evidence] = []
    for line in hunk.added_lines:
        ln = line.new_line_no or 0
        for pattern, desc in [
            (_EVAL_RE, "eval()"),
            (_EXEC_RE, "exec()"),
            (_PICKLE_RE, "pickle.loads()"),
        ]:
            if pattern.search(line.content):
                results.append(
                    _make_evidence(
                        "SEC004", "unsafe_deserialize",
                        f"Unsafe deserialization: {desc} at line {ln}",
                        Severity.HIGH, ln, line.content,
                    )
                )
    return results


registry.register("sql_injection", sql_injection)
registry.register("hardcoded_secret", hardcoded_secret)
registry.register("command_injection", command_injection)
registry.register("unsafe_deserialize", unsafe_deserialize)
