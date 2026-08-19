"""Security rules: SEC001-SEC005."""

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

# Patterns that indicate a removed line was a security/robustness guard.
# Deleting such lines (e.g. validation, permission checks, lock release,
# defensive raises) is a high-risk change shape that added-line rules
# cannot see.
_REMOVED_GUARD_RES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*assert\b"), "assert guard"),
    (re.compile(r"^\s*raise\b"), "defensive raise"),
    (re.compile(r"^\s*if\s+not\b.*:\s*(#.*)?$"), "negated guard clause"),
    (re.compile(r"^\s*(with\b.*lock|finally\s*:)|\.release\(|\.unlock\("), "lock/resource release"),
    (re.compile(
        r"\b(validate|sanitize|verify|permission|authorize|authenticate|"
        r"is_admin|check_auth|escape)\w*\s*\(",
        re.IGNORECASE,
    ), "validation/authorization call"),
]


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


def removed_security_check(hunk: HunkInfo) -> list[Evidence]:
    """Flag removed lines that look like security/robustness guards.

    Low-confidence by design: deletion may be a benign refactor, so this
    only surfaces the suspicion for the Judge to adjudicate with context.
    Line numbers refer to the OLD file (removed lines have no new-file
    position); the message says so explicitly.
    """
    results: list[Evidence] = []
    for line in hunk.removed_lines:
        if line.change_type != ChangeType.REMOVED:
            continue
        for pattern, desc in _REMOVED_GUARD_RES:
            if pattern.search(line.content):
                old_ln = line.old_line_no or hunk.old_start or 0
                results.append(
                    Evidence(
                        source="removed_security_check", rule_id="SEC005",
                        category=RiskCategory.SECURITY,
                        severity=Severity.MEDIUM,
                        message=(
                            f"Removed line looks like a {desc} "
                            f"(old line {old_ln}); verify the guard is "
                            f"relocated, not dropped"
                        ),
                        line_range=(old_ln, old_ln),
                        snippet=line.content,
                        confidence=0.6,
                        source_type="deterministic",
                    )
                )
                break  # one evidence per removed line is enough
    return results


registry.register("sql_injection", sql_injection)
registry.register("hardcoded_secret", hardcoded_secret)
registry.register("command_injection", command_injection)
registry.register("unsafe_deserialize", unsafe_deserialize)
registry.register("removed_security_check", removed_security_check)
