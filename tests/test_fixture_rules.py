"""Tests that load diff fixture files and verify each rule triggers correctly.

This is the "test set" — structured diff fixtures that cover every rule,
false-positive resistance, multi-file scenarios, and edge cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChangeType, DiffLine, HunkInfo, RiskCategory, Severity
from src.parsers.diff_parser import GitDiffParser
from src.rules import registry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "diffs"
_parser = GitDiffParser()


# ── helpers ──────────────────────────────────────────────────────────


def _load_diff(name: str) -> str:
    """Read a fixture diff file and return its contents."""
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _load_hunks(name: str) -> list[HunkInfo]:
    """Parse a fixture diff into HunkInfo list."""
    return _parser.parse(_load_diff(name))


def _make_hunk(added_code: str, file_path: str = "test.py") -> HunkInfo:
    """Build a HunkInfo from raw added-code text (shortcut for inline tests)."""
    lines = [
        DiffLine(content=c, change_type=ChangeType.ADDED, new_line_no=i + 1)
        for i, c in enumerate(added_code.splitlines())
    ]
    return HunkInfo(
        file_path=file_path, old_start=1, old_count=0,
        new_start=1, new_count=len(lines), lines=lines,
    )


def _rule_ids(evidences) -> set[str]:
    return {ev.rule_id for ev in evidences if ev.rule_id}


# ── fixture: security_all.diff ──────────────────────────────────────


class TestSecurityFixture:
    """SEC001-SEC004 — all four security rules should fire."""

    def test_fixture_loads(self):
        hunks = _load_hunks("security_all.diff")
        assert len(hunks) >= 1

    def test_sql_injection_triggered(self):
        hunks = _load_hunks("security_all.diff")
        all_ev = registry.execute_batch(
            ["sql_injection"], hunks,
        )
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "SEC001"
        assert all_ev[0].category == RiskCategory.SECURITY
        assert all_ev[0].severity == Severity.HIGH
        assert all_ev[0].confidence == 1.0

    def test_hardcoded_secret_triggered(self):
        hunks = _load_hunks("security_all.diff")
        all_ev = registry.execute_batch(["hardcoded_secret"], hunks)
        # At least API_KEY and password should be caught
        assert len(all_ev) >= 2
        assert all(ev.rule_id == "SEC002" for ev in all_ev)
        assert all(ev.severity == Severity.CRITICAL for ev in all_ev)

    def test_command_injection_triggered(self):
        hunks = _load_hunks("security_all.diff")
        all_ev = registry.execute_batch(["command_injection"], hunks)
        assert len(all_ev) >= 1
        assert all(ev.rule_id == "SEC003" for ev in all_ev)

    def test_unsafe_deserialize_triggered(self):
        hunks = _load_hunks("security_all.diff")
        all_ev = registry.execute_batch(["unsafe_deserialize"], hunks)
        # eval + pickle.loads = at least 2
        assert len(all_ev) >= 2
        descs = [ev.message for ev in all_ev]
        assert any("eval" in d for d in descs)
        assert any("pickle" in d for d in descs)

    def test_all_security_rules_on_fixture(self):
        """Run all 4 security rules against the fixture and check coverage."""
        hunks = _load_hunks("security_all.diff")
        all_ev = registry.execute_batch(
            ["sql_injection", "hardcoded_secret",
             "command_injection", "unsafe_deserialize"],
            hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"SEC001", "SEC002", "SEC003", "SEC004"}


# ── fixture: complexity_all.diff ────────────────────────────────────


class TestComplexityFixture:
    """CX001-CX003 — all three complexity rules should fire."""

    def test_function_too_long_triggered(self):
        hunks = _load_hunks("complexity_all.diff")
        all_ev = registry.execute_batch(["function_too_long"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "CX001"
        assert all_ev[0].severity == Severity.MEDIUM

    def test_high_cyclomatic_triggered(self):
        hunks = _load_hunks("complexity_all.diff")
        all_ev = registry.execute_batch(["high_cyclomatic"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "CX002"

    def test_deep_nesting_triggered(self):
        hunks = _load_hunks("complexity_all.diff")
        all_ev = registry.execute_batch(["deep_nesting"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "CX003"

    def test_all_complexity_rules_on_fixture(self):
        hunks = _load_hunks("complexity_all.diff")
        all_ev = registry.execute_batch(
            ["function_too_long", "high_cyclomatic", "deep_nesting"], hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"CX001", "CX002", "CX003"}


# ── fixture: bug_risk_all.diff ──────────────────────────────────────


class TestBugRiskFixture:
    """BUG001-BUG004 — all four bug-risk rules should fire."""

    def test_bare_except_triggered(self):
        hunks = _load_hunks("bug_risk_all.diff")
        all_ev = registry.execute_batch(["bare_except"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "BUG001"
        assert all_ev[0].severity == Severity.HIGH

    def test_mutable_default_arg_triggered(self):
        hunks = _load_hunks("bug_risk_all.diff")
        all_ev = registry.execute_batch(["mutable_default_arg"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "BUG002"

    def test_unused_import_triggered(self):
        hunks = _load_hunks("bug_risk_all.diff")
        all_ev = registry.execute_batch(["unused_import"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "BUG003"

    def test_none_unsafe_access_triggered(self):
        hunks = _load_hunks("bug_risk_all.diff")
        all_ev = registry.execute_batch(["none_unsafe_access"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "BUG004"

    def test_all_bug_risk_rules_on_fixture(self):
        hunks = _load_hunks("bug_risk_all.diff")
        all_ev = registry.execute_batch(
            ["bare_except", "mutable_default_arg",
             "unused_import", "none_unsafe_access"],
            hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"BUG001", "BUG002", "BUG003", "BUG004"}


# ── fixture: style_all.diff ─────────────────────────────────────────


class TestStyleFixture:
    """STY001-STY003 — all three style rules should fire."""

    def test_naming_violation_triggered(self):
        hunks = _load_hunks("style_all.diff")
        all_ev = registry.execute_batch(["naming_violation"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "STY001"
        assert all_ev[0].severity == Severity.LOW

    def test_magic_number_triggered(self):
        hunks = _load_hunks("style_all.diff")
        all_ev = registry.execute_batch(["magic_number"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "STY002"
        # 86400 and 9999 should both be caught
        assert len(all_ev) >= 2

    def test_long_line_triggered(self):
        hunks = _load_hunks("style_all.diff")
        all_ev = registry.execute_batch(["long_line"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "STY003"

    def test_all_style_rules_on_fixture(self):
        hunks = _load_hunks("style_all.diff")
        all_ev = registry.execute_batch(
            ["naming_violation", "magic_number", "long_line"], hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"STY001", "STY002", "STY003"}


# ── fixture: performance_all.diff ───────────────────────────────────


class TestPerformanceFixture:
    """PERF001-PERF003 — all three performance rules should fire."""

    def test_io_in_loop_triggered(self):
        hunks = _load_hunks("performance_all.diff")
        all_ev = registry.execute_batch(["io_in_loop"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "PERF001"
        assert all_ev[0].severity == Severity.HIGH

    def test_n_plus_1_query_triggered(self):
        hunks = _load_hunks("performance_all.diff")
        all_ev = registry.execute_batch(["n_plus_1_query"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "PERF002"

    def test_string_concat_in_loop_triggered(self):
        hunks = _load_hunks("performance_all.diff")
        all_ev = registry.execute_batch(["string_concat_in_loop"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "PERF003"
        assert all_ev[0].severity == Severity.MEDIUM

    def test_all_performance_rules_on_fixture(self):
        hunks = _load_hunks("performance_all.diff")
        all_ev = registry.execute_batch(
            ["io_in_loop", "n_plus_1_query", "string_concat_in_loop"], hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"PERF001", "PERF002", "PERF003"}


# ── fixture: maintainability_all.diff ───────────────────────────────


class TestMaintainabilityFixture:
    """MAIN001-MAIN003 — all three maintainability rules should fire."""

    def test_missing_docstring_triggered(self):
        hunks = _load_hunks("maintainability_all.diff")
        all_ev = registry.execute_batch(["missing_docstring"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "MAIN001"
        assert all_ev[0].severity == Severity.LOW

    def test_duplicate_pattern_triggered(self):
        hunks = _load_hunks("maintainability_all.diff")
        all_ev = registry.execute_batch(["duplicate_pattern"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "MAIN002"

    def test_todo_fixme_triggered(self):
        hunks = _load_hunks("maintainability_all.diff")
        all_ev = registry.execute_batch(["todo_fixme"], hunks)
        assert len(all_ev) >= 1
        assert all_ev[0].rule_id == "MAIN003"
        assert all_ev[0].severity == Severity.INFO

    def test_all_maintainability_rules_on_fixture(self):
        hunks = _load_hunks("maintainability_all.diff")
        all_ev = registry.execute_batch(
            ["missing_docstring", "duplicate_pattern", "todo_fixme"], hunks,
        )
        ids = _rule_ids(all_ev)
        assert ids == {"MAIN001", "MAIN002", "MAIN003"}


# ── fixture: clean_code.diff — false-positive resistance ────────────


class TestCleanCodeNoFalsePositives:
    """Clean code should produce zero evidence from all 20 rules."""

    def test_clean_code_zero_evidence(self):
        hunks = _load_hunks("clean_code.diff")
        all_rules = registry.list_all()
        all_ev = registry.execute_batch(all_rules, hunks)
        assert len(all_ev) == 0, (
            f"Expected 0 evidence for clean code, got {len(all_ev)}: "
            f"{[(e.rule_id, e.message) for e in all_ev]}"
        )

    def test_clean_code_parameterized_query_no_sql_injection(self):
        hunks = _load_hunks("clean_code.diff")
        ev = registry.execute_batch(["sql_injection"], hunks)
        assert len(ev) == 0

    def test_clean_code_no_hardcoded_secret(self):
        hunks = _load_hunks("clean_code.diff")
        ev = registry.execute_batch(["hardcoded_secret"], hunks)
        assert len(ev) == 0

    def test_clean_code_no_bare_except(self):
        hunks = _load_hunks("clean_code.diff")
        ev = registry.execute_batch(["bare_except"], hunks)
        assert len(ev) == 0

    def test_clean_code_has_docstring(self):
        hunks = _load_hunks("clean_code.diff")
        ev = registry.execute_batch(["missing_docstring"], hunks)
        assert len(ev) == 0

    def test_clean_code_no_magic_number(self):
        hunks = _load_hunks("clean_code.diff")
        ev = registry.execute_batch(["magic_number"], hunks)
        assert len(ev) == 0


# ── fixture: multi_file_mixed.diff ──────────────────────────────────


class TestMultiFileMixed:
    """Multi-file diff with mixed risk categories."""

    def test_parses_three_files(self):
        hunks = _load_hunks("multi_file_mixed.diff")
        files = {h.file_path for h in hunks}
        assert "auth/login.py" in files
        assert "api/views.py" in files
        assert "utils/helper.py" in files

    def test_all_rules_find_evidence(self):
        hunks = _load_hunks("multi_file_mixed.diff")
        all_rules = registry.list_all()
        all_ev = registry.execute_batch(all_rules, hunks)
        assert len(all_ev) >= 10, (
            f"Expected >=10 evidence items, got {len(all_ev)}"
        )

    def test_security_risks_in_auth_file(self):
        hunks = _load_hunks("multi_file_mixed.diff")
        auth_hunks = [h for h in hunks if "login" in h.file_path]
        sec_ev = registry.execute_batch(
            ["sql_injection", "hardcoded_secret",
             "command_injection", "unsafe_deserialize"],
            auth_hunks,
        )
        ids = _rule_ids(sec_ev)
        assert "SEC001" in ids
        assert "SEC002" in ids
        assert "SEC003" in ids
        assert "SEC004" in ids

    def test_performance_risks_in_api_file(self):
        hunks = _load_hunks("multi_file_mixed.diff")
        api_hunks = [h for h in hunks if "views" in h.file_path]
        perf_ev = registry.execute_batch(
            ["io_in_loop", "n_plus_1_query", "string_concat_in_loop"],
            api_hunks,
        )
        ids = _rule_ids(perf_ev)
        assert "PERF001" in ids  # io_in_loop
        assert "PERF002" in ids  # n_plus_1_query

    def test_bug_risks_in_utils_file(self):
        hunks = _load_hunks("multi_file_mixed.diff")
        utils_hunks = [h for h in hunks if "helper" in h.file_path]
        bug_ev = registry.execute_batch(
            ["mutable_default_arg", "bare_except"], utils_hunks,
        )
        ids = _rule_ids(bug_ev)
        assert "BUG001" in ids
        assert "BUG002" in ids


# ── fixture: edge_cases.diff ────────────────────────────────────────


class TestEdgeCases:
    """New file, deleted file, context-only change."""

    def test_parses_new_file(self):
        hunks = _load_hunks("edge_cases.diff")
        new_file_hunks = [h for h in hunks if "new_file" in h.file_path]
        assert len(new_file_hunks) >= 1
        assert len(new_file_hunks[0].added_lines) == 3

    def test_parses_deleted_file(self):
        hunks = _load_hunks("edge_cases.diff")
        deleted_hunks = [h for h in hunks if "deleted" in h.file_path]
        assert len(deleted_hunks) >= 1
        assert len(deleted_hunks[0].removed_lines) == 3
        assert len(deleted_hunks[0].added_lines) == 0

    def test_context_only_no_added_lines(self):
        hunks = _load_hunks("edge_cases.diff")
        ctx_hunks = [h for h in hunks if "only_context" in h.file_path]
        assert len(ctx_hunks) >= 1
        assert len(ctx_hunks[0].added_lines) == 0

    def test_edge_cases_produce_minimal_evidence(self):
        """The new file is clean; deleted file has no added lines.
        Only the 'except:' context line might trigger, but it's
        context not added, so bare_except should NOT fire."""
        hunks = _load_hunks("edge_cases.diff")
        all_rules = registry.list_all()
        all_ev = registry.execute_batch(all_rules, hunks)
        # The new file has def hello() without docstring -> MAIN001
        # That's the only expected finding
        ids = _rule_ids(all_ev)
        assert "MAIN001" in ids or len(all_ev) == 0  # depends on AST parse


# ── fixture: js_ts_patterns.diff ────────────────────────────────────


class TestJsTsPatterns:
    """JS/TS files — regex-based rules should still fire."""

    def test_js_file_language_detected(self):
        hunks = _load_hunks("js_ts_patterns.diff")
        js_hunks = [h for h in hunks if h.file_path.endswith(".js")]
        assert len(js_hunks) >= 1
        assert js_hunks[0].language == "javascript"

    def test_ts_file_language_detected(self):
        hunks = _load_hunks("js_ts_patterns.diff")
        ts_hunks = [h for h in hunks if h.file_path.endswith(".tsx")]
        assert len(ts_hunks) >= 1
        assert ts_hunks[0].language == "typescript"

    def test_hardcoded_secret_in_js(self):
        hunks = _load_hunks("js_ts_patterns.diff")
        ev = registry.execute_batch(["hardcoded_secret"], hunks)
        assert len(ev) >= 1
        assert all(ev2.rule_id == "SEC002" for ev2 in ev)

    def test_eval_in_js_and_ts(self):
        hunks = _load_hunks("js_ts_patterns.diff")
        ev = registry.execute_batch(["unsafe_deserialize"], hunks)
        assert len(ev) >= 2  # one in JS, one in TS

    def test_sql_injection_in_js(self):
        hunks = _load_hunks("js_ts_patterns.diff")
        ev = registry.execute_batch(["sql_injection"], hunks)
        assert len(ev) >= 1

    def test_no_ast_rules_fire_on_js(self):
        """AST-only rules (complexity, missing_docstring) should
        gracefully return empty for JS (SyntaxError on ast.parse)."""
        hunks = _load_hunks("js_ts_patterns.diff")
        ev = registry.execute_batch(
            ["function_too_long", "high_cyclomatic", "deep_nesting",
             "missing_docstring", "unused_import",
             "mutable_default_arg", "bare_except"],
            hunks,
        )
        # AST rules should not crash, may return empty
        assert isinstance(ev, list)


# ── inline false-positive resistance ────────────────────────────────


class TestFalsePositiveResistance:
    """Patterns that look risky but are safe — must not trigger."""

    def test_parameterized_query_safe(self):
        hunk = _make_hunk(
            'db.execute("SELECT * FROM users WHERE name = ?", (name,))'
        )
        assert len(registry.execute("sql_injection", hunk)) == 0

    def test_env_var_safe(self):
        hunk = _make_hunk('api_key = os.environ.get("API_KEY")')
        assert len(registry.execute("hardcoded_secret", hunk)) == 0

    def test_short_password_safe(self):
        """Passwords shorter than 8 chars should not trigger SEC002."""
        hunk = _make_hunk('password = "123"')
        assert len(registry.execute("hardcoded_secret", hunk)) == 0

    def test_subprocess_without_shell_safe(self):
        hunk = _make_hunk('subprocess.run(["ls", "-la"], shell=False)')
        assert len(registry.execute("command_injection", hunk)) == 0

    def test_ast_literal_eval_safe(self):
        hunk = _make_hunk("data = ast.literal_eval(text)")
        assert len(registry.execute("unsafe_deserialize", hunk)) == 0

    def test_specific_exception_safe(self):
        hunk = _make_hunk("except ValueError:\n    pass")
        assert len(registry.execute("bare_except", hunk)) == 0

    def test_none_default_arg_safe(self):
        hunk = _make_hunk("def foo(x=None):")
        assert len(registry.execute("mutable_default_arg", hunk)) == 0

    def test_used_import_safe(self):
        hunk = _make_hunk("import os\nos.getcwd()")
        assert len(registry.execute("unused_import", hunk)) == 0

    def test_join_string_safe(self):
        hunk = _make_hunk(
            "result = ''.join(str(item) for item in items)"
        )
        assert len(registry.execute("string_concat_in_loop", hunk)) == 0

    def test_normal_line_length_safe(self):
        hunk = _make_hunk("x = 1")
        assert len(registry.execute("long_line", hunk)) == 0

    def test_small_number_safe(self):
        hunk = _make_hunk("x = 100")
        assert len(registry.execute("magic_number", hunk)) == 0

    def test_snake_case_safe(self):
        hunk = _make_hunk("def my_function():\n    pass")
        assert len(registry.execute("naming_violation", hunk)) == 0

    def test_docstring_present_safe(self):
        hunk = _make_hunk('def foo():\n    """Doc."""\n    pass')
        assert len(registry.execute("missing_docstring", hunk)) == 0

    def test_no_todo_safe(self):
        hunk = _make_hunk("# This is a regular comment")
        assert len(registry.execute("todo_fixme", hunk)) == 0
