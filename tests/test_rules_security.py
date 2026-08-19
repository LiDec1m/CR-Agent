import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChangeType, DiffLine, HunkInfo
from src.rules import registry


def _make_hunk(added_code: str, file_path: str = "test.py") -> HunkInfo:
    lines = [
        DiffLine(content=code, change_type=ChangeType.ADDED, new_line_no=i + 1)
        for i, code in enumerate(added_code.splitlines())
    ]
    return HunkInfo(
        file_path=file_path, old_start=1, old_count=0, new_start=1,
        new_count=len(lines), lines=lines,
    )


def test_sql_injection_detects_string_concat():
    hunk = _make_hunk('query = "SELECT * FROM users WHERE name=\'" + username + "\'"')
    results = registry.execute("sql_injection", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "SEC001"


def test_sql_injection_no_false_positive_on_parameterized():
    hunk = _make_hunk('db.execute("SELECT * FROM users WHERE name = ?", (name,))')
    results = registry.execute("sql_injection", hunk)
    assert len(results) == 0


def test_hardcoded_secret_detects_api_key():
    hunk = _make_hunk('API_KEY = "sk-1234567890abcdef"')
    results = registry.execute("hardcoded_secret", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "SEC002"


def test_hardcoded_secret_no_false_positive_on_env():
    hunk = _make_hunk('api_key = os.environ.get("API_KEY")')
    results = registry.execute("hardcoded_secret", hunk)
    assert len(results) == 0


def test_command_injection_detects_os_system():
    hunk = _make_hunk('os.system("echo " + username)')
    results = registry.execute("command_injection", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "SEC003"


def test_command_injection_detects_shell_true():
    hunk = _make_hunk('subprocess.run(cmd, shell=True)')
    results = registry.execute("command_injection", hunk)
    assert len(results) >= 1


def test_unsafe_deserialize_detects_eval():
    hunk = _make_hunk("result = eval(user_input)")
    results = registry.execute("unsafe_deserialize", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "SEC004"


def test_unsafe_deserialize_detects_pickle():
    hunk = _make_hunk("data = pickle.loads(raw_data)")
    results = registry.execute("unsafe_deserialize", hunk)
    assert len(results) >= 1


def test_registry_list_all_includes_security():
    names = registry.list_all()
    assert "sql_injection" in names
    assert "hardcoded_secret" in names
    assert "command_injection" in names
    assert "unsafe_deserialize" in names


def test_registry_execute_batch():
    hunk = _make_hunk('eval(input("SQL: " + name))')
    results = registry.execute_batch(
        ["sql_injection", "unsafe_deserialize"], [hunk]
    )
    assert len(results) >= 2


def _make_removed_hunk(removed_code: str, file_path: str = "test.py") -> HunkInfo:
    lines = [
        DiffLine(content=code, change_type=ChangeType.REMOVED, old_line_no=i + 1)
        for i, code in enumerate(removed_code.splitlines())
    ]
    return HunkInfo(
        file_path=file_path, old_start=1, old_count=len(lines), new_start=1,
        new_count=0, lines=lines,
    )


def test_removed_security_check_flags_deleted_assert():
    hunk = _make_removed_hunk("    assert user.is_active, 'inactive user'")
    results = registry.execute("removed_security_check", hunk)
    assert len(results) == 1
    ev = results[0]
    assert ev.rule_id == "SEC005"
    assert ev.confidence == 0.6
    assert ev.source_type == "deterministic"
    assert "old line 1" in ev.message


def test_removed_security_check_flags_deleted_validation_call():
    hunk = _make_removed_hunk("    sanitize_input(raw)")
    results = registry.execute("removed_security_check", hunk)
    assert len(results) == 1
    assert "validation" in results[0].message


def test_removed_security_check_flags_finally_and_release():
    hunk = _make_removed_hunk("finally:\n    lock.release()")
    results = registry.execute("removed_security_check", hunk)
    # one evidence per removed line (break after first pattern hit)
    assert len(results) == 2


def test_removed_security_check_ignores_benign_removal():
    hunk = _make_removed_hunk("    print('debug info')\n    x = 1")
    results = registry.execute("removed_security_check", hunk)
    assert len(results) == 0


def test_removed_security_check_no_trigger_on_added_lines():
    # guard patterns on ADDED lines must not trigger the removal rule
    hunk = _make_hunk("    assert user.is_active")
    results = registry.execute("removed_security_check", hunk)
    assert len(results) == 0
