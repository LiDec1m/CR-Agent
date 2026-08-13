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
