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


def test_function_too_long():
    code = "\n".join(["    x = 1"] * 55)
    hunk = _make_hunk(f"def long_func():\n{code}")
    results = registry.execute("function_too_long", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "CX001"


def test_high_cyclomatic():
    code = "def f():\n" + "\n".join(["    if x: pass"] * 12)
    hunk = _make_hunk(code)
    results = registry.execute("high_cyclomatic", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "CX002"


def test_deep_nesting():
    code = "def f():\n"
    for i in range(5):
        code += "    " * (i + 1) + "if x:\n"
    code += "    " * 6 + "pass"
    hunk = _make_hunk(code)
    results = registry.execute("deep_nesting", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "CX003"


def test_bare_except():
    hunk = _make_hunk("except:\n    pass")
    results = registry.execute("bare_except", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "BUG001"


def test_mutable_default_arg():
    hunk = _make_hunk("def foo(x=[]):")
    results = registry.execute("mutable_default_arg", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "BUG002"


def test_unused_import():
    hunk = _make_hunk("import os\nimport sys\nx = 1")
    results = registry.execute("unused_import", hunk)
    assert len(results) >= 1


def test_magic_number():
    hunk = _make_hunk("timeout = 86400")
    results = registry.execute("magic_number", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "STY002"


def test_long_line():
    hunk = _make_hunk("x = " + "a" * 130)
    results = registry.execute("long_line", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "STY003"


def test_io_in_loop():
    hunk = _make_hunk("for item in items:\n    with open('f.txt') as f:\n        f.write(item)")
    results = registry.execute("io_in_loop", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "PERF001"


def test_n_plus_1_query():
    hunk = _make_hunk("for user in users:\n    db.execute('SELECT * FROM profile WHERE user=?', user)")
    results = registry.execute("n_plus_1_query", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "PERF002"


def test_string_concat_in_loop():
    hunk = _make_hunk("for item in items:\n    data += str(item)")
    results = registry.execute("string_concat_in_loop", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "PERF003"


def test_missing_docstring():
    hunk = _make_hunk("def foo():\n    pass")
    results = registry.execute("missing_docstring", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "MAIN001"


def test_todo_fixme():
    hunk = _make_hunk("# TODO: fix this later")
    results = registry.execute("todo_fixme", hunk)
    assert len(results) >= 1
    assert results[0].rule_id == "MAIN003"


def test_all_21_rules_registered():
    names = registry.list_all()
    assert len(names) == 21
    assert "removed_security_check" in names
