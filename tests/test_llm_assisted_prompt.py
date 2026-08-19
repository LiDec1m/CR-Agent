import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChangeType, DiffLine, HunkInfo
from src.rules.llm_assisted import create_llm_assisted_rule


def _make_hunk(added: str, removed: str = "") -> HunkInfo:
    lines = [
        DiffLine(content=c, change_type=ChangeType.ADDED, new_line_no=i + 1)
        for i, c in enumerate(added.splitlines()) if c
    ] + [
        DiffLine(content=c, change_type=ChangeType.REMOVED, old_line_no=100 + i)
        for i, c in enumerate(removed.splitlines()) if c
    ]
    return HunkInfo(
        file_path="app/views.py",
        old_start=100, old_count=len(removed.splitlines()),
        new_start=1, new_count=len(added.splitlines()),
        lines=lines,
    )


def test_prompt_includes_removed_code():
    llm = MagicMock()
    llm.chat_json.return_value = {"evidences": []}
    rule = create_llm_assisted_rule(llm)

    hunk = _make_hunk(
        added="def delete_user(uid):\n    db.delete(uid)",
        removed="    assert request.user.is_admin\n    verify_permission(uid)",
    )
    rule(hunk)

    prompt = llm.chat_json.call_args[0][1]
    assert "Removed code (old lines):" in prompt
    assert "assert request.user.is_admin" in prompt
    assert "deletion drops a validation" in prompt


def test_prompt_omits_removed_section_when_no_removals():
    llm = MagicMock()
    llm.chat_json.return_value = {"evidences": []}
    rule = create_llm_assisted_rule(llm)

    hunk = _make_hunk(added="x = 1")
    rule(hunk)

    prompt = llm.chat_json.call_args[0][1]
    assert "Removed code" not in prompt


def test_removal_only_hunk_still_analyzed():
    llm = MagicMock()
    llm.chat_json.return_value = {"evidences": []}
    rule = create_llm_assisted_rule(llm)

    hunk = _make_hunk(added="", removed="    sanitize(raw)")
    rule(hunk)

    assert llm.chat_json.called
    prompt = llm.chat_json.call_args[0][1]
    assert "Removed code (old lines):" in prompt
