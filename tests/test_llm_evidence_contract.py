"""Tests for the llm_assisted response-parsing hardening (#11/#12/#13/#14)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rules.llm_assisted import _validate_llm_evidences


def _ok_item(**overrides):
    item = {
        "rule_id": "LLM001", "category": "bug_risk", "severity": "low",
        "message": "finding", "line_no": 3,
    }
    item.update(overrides)
    return item


# --- #11: a missing `evidences` key is a contract violation, not "empty" ---

def test_missing_evidences_key_rejected():
    with pytest.raises(ValueError, match="'evidences' key"):
        _validate_llm_evidences({"something_else": []})


def test_empty_evidences_list_is_valid():
    _validate_llm_evidences({"evidences": []})


# --- #12: incomplete evidence items are rejected ---

@pytest.mark.parametrize("overrides", [
    {"rule_id": None},
    {"rule_id": ""},
    {"rule_id": 7},
    {"message": ""},
    {"message": "   "},
    {"message": None},
])
def test_incomplete_items_rejected(overrides):
    with pytest.raises(ValueError):
        _validate_llm_evidences({"evidences": [_ok_item(**overrides)]})


def test_complete_item_passes():
    _validate_llm_evidences({"evidences": [_ok_item()]})


# --- #14: line_no must be a non-negative int ---

@pytest.mark.parametrize("line_no", ["12", -1, 1.5, None, True])
def test_bad_line_no_rejected(line_no):
    with pytest.raises(ValueError, match="line_no"):
        _validate_llm_evidences({"evidences": [_ok_item(line_no=line_no)]})


# --- #13: construction-side fallbacks tolerate TypeError too ---

def test_enum_fallback_survives_non_string_input():
    """A contract regression slipping past the validator (e.g. a list
    category) must degrade to a labeled fallback, not raise.

    Note: these str-mixin enums actually raise ValueError (not
    TypeError) for non-string inputs; the (ValueError, TypeError)
    catch is belt-and-suspenders against future enum changes.
    """
    from unittest.mock import MagicMock
    from src.llm.client import LLMClient
    from src.models import ChangeType, DiffLine, HunkInfo, RiskCategory, Severity
    from src.rules.llm_assisted import create_llm_assisted_rule

    llm = MagicMock(spec=LLMClient)
    # Validator bypassed: simulate a response that already passed the
    # retry loop but regressed (list category, string line_no).
    llm.chat_json.return_value = {"evidences": [{
        "rule_id": "LLM001", "category": ["security"], "severity": "medium",
        "message": "m", "line_no": "12",
    }]}
    rule = create_llm_assisted_rule(llm)
    hunk = HunkInfo(
        file_path="a.py", old_start=0, old_count=0, new_start=1, new_count=1,
        lines=[DiffLine(content="x = 1", change_type=ChangeType.ADDED)],
    )
    evidences = rule(hunk)
    assert len(evidences) == 1
    assert evidences[0].category == RiskCategory.BUG_RISK
    assert evidences[0].severity == Severity.MEDIUM
    assert evidences[0].line_range == (0, 0)  # "12" is not an int -> 0
