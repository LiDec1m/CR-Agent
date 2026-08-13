import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    AgentState, ChangeType, DiffLine, Evidence, HunkInfo,
    RiskCategory, RiskItem, RiskReport, Severity,
)


def test_diffline_creation():
    line = DiffLine(content="x = 1", change_type=ChangeType.ADDED, new_line_no=10)
    assert line.content == "x = 1"
    assert line.change_type == ChangeType.ADDED
    assert line.old_line_no is None


def test_hunkinfo_added_lines():
    hunk = HunkInfo(
        file_path="foo.py", old_start=1, old_count=2, new_start=1, new_count=3,
        lines=[
            DiffLine(content="x = 1", change_type=ChangeType.CONTEXT),
            DiffLine(content="y = 2", change_type=ChangeType.ADDED, new_line_no=2),
            DiffLine(content="z = 3", change_type=ChangeType.ADDED, new_line_no=3),
        ],
    )
    assert len(hunk.added_lines) == 2
    assert hunk.added_code == "y = 2\nz = 3"
    assert hunk.language == "python"


def test_hunkinfo_language_js():
    hunk = HunkInfo(
        file_path="app/index.ts", old_start=1, old_count=1, new_start=1, new_count=1,
    )
    assert hunk.language == "typescript"


def test_evidence_defaults():
    ev = Evidence(
        source="sql_injection", category=RiskCategory.SECURITY,
        severity=Severity.HIGH, message="SQL injection risk",
    )
    assert ev.confidence == 1.0
    assert ev.source_type == "deterministic"


def test_risk_report_to_text():
    report = RiskReport(
        repo="myrepo", summary="No major risks",
        files_scanned=["a.py"], total_hunks=1, reflection_rounds=0,
    )
    text = report.to_text()
    assert "Code Change Risk Report" in text
    assert "No significant risks" in text


def test_agent_state_defaults():
    state = AgentState(raw_diff="diff --git")
    assert state.hunks == []
    assert state.reflection_round == 0
    assert state.report is None
    assert state.rag_context == {}
