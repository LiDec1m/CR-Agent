import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    AgentState, ChangeType, DiffLine, HunkInfo, RiskItem, RuleOutcome,
    RuleOutcomeStatus,
)
from src.nodes.judge import JudgeNode
from src.nodes.planner import PlannerNode
from src.nodes.reflection import ReflectionNode
from src.nodes.reporter import ReporterNode
from src.nodes.tool_router import ToolRouterNode
from src.rules import registry as _  # trigger registration


def _make_hunk(code, file_path="test.py", start=1):
    lines = [
        DiffLine(content=line, change_type=ChangeType.ADDED, new_line_no=start + index)
        for index, line in enumerate(code.splitlines())
    ]
    return HunkInfo(file_path=file_path, old_start=start, old_count=0,
                    new_start=start, new_count=len(lines), lines=lines)


def test_planner_node_returns_hunk_keyed_plan():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "plan_by_hunk": {"test.py:1": ["sql_injection", "hardcoded_secret"]},
    }
    rag, ltm = MagicMock(), MagicMock()
    rag.search_history.return_value = []
    ltm.get_feedback.return_value = []

    result = PlannerNode(llm, rag, ltm)(AgentState(hunks=[_make_hunk("x = 1")]))

    assert result["pending_tools_by_hunk"] == {
        "test.py:1": ["sql_injection", "hardcoded_secret"],
    }
    assert result["phase"].value == "tool_routing"


def test_planner_drops_malformed_or_unknown_hunk_assignments():
    rag, ltm = MagicMock(), MagicMock()
    rag.search_history.return_value = []
    ltm.get_feedback.return_value = []
    node = PlannerNode(MagicMock(), rag, ltm)
    state = AgentState(hunks=[_make_hunk("x = 1")])
    cases = [
        ({"plan_by_hunk": None}, {}),
        ({"plan_by_hunk": {"wrong.py:1": ["sql_injection"]}}, {}),
        ({"plan_by_hunk": {"test.py:1": ["sql_injection", 42, "long_line"]}},
         {"test.py:1": ["sql_injection", "long_line"]}),
    ]
    for response, expected in cases:
        node.llm.chat_json.return_value = response
        assert node(state)["pending_tools_by_hunk"] == expected


def test_tool_router_node_records_execution_per_hunk():
    rag = MagicMock()
    rag.search_codebase.return_value = []
    state = AgentState(
        hunks=[_make_hunk('os.system("rm -rf /")')],
        pending_tools_by_hunk={"test.py:1": ["command_injection"]},
    )
    result = ToolRouterNode(_, rag)(state)
    assert result["evidence_pool"]
    assert result["executed_tools_by_hunk"] == {"test.py:1": ["command_injection"]}
    assert result["rule_outcomes"][0].status is RuleOutcomeStatus.EVIDENCE_PRODUCED


def test_judge_node():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "risks": [{"title": "Command Injection", "category": "security",
                   "severity": "high", "description": "d", "evidence_refs": [0],
                   "suggestion": "s", "file_path": "test.py", "line_range": [1, 1],
                   "risk_score": 0.9}],
    }
    rag = MagicMock()
    rag.search_security.return_value = []
    from src.models import Evidence, RiskCategory, Severity
    state = AgentState(evidence_pool=[Evidence(
        source="command_injection", rule_id="SEC003", category=RiskCategory.SECURITY,
        severity=Severity.HIGH, message="os.system()", line_range=(1, 1),
    )])
    result = JudgeNode(llm, rag)(state)
    assert len(result["risks"]) == 1


def test_reflection_schedules_only_new_hunk_rule_assignment():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": True,
        "additional_tools_by_hunk": {"test.py:1": ["deep_nesting"]},
        "reason": "Coverage insufficient", "coverage_assessment": "partial",
    }
    result = ReflectionNode(llm)(AgentState(hunks=[_make_hunk("x = 1")]))
    assert result["needs_more_analysis"] is True
    assert result["pending_tools_by_hunk"] == {"test.py:1": ["deep_nesting"]}


def test_reflection_does_not_rerun_completed_assignment():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": True,
        "additional_tools_by_hunk": {"test.py:1": ["unused_import"]},
        "reason": "More", "coverage_assessment": "partial",
    }
    state = AgentState(
        hunks=[_make_hunk("x = 1")],
        executed_tools_by_hunk={"test.py:1": ["unused_import"]},
        rule_outcomes=[RuleOutcome(hunk_key="test.py:1", rule="unused_import",
                                   status=RuleOutcomeStatus.CLEAN)],
    )
    result = ReflectionNode(llm)(state)
    assert result["needs_more_analysis"] is False
    assert "Finalizing" in result["reflection_notes"][-1]


def test_reflection_final_round_preserves_new_work_observation():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": True,
        "additional_tools_by_hunk": {"test.py:1": ["unused_import"]},
        "reason": "Coverage incomplete", "coverage_assessment": "partial",
    }
    result = ReflectionNode(llm, max_rounds=3)(
        AgentState(hunks=[_make_hunk("x = 1")], reflection_round=2)
    )
    assert result["needs_more_analysis"] is True


def test_reporter_builds_derived_rule_list_and_coverage():
    hunk = _make_hunk("def f():\n    pass", "app.py")
    risk = RiskItem(title="Test risk", category="security", severity="critical",
                    description="d", evidence_chain=[], file_path="app.py", risk_score=0.9)
    state = AgentState(
        repo="r", commit_sha="abc", hunks=[hunk], risks=[risk], reflection_round=3,
        executed_tools_by_hunk={"app.py:1": ["sql_injection"]},
        rule_outcomes=[RuleOutcome(hunk_key="app.py:1", rule="sql_injection",
                                   status=RuleOutcomeStatus.CLEAN)],
    )
    report = ReporterNode()(state)["report"]
    assert report.rules_executed == ["sql_injection"]
    assert report.conclusively_examined_hunks == 1
    assert report.coverage_limited_hunks == 0


def test_reporter_empty_state_zero_score():
    report = ReporterNode()(AgentState())["report"]
    assert report.overall_risk_score == 0.0
    assert report.summary == "No significant risks detected."
