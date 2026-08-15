import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AgentState, ChangeType, DiffLine, HunkInfo, RiskItem
from src.nodes.planner import PlannerNode
from src.nodes.tool_router import ToolRouterNode
from src.nodes.judge import JudgeNode
from src.nodes.reflection import ReflectionNode
from src.rules import registry as _  # trigger registration


def _make_hunk(code, file_path="test.py"):
    lines = [
        DiffLine(content=c, change_type=ChangeType.ADDED, new_line_no=i + 1)
        for i, c in enumerate(code.splitlines())
    ]
    return HunkInfo(file_path=file_path, old_start=1, old_count=0,
                    new_start=1, new_count=len(lines), lines=lines)


def test_planner_node():
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "summary": "Modified login function",
        "plan": ["sql_injection", "hardcoded_secret"],
        "risk_areas": ["login.py"],
    })
    mock_rag = MagicMock()
    mock_rag.search_history.return_value = []
    mock_ltm = MagicMock()
    mock_ltm.get_feedback.return_value = []
    node = PlannerNode(mock_llm, mock_rag, mock_ltm)
    state = AgentState(
        hunks=[_make_hunk('query = "SELECT * FROM users WHERE name=\'" + username')],
    )
    result = node(state)
    assert "pending_tools" in result
    assert "sql_injection" in result["pending_tools"]
    assert result["phase"].value == "tool_routing"


def test_tool_router_node():
    mock_rag = MagicMock()
    mock_rag.search_codebase.return_value = []
    from src.rules import registry
    node = ToolRouterNode(registry, mock_rag)
    state = AgentState(
        hunks=[_make_hunk('os.system("rm -rf /")')],
        pending_tools=["command_injection"],
    )
    result = node(state)
    assert "evidence_pool" in result
    assert len(result["evidence_pool"]) >= 1
    assert "command_injection" in result["rules_executed"]


def test_judge_node():
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "risks": [{
            "title": "Command Injection",
            "category": "security",
            "severity": "high",
            "description": "os.system with user input",
            "evidence_refs": [0],
            "suggestion": "Use subprocess with shell=False",
            "file_path": "test.py",
            "line_range": [1, 1],
            "risk_score": 0.9,
        }],
        "overall_risk_score": 0.9,
    })
    mock_rag = MagicMock()
    mock_rag.search_security.return_value = []
    node = JudgeNode(mock_llm, mock_rag)
    from src.models import Evidence, RiskCategory, Severity
    state = AgentState(
        evidence_pool=[
            Evidence(
                source="command_injection", rule_id="SEC003",
                category=RiskCategory.SECURITY, severity=Severity.HIGH,
                message="os.system() at line 1", line_range=(1, 1),
            )
        ],
    )
    result = node(state)
    assert "risks" in result
    assert len(result["risks"]) == 1
    assert result["risks"][0].title == "Command Injection"
    assert len(result["risks"][0].evidence_chain) == 1


def test_reflection_node_needs_more():
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": True,
        "additional_tools_needed": ["deep_nesting"],
        "reason": "Coverage insufficient",
        "coverage_assessment": "60%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=0)
    result = node(state)
    assert result["needs_more_analysis"] is True
    assert "deep_nesting" in result["pending_tools"]
    assert result["reflection_round"] == 1


def test_reflection_node_done():
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": False,
        "additional_tools_needed": [],
        "reason": "Coverage sufficient",
        "coverage_assessment": "95%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=0)
    result = node(state)
    assert result["needs_more_analysis"] is False
    assert result["phase"].value == "done"


def test_reflection_node_max_rounds():
    mock_llm = MagicMock()
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=3)
    result = node(state)
    assert result["needs_more_analysis"] is False
    assert result["phase"].value == "done"
    mock_llm.chat.assert_not_called()


def test_reflection_final_round_preserves_observation():
    """At the final allowed round, the LLM's true verdict is preserved
    (needs_more stays True) for observability; the routing condition
    sends the graph to reporter regardless, so the report is still
    produced. Anti-idle: only if NEW rules are suggested."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": True,
        "additional_tools_needed": ["unused_import"],  # not yet executed
        "reason": "Coverage still incomplete",
        "coverage_assessment": "60%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=2)  # next round = 3 == max_rounds
    result = node(state)
    assert result["needs_more_analysis"] is True  # preserved, not overridden
    assert "report" not in result  # reporter builds it downstream


def test_reflection_non_final_round_still_loops():
    """Before the final round, a 'need more' verdict with NEW rules
    must still loop."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": True,
        "additional_tools_needed": ["unused_import"],
        "reason": "More",
        "coverage_assessment": "40%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=0)  # next round = 1 < 3
    result = node(state)
    assert result["needs_more_analysis"] is True
    assert "report" not in result


def test_reflection_no_new_rules_finalizes_instead_of_idling():
    """Anti-idle: if every suggested rule was already executed (or is
    unknown), looping back would collect zero new evidence — the node
    must finalize even on a non-final round."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": True,
        "additional_tools_needed": ["sql_injection", "not_a_rule"],
        "reason": "More",
        "coverage_assessment": "40%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(
        reflection_round=0,
        rules_executed=["sql_injection"],
    )
    result = node(state)
    assert result["needs_more_analysis"] is False
    assert "no new rules" in result["reflection_notes"][-1]


def test_reflection_filters_already_executed_from_suggestions():
    """Suggested rules that already ran are dropped from the returned
    additional_tools_needed so tool_router only executes new ones."""
    mock_llm = MagicMock()
    mock_llm.chat_json.return_value = dict({
        "needs_more_analysis": True,
        "additional_tools_needed": ["sql_injection", "magic_number"],
        "reason": "More",
        "coverage_assessment": "40%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(
        reflection_round=0,
        rules_executed=["sql_injection"],
    )
    result = node(state)
    assert result["needs_more_analysis"] is True
    assert result["pending_tools"] == ["magic_number"]


# ------------------------------------------------------------------
# ReporterNode tests
# ------------------------------------------------------------------

def test_reporter_builds_report_from_state():
    from src.nodes.reporter import ReporterNode

    hunk = HunkInfo(
        file_path="app.py", old_start=1, old_count=0,
        new_start=1, new_count=2,
        lines=[
            DiffLine(content="def f():", change_type=ChangeType.ADDED, new_line_no=1),
            DiffLine(content="    pass", change_type=ChangeType.ADDED, new_line_no=2),
        ],
    )
    risk = RiskItem(
        title="Test risk", category="security", severity="critical",
        description="d", evidence_refs=[], suggestion="s",
        file_path="app.py", risk_score=0.9,
    )
    state = AgentState(
        repo="r", commit_sha="abc", hunks=[hunk],
        risks=[risk], rules_executed=["sql_injection"],
        reflection_round=3, reflection_notes=["Round 1: x"],
        long_term_feedback=["fb"],
    )
    result = ReporterNode()(state)
    report = result["report"]
    assert report is not None
    assert report.repo == "r"
    assert report.commit_sha == "abc"
    assert report.overall_risk_score == 0.9
    assert report.files_scanned == ["app.py"]
    assert report.total_hunks == 1
    assert report.rules_executed == ["sql_injection"]
    assert report.reflection_rounds == 3
    assert report.long_term_feedback_applied == ["fb"]
    # needs_more_analysis is NOT overwritten by reporter — it stays as
    # an observability signal in state (default False here).
    assert "needs_more_analysis" not in result
    assert result["phase"].value == "done"


def test_reporter_empty_state_zero_score():
    from src.nodes.reporter import ReporterNode

    result = ReporterNode()(AgentState())
    report = result["report"]
    assert report is not None
    assert report.overall_risk_score == 0.0
    assert report.summary == "No significant risks detected."
    assert report.risks == []
