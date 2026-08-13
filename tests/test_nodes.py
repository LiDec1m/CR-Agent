import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AgentState, ChangeType, DiffLine, HunkInfo
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
    mock_llm.chat.return_value = json.dumps({
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
    assert "plan" in result
    assert "sql_injection" in result["plan"]
    assert result["phase"].value == "tool_routing"


def test_tool_router_node():
    mock_rag = MagicMock()
    mock_rag.search_codebase.return_value = []
    from src.rules import registry
    node = ToolRouterNode(registry, mock_rag)
    state = AgentState(
        hunks=[_make_hunk('os.system("rm -rf /")')],
        plan=["command_injection"],
    )
    result = node(state)
    assert "evidence_pool" in result
    assert len(result["evidence_pool"]) >= 1
    assert "command_injection" in result["rules_executed"]


def test_judge_node():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = json.dumps({
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
    mock_llm.chat.return_value = json.dumps({
        "needs_more_analysis": True,
        "additional_tools_needed": ["deep_nesting"],
        "reason": "Coverage insufficient",
        "coverage_assessment": "60%",
    })
    node = ReflectionNode(mock_llm, max_rounds=3)
    state = AgentState(reflection_round=0)
    result = node(state)
    assert result["needs_more_analysis"] is True
    assert "deep_nesting" in result["additional_tools_needed"]
    assert result["reflection_round"] == 1


def test_reflection_node_done():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = json.dumps({
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
