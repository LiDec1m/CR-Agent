import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    AgentState, ChangeType, DiffLine, HunkInfo, RiskItem, RuleOutcome,
    RuleOutcomeStatus,
)
from src.nodes.judge import JudgeNode, _validate_judge_response
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
        "plan_by_hunk": {
            "test.py:1": {
                "tools": ["sql_injection", "hardcoded_secret"],
                "reason": "sql + secret keywords",
            },
        },
    }
    rag = MagicMock()
    rag.search_history.return_value = []

    result = PlannerNode(llm, rag)(AgentState(hunks=[_make_hunk("x = 1")]))

    assert result["pending_tools_by_hunk"] == {
        "test.py:1": ["sql_injection", "hardcoded_secret"],
    }
    assert result["planning_reasons"] == {"test.py:1": "sql + secret keywords"}
    assert result["phase"].value == "tool_routing"


def test_planner_drops_malformed_or_unknown_hunk_assignments():
    rag = MagicMock()
    rag.search_history.return_value = []
    node = PlannerNode(MagicMock(), rag)
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
    # executed_tools_by_hunk was removed; rule_outcomes is the only ledger.
    assert [(o.rule, o.status) for o in result["rule_outcomes"]] == [
        ("command_injection", RuleOutcomeStatus.EVIDENCE_PRODUCED)
    ]


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
        "reason": "Coverage insufficient",
    }
    result = ReflectionNode(llm)(AgentState(hunks=[_make_hunk("x = 1")]))
    assert result["needs_more_analysis"] is True
    assert result["pending_tools_by_hunk"] == {"test.py:1": ["deep_nesting"]}


def test_reflection_does_not_rerun_completed_assignment():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": True,
        "additional_tools_by_hunk": {"test.py:1": ["unused_import"]},
        "reason": "More",
    }
    state = AgentState(
        hunks=[_make_hunk("x = 1")],
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
        "reason": "Coverage incomplete",
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
        rule_outcomes=[RuleOutcome(hunk_key="app.py:1", rule="sql_injection",
                                   status=RuleOutcomeStatus.CLEAN)],
    )
    report = ReporterNode()(state)["report"]
    assert report.rules_executed == ["sql_injection"]
    assert report.coverage_limited_hunks == 0
    # conclusively_examined_hunks was removed; conclusive coverage is now
    # derived inline as total_hunks - coverage_limited_hunks.
    assert report.total_hunks - report.coverage_limited_hunks == 1


def test_reporter_empty_state_zero_score():
    report = ReporterNode()(AgentState())["report"]
    assert report.overall_risk_score == 0.0
    assert report.summary == "No significant risks detected."


def test_reflection_digest_includes_planning_reason():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": False, "additional_tools_by_hunk": {},
        "reason": "done",
    }
    state = AgentState(
        hunks=[_make_hunk("x = 1", "app.py", 5)],
        planning_reasons={"app.py:5": "untrusted input to subprocess"},
    )
    ReflectionNode(llm)(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "planned: untrusted input to subprocess" in prompt


def test_reflection_finalize_phase_is_reporting():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": False, "additional_tools_by_hunk": {},
        "reason": "converged",
    }
    result = ReflectionNode(llm)(AgentState(hunks=[_make_hunk("x = 1")]))
    assert result["needs_more_analysis"] is False
    assert result["phase"].value == "reporting"


def test_reflection_round_cap_phase_is_reporting():
    llm = MagicMock()
    result = ReflectionNode(llm, max_rounds=2)(
        AgentState(hunks=[_make_hunk("x = 1")], reflection_round=2)
    )
    assert result["needs_more_analysis"] is False
    assert result["phase"].value == "reporting"


def test_judge_severity_validator_accepts_valid_rejects_invalid():
    # Valid severities pass without raising.
    _validate_judge_response({"risks": [{"severity": "high"}]})
    _validate_judge_response({"risks": []})
    # Unknown severity raises ValueError (drives the chat_json repair loop).
    try:
        _validate_judge_response({"risks": [{"severity": "bogus"}]})
        assert False, "expected ValueError"
    except ValueError:
        pass
    # Missing severity also raises (contract requires it).
    try:
        _validate_judge_response({"risks": [{}]})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_judge_feedback_precedents_injected_into_prompt(tmp_path):
    from src.memory.long_term import LongTermMemory
    from src.models import Evidence, RiskCategory, Severity

    db = str(tmp_path / "fb.db")
    ltm = LongTermMemory(db)
    ltm.init_tables()
    # feedback_content names the symbol under review; file_pattern matches
    # the evidence file_path by equality so the SQL filter keeps the row.
    ltm.add_feedback("t1", "app.py", "SEC003", "false_positive",
                      "do not flag subprocess.run as injection when input is hardcoded")
    # A 'missing' row must be excluded by the recall contract.
    ltm.add_feedback("t2", "app.py", "SEC003", "missing",
                      "should have caught eval here")

    llm = MagicMock()
    llm.chat_json.return_value = {"risks": [], "dismissed_evidence": []}
    rag = MagicMock()
    rag.search_security.return_value = []
    # Codebase context: one diff-file symbol overlapping the evidence line
    # range, so _evidence_symbols resolves the symbol name; its name appears
    # in the feedback_content so FTS matches and the row is recalled.
    state = AgentState(
        hunks=[_make_hunk("def run():\n    subprocess.run(x)", "app.py", 1)],
        evidence_pool=[Evidence(
            source="command_injection", rule_id="SEC003",
            category=RiskCategory.SECURITY, severity=Severity.HIGH,
            message="subprocess.run", line_range=(1, 2), file_path="app.py",
        )],
        rag_context={"codebase": {"app.py": [{
            "file_path": "app.py", "symbol_name": "run",
            "symbol_type": "function", "line_range": "1-2",
            "source": "diff_file", "content": "def run():\n    subprocess.run(x)",
        }]}},
    )
    JudgeNode(llm, rag, ltm=ltm)(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "do not flag subprocess.run" in prompt
    assert "should have caught eval here" not in prompt  # 'missing' excluded


def test_judge_no_ltm_skips_feedback_recall():
    llm = MagicMock()
    llm.chat_json.return_value = {"risks": [], "dismissed_evidence": []}
    rag = MagicMock()
    rag.search_security.return_value = []
    result = JudgeNode(llm, rag)(AgentState(evidence_pool=[]))
    # Empty pool: no batch to adjudicate, so the judge must not spend an
    # LLM call at all (previously it judged a prompt with zero evidence).
    llm.chat_json.assert_not_called()
    assert result["risks"] == []
    assert result["judge_unadjudicated_evidence"] == 0
