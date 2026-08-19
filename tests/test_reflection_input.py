import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    AgentState, Evidence, HunkInfo, RiskCategory, RiskItem, Severity,
)
from src.nodes.reflection import ReflectionNode


def _hunk(file_path: str, start: int = 5) -> HunkInfo:
    return HunkInfo(file_path=file_path, old_start=1, old_count=2,
                    new_start=start, new_count=3, lines=[])


def _evidence(file_path: str, source_type: str = "deterministic",
              source: str = "sql_injection") -> Evidence:
    return Evidence(
        source=source, rule_id="SEC001",
        category=RiskCategory.SECURITY, severity=Severity.HIGH,
        message="m", line_range=(1, 1), snippet="s",
        confidence=1.0, source_type=source_type, file_path=file_path,
    )


def _state(hunks, evidences, risks=None) -> AgentState:
    return AgentState(hunks=hunks, evidence_pool=evidences,
                      rules_executed=["sql_injection"],
                      risks=risks or [])


def _node_with(response) -> tuple[ReflectionNode, MagicMock]:
    llm = MagicMock()
    llm.chat_json.return_value = response
    return ReflectionNode(llm=llm), llm


def test_digest_shows_per_hunk_density_and_zero_evidence_gap():
    node, llm = _node_with({
        "needs_more_analysis": False, "additional_tools_needed": [],
        "reason": "covered", "coverage_assessment": "ok",
    })
    state = _state(
        hunks=[_hunk("a.py"), _hunk("b.py")],
        evidences=[_evidence("a.py"), _evidence("a.py")],
    )
    node(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "a.py hunk@5 (+3/-2): 2 evidences, 0 risks" in prompt
    assert "b.py hunk@5 (+3/-2): 0 evidences, 0 risks" in prompt
    assert "coverage gaps" in prompt


def test_digest_includes_failed_rules():
    node, llm = _node_with({
        "needs_more_analysis": False, "additional_tools_needed": [],
        "reason": "covered", "coverage_assessment": "ok",
    })
    state = _state(
        hunks=[_hunk("a.py")],
        evidences=[_evidence("a.py"), _evidence("a.py", source_type="error", source="n_plus_1_query")],
    )
    node(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "Failed rules (blind spots this round): n_plus_1_query on a.py" in prompt


def test_digest_counts_risks_per_hunk():
    node, llm = _node_with({
        "needs_more_analysis": False, "additional_tools_needed": [],
        "reason": "covered", "coverage_assessment": "ok",
    })
    risk = RiskItem(title="t", category=RiskCategory.SECURITY,
                    severity=Severity.HIGH, description="d",
                    evidence_chain=[], risk_score=0.8,
                    file_path="a.py", line_range=(1, 1))
    state = _state(hunks=[_hunk("a.py")], evidences=[_evidence("a.py")],
                   risks=[risk])
    node(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "1 evidences, 1 risks" in prompt


def test_no_digest_without_hunks():
    node, llm = _node_with({
        "needs_more_analysis": False, "additional_tools_needed": [],
        "reason": "covered", "coverage_assessment": "ok",
    })
    node(AgentState(hunks=[], evidence_pool=[_evidence("a.py")]))
    prompt = llm.chat_json.call_args[0][1]
    assert "Per-hunk coverage" not in prompt
