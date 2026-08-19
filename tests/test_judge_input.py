import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AgentPhase, AgentState, Evidence, HunkInfo, RiskCategory, Severity
from src.nodes.judge import JudgeNode, _codebase_is_fallback


def _evidence(idx: int, file_path: str = "db/queries.py") -> Evidence:
    return Evidence(
        source="sql_injection", rule_id="SEC001",
        category=RiskCategory.SECURITY, severity=Severity.HIGH,
        message=f"Potential SQL injection at line {idx}",
        line_range=(idx, idx), snippet="query = 'SELECT' + name",
        confidence=1.0, source_type="deterministic", file_path=file_path,
    )


def _hunk(file_path: str = "db/queries.py") -> HunkInfo:
    return HunkInfo(file_path=file_path, old_start=1, old_count=1,
                    new_start=5, new_count=3, lines=[])


def _state(evidences, hunks, codebase=None, history=None) -> AgentState:
    return AgentState(
        hunks=hunks, evidence_pool=evidences,
        rag_context={"codebase": codebase or {}, "history": history or []},
    )


def _judge_with(response) -> tuple[JudgeNode, MagicMock]:
    llm = MagicMock()
    llm.chat_json.return_value = response
    rag = MagicMock()
    rag.search_security.return_value = [
        {"rule_id": "SEC001", "title": "SQL Injection Prevention",
         "content": "long narrative " * 50,
         "best_practice": "Use parameterized queries."}
    ]
    return JudgeNode(llm=llm, rag=rag), llm


def test_risk_without_evidence_refs_is_discarded():
    ev = [_evidence(1)]
    judge, _ = _judge_with({
        "risks": [
            {"title": "Real risk", "evidence_refs": [0], "category": "security",
             "severity": "high", "description": "d", "suggestion": "s",
             "file_path": "db/queries.py", "line_range": [1, 1], "risk_score": 0.9},
            {"title": "Hallucinated risk (no refs)", "evidence_refs": [],
             "category": "security", "severity": "high", "description": "d",
             "suggestion": "s", "risk_score": 0.8},
            {"title": "Out-of-range refs", "evidence_refs": [99],
             "category": "security", "severity": "high", "description": "d",
             "suggestion": "s", "risk_score": 0.8},
        ],
        "overall_risk_score": 0.8,
    })
    result = judge(_state(ev, [_hunk()]))
    assert len(result["risks"]) == 1
    assert result["risks"][0].title == "Real risk"
    assert len(result["risks"][0].evidence_chain) == 1


def test_prompt_contains_evidence_refs_constraint():
    judge, llm = _judge_with({"risks": [], "overall_risk_score": 0.0})
    judge(_state([_evidence(1)], [_hunk()]))
    prompt = llm.chat_json.call_args[0][1]
    assert "MUST reference at least one evidence index" in prompt


def test_security_knowledge_slimmed_in_prompt():
    judge, llm = _judge_with({"risks": [], "overall_risk_score": 0.0})
    judge(_state([_evidence(1)], [_hunk()]))
    prompt = llm.chat_json.call_args[0][1]
    assert "long narrative" not in prompt
    assert "SQL Injection Prevention" in prompt
    assert "Use parameterized queries." in prompt


def test_history_slimmed_in_prompt():
    judge, llm = _judge_with({"risks": [], "overall_risk_score": 0.0})
    history = [{
        "file_path": "db/queries.py", "diff_summary": "long summary " * 20,
        "risk_titles": ["SQL injection", "Hardcoded secret"],
        "risk_categories": ["security"],
    }]
    judge(_state([_evidence(1)], [_hunk()], history=history))
    prompt = llm.chat_json.call_args[0][1]
    assert "long summary" not in prompt
    assert "SQL injection" in prompt


def test_snippet_kept_when_codebase_missing():
    judge, llm = _judge_with({"risks": [], "overall_risk_score": 0.0})
    judge(_state([_evidence(1)], [_hunk()], codebase={}))
    prompt = llm.chat_json.call_args[0][1]
    assert "SELECT" in prompt  # snippet retained


def test_snippet_dropped_when_symbols_selected():
    judge, llm = _judge_with({"risks": [], "overall_risk_score": 0.0})
    codebase = {"db/queries.py": [{
        "file_path": "db/queries.py", "symbol_name": "get_user",
        "symbol_type": "function", "line_range": "1-20",
        "content": "def get_user(): ...", "source": "diff_file",
    }]}
    judge(_state([_evidence(6)], [_hunk()], codebase=codebase))
    prompt = llm.chat_json.call_args[0][1]
    assert "SELECT" not in prompt  # snippet stripped
    assert "get_user" in prompt   # symbol source present


def test_codebase_is_fallback_detection():
    codebase = {"db/queries.py": [{
        "file_path": "db/queries.py", "symbol_name": "get_user",
        "symbol_type": "function", "line_range": "1-20",
        "content": "def get_user(): ...", "source": "diff_file",
    }]}
    hunk_overlap = HunkInfo(file_path="db/queries.py", old_start=1,
                            old_count=1, new_start=5, new_count=3, lines=[])
    hunk_far = HunkInfo(file_path="db/queries.py", old_start=1,
                        old_count=1, new_start=999, new_count=1, lines=[])
    assert _codebase_is_fallback([hunk_overlap], codebase, {}) is False
    assert _codebase_is_fallback([hunk_far], codebase, {}) is True
    assert _codebase_is_fallback([hunk_far], {}, {}) is False  # no symbols at all
