"""Tests for pipeline-honesty fixes: fail-fast, degraded marking,
evidence hunk-attribution dedup and judge batching."""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    AgentState, Evidence, RiskCategory, RiskItem, Severity,
)


def _evidence(idx, file_path="a.py", message=None):
    return Evidence(
        source=f"rule_{idx}", rule_id=f"SEC{idx:03d}",
        category=RiskCategory.SECURITY, severity=Severity.HIGH,
        message=message or f"finding {idx}",
        line_range=(idx, idx), file_path=file_path,
    )


def _hunk(file_path="a.py", new_start=1, code=None):
    from src.models import DiffLine, HunkInfo
    from src.models import ChangeType
    return HunkInfo(
        file_path=file_path, old_start=0, old_count=0,
        new_start=new_start, new_count=1,
        lines=[DiffLine(content=code or "    password = 'x'", change_type=ChangeType.ADDED)],
    )


def _mock_rag():
    rag = MagicMock()
    rag.search_history.return_value = []
    rag.search_codebase.return_value = []
    rag.search_security.return_value = []
    rag.add_history = MagicMock()
    return rag


# ---------------------------------------------------------------------------
# f1: Planner fail-fast
# ---------------------------------------------------------------------------

def test_planner_degradation_fails_fast():
    """chat_json returning None (degradation) must NOT fold into an empty
    plan: the graph routes straight to reporter with a failed report."""
    from src.graph import build_graph
    from src.rules import registry

    diff = (
        "diff --git a/a.py b/a.py\n@@ -0,0 +1,2 @@\n"
        "+def f():\n+    password = 'sk-1234567890'\n"
    )
    from src.parsers.diff_parser import GitDiffParser
    hunks = GitDiffParser().parse(diff)

    mock_llm = MagicMock()
    # Planner degrades (None), everything after should never run.
    mock_llm.chat_json.return_value = None
    mock_ltm = MagicMock()

    graph = build_graph(mock_llm, _mock_rag(), mock_ltm, registry, max_rounds=3)
    result = graph.invoke(
        {"repo": "t", "raw_diff": diff, "hunks": [h.model_dump() for h in hunks]},
        {"recursion_limit": 30},
    )
    report = result["report"]
    assert report.status == "failed"
    assert report.total_hunks == len(hunks)
    assert report.risks == []
    assert report.rules_executed == []
    assert "Planning failed" in report.summary
    assert result["fatal_error"]
    # No rule execution or judgment happened after the failed planning.
    assert mock_llm.chat_json.call_count == 1
    assert result["evidence_pool"] == []


def test_planner_empty_plan_is_not_failure():
    """A valid JSON response with zero assignments continues the pipeline
    normally — an empty plan is a legitimate verdict, not a crash."""
    from src.nodes.planner import PlannerNode

    llm = MagicMock()
    llm.chat_json.return_value = {"plan_by_hunk": {}}
    node = PlannerNode(llm, _mock_rag())
    result = node(AgentState(hunks=[_hunk()]))
    assert "fatal_error" not in result
    assert result["pending_tools_by_hunk"] == {}
    assert result["phase"].value == "tool_routing"


# ---------------------------------------------------------------------------
# f2: Judge degraded marking
# ---------------------------------------------------------------------------

def test_judge_full_degradation_counts_unadjudicated_and_preserves_risks():
    from src.nodes.judge import JudgeNode

    llm = MagicMock()
    llm.chat_json.return_value = None  # every batch degrades
    judge = JudgeNode(llm, _mock_rag())
    prior_risk = RiskItem(
        title="Prior", category=RiskCategory.SECURITY, severity=Severity.HIGH,
        description="d", evidence_chain=[_evidence(1)], risk_score=0.5,
    )
    state = AgentState(
        hunks=[_hunk()],
        evidence_pool=[_evidence(1), _evidence(2)],
        risks=[prior_risk],
    )
    result = judge(state)
    # Whole-call degradation: risks key omitted to preserve prior round.
    assert "risks" not in result
    assert result["judge_unadjudicated_evidence"] == 2


def test_judge_partial_degradation_isolated_per_batch():
    from src.nodes.judge import JudgeNode

    llm = MagicMock()
    # First batch degrades, second succeeds.
    responses = [None, {"risks": [], "dismissed_evidence": []}]
    llm.chat_json.side_effect = responses
    judge = JudgeNode(llm, _mock_rag())
    judge._BATCH_SIZE = 1  # one evidence per batch for the test
    state = AgentState(
        hunks=[_hunk()],
        evidence_pool=[_evidence(1), _evidence(2)],
    )
    result = judge(state)
    assert result["judge_unadjudicated_evidence"] == 1
    assert result["risks"] == []
    assert llm.chat_json.call_count == 2


# ---------------------------------------------------------------------------
# f3: Evidence hunk-key merge dedup
# ---------------------------------------------------------------------------

def test_identical_evidence_across_hunks_merges_hunk_keys():
    from src.nodes.tool_router import ToolRouterNode

    registry = MagicMock()
    # Same rule, identical enriched content for both hunks.
    ev = _evidence(1)
    registry.execute.return_value = [ev.model_copy()]
    registry.list_all.return_value = ["rule_a"]

    node = ToolRouterNode(registry, _mock_rag())
    state = AgentState(
        hunks=[_hunk(new_start=1), _hunk(new_start=10)],
        pending_tools_by_hunk={"a.py:1": ["rule_a"], "a.py:10": ["rule_a"]},
    )
    result = node(state)
    # ONE evidence in this round's return, carrying both hunk keys.
    assert len(result["evidence_pool"]) == 1
    assert set(result["evidence_pool"][0].hunk_keys) == {"a.py:1", "a.py:10"}


def test_prior_round_duplicate_evidence_merges_not_appends():
    from src.nodes.tool_router import ToolRouterNode

    registry = MagicMock()
    existing = _evidence(1)
    existing.hunk_keys = ["a.py:1"]
    registry.execute.return_value = [existing.model_copy()]
    registry.list_all.return_value = ["rule_a"]

    node = ToolRouterNode(registry, _mock_rag())
    state = AgentState(
        hunks=[_hunk(new_start=1)],
        evidence_pool=[existing],
        pending_tools_by_hunk={"a.py:1": ["rule_a"]},
    )
    result = node(state)
    # The regenerated twin merged into the prior evidence: no new copy.
    assert result["evidence_pool"] == []
    assert existing.hunk_keys == ["a.py:1"]


# ---------------------------------------------------------------------------
# f4: Judge batching
# ---------------------------------------------------------------------------

def test_judge_batches_by_file_and_merges_global_refs():
    from src.nodes.judge import JudgeNode

    llm = MagicMock()
    # 3 evidence in file a.py, 2 in file b.py, batch size 2:
    # batches -> a:[0,1], a:[2], b:[3,4] (file-grouped).
    llm.chat_json.side_effect = [
        {"risks": [{"title": "R1", "evidence_refs": [0, 1],
                    "severity": "high", "category": "security",
                    "description": "d", "risk_score": 0.9}],
         "dismissed_evidence": []},
        {"risks": [{"title": "R1", "evidence_refs": [2],
                    "severity": "high", "category": "security",
                    "description": "d", "risk_score": 0.8}],
         "dismissed_evidence": []},
        {"risks": [{"title": "R2", "evidence_refs": [3, 4],
                    "severity": "medium", "category": "security",
                    "description": "d", "risk_score": 0.5}],
         "dismissed_evidence": []},
    ]
    judge = JudgeNode(llm, _mock_rag())
    judge._BATCH_SIZE = 2
    pool = [
        _evidence(0, "a.py"), _evidence(1, "a.py"), _evidence(2, "a.py"),
        _evidence(3, "b.py"), _evidence(4, "b.py"),
    ]
    result = judge(AgentState(hunks=[_hunk("a.py"), _hunk("b.py")],
                              evidence_pool=pool))
    assert llm.chat_json.call_count == 3
    # R1 surfaced in two batches with disjoint refs -> merged, chain union.
    titles = [r.title for r in result["risks"]]
    assert titles == ["R1", "R2"]
    r1 = result["risks"][0]
    assert len(r1.evidence_chain) == 3
    assert r1.risk_score == 0.9
    assert result["judge_unadjudicated_evidence"] == 0


def test_judge_out_of_batch_ref_is_rejected():
    from src.nodes.judge import JudgeNode

    llm = MagicMock()
    # First batch only sees id {0}; its ref to 1 is a hallucinated
    # cross-batch reference and must be dropped. Second batch: clean.
    llm.chat_json.side_effect = [
        {"risks": [{"title": "R1", "evidence_refs": [0, 1],
                    "severity": "high", "category": "security",
                    "description": "d", "risk_score": 0.9}],
         "dismissed_evidence": []},
        {"risks": [], "dismissed_evidence": []},
    ]
    judge = JudgeNode(llm, _mock_rag())
    judge._BATCH_SIZE = 1
    pool = [_evidence(0, "a.py"), _evidence(1, "a.py")]
    result = judge(AgentState(hunks=[_hunk("a.py")], evidence_pool=pool))
    r1 = result["risks"][0]
    assert len(r1.evidence_chain) == 1
    assert r1.evidence_chain[0].rule_id == "SEC000"


def test_judge_grouping_never_splits_batch_across_files():
    """Batch construction groups by file first: a batch never mixes files."""
    from src.nodes.judge import JudgeNode

    llm = MagicMock()
    llm.chat_json.return_value = {"risks": [], "dismissed_evidence": []}
    judge = JudgeNode(llm, _mock_rag())
    judge._BATCH_SIZE = 50
    pool = [_evidence(i, "a.py") for i in range(3)] + [_evidence(10, "b.py")]
    judge(AgentState(hunks=[_hunk("a.py"), _hunk("b.py")], evidence_pool=pool))
    seen_files = set()
    for call in llm.chat_json.call_args_list:
        prompt = call[0][1]
        for i, ev in enumerate(pool):
            marker = f'"file_path": "{ev.file_path}"'
            if f'"id": {i}' in prompt and marker in prompt:
                seen_files.add(ev.file_path)
    # Each call's evidence belonged to exactly one file per batch: we can
    # verify indirectly via call count (one batch per file).
    assert llm.chat_json.call_count == 2


# ---------------------------------------------------------------------------
# Reporter: status derivation + hunk summaries
# ---------------------------------------------------------------------------

def test_reporter_failed_status_from_fatal_error():
    from src.nodes.reporter import ReporterNode

    state = AgentState(
        repo="r", hunks=[_hunk()], fatal_error="Planning failed: boom",
    )
    report = ReporterNode._build_report(state)
    assert report.status == "failed"
    assert report.risks == []
    assert "Planning failed: boom" in report.summary
    text = report.to_text()
    assert "Analysis aborted" in text


def test_reporter_degraded_status_from_unadjudicated():
    from src.nodes.reporter import ReporterNode

    state = AgentState(
        repo="r", hunks=[_hunk()], judge_unadjudicated_evidence=7,
    )
    report = ReporterNode._build_report(state)
    assert report.status == "degraded"
    assert report.unadjudicated_evidence == 7
    assert "7" in report.summary
    assert "never adjudicated" in report.to_text()


def test_reporter_builds_hunk_summaries():
    from src.nodes.reporter import ReporterNode
    from src.models import RuleOutcome, RuleOutcomeStatus

    ev = _evidence(1)
    ev.hunk_keys = ["a.py:1", "a.py:5"]
    risk = RiskItem(
        title="T", category=RiskCategory.SECURITY, severity=Severity.HIGH,
        description="d", evidence_chain=[ev], risk_score=0.5, file_path="a.py",
    )
    state = AgentState(
        repo="r",
        hunks=[_hunk(new_start=1), _hunk(new_start=5), _hunk(new_start=9)],
        evidence_pool=[ev],
        risks=[risk],
        rule_outcomes=[
            RuleOutcome(hunk_key="a.py:1", rule="r1",
                        status=RuleOutcomeStatus.EVIDENCE_PRODUCED),
        ],
    )
    report = ReporterNode._build_report(state)
    assert len(report.hunk_summaries) == 3
    by_key = {hs.hunk_key: hs for hs in report.hunk_summaries}
    assert by_key["a.py:1"].evidence_count == 1
    assert by_key["a.py:5"].evidence_count == 1  # same evidence, two hunks
    assert by_key["a.py:1"].risk_titles == ["T"]
    assert by_key["a.py:5"].risk_titles == ["T"]
    assert by_key["a.py:9"].rule_statuses == {}
    assert by_key["a.py:9"].evidence_count == 0
    assert "Per-hunk summary" in report.to_text()


# ---------------------------------------------------------------------------
# f5: small fixes
# ---------------------------------------------------------------------------

def test_normalize_file_pattern():
    from src.memory.long_term import normalize_file_pattern
    assert normalize_file_pattern("src/auth/*") == "src/auth/"
    assert normalize_file_pattern("src/auth/**") == "src/auth/"
    assert normalize_file_pattern("src/auth") == "src/auth"
    assert normalize_file_pattern(" src/auth/* ") == "src/auth/"
    assert normalize_file_pattern("") == ""


def test_add_feedback_normalizes_pattern(tmp_path):
    from src.memory.long_term import LongTermMemory

    ltm = LongTermMemory(str(tmp_path / "fb.db"))
    ltm.init_tables()
    ltm.add_feedback("t1", "src/auth/*", None, "false_positive", "login")
    rows = ltm.get_all_feedback()
    assert rows[0]["file_pattern"] == "src/auth/"


def test_chat_json_validator_non_valueerror_enters_retry():
    """A validator raising e.g. KeyError must trigger the repair-retry
    path instead of escaping and crashing the pipeline."""
    from src.llm.client import LLMClient

    client = LLMClient(api_key="fake", api_base="http://localhost", model="m")
    calls = {"n": 0}

    def validator(parsed):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyError("missing field")

    client.chat_with_messages = MagicMock(return_value='{"a": 1}')
    result = client.chat_json("sys", "usr", validator=validator)
    assert result == {"a": 1}
    assert calls["n"] == 2  # first rejected, second accepted
    assert client.chat_with_messages.call_count == 2


def test_llm_client_default_timeout_is_500():
    from src.llm.client import LLMClient
    client = LLMClient(api_key="fake", api_base="http://localhost", model="m")
    assert client._client.timeout == 500.0
