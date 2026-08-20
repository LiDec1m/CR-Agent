"""Tests for hunk-level rule outcomes and targeted scheduling."""

from unittest.mock import MagicMock

from src.models import (
    AgentState, ChangeType, DiffLine, Evidence, HunkInfo, RiskCategory,
    RuleOutcome, RuleOutcomeStatus, Severity,
)
from src.nodes.reflection import ReflectionNode
from src.nodes.tool_router import ToolRouterNode
from src.rules.registry import ToolRegistry


def _hunk(path: str = "a.py", start: int = 10) -> HunkInfo:
    return HunkInfo(
        file_path=path, old_start=start, old_count=1, new_start=start,
        new_count=1, lines=[DiffLine(content="x = 1", change_type=ChangeType.ADDED)],
    )


def _evidence() -> Evidence:
    return Evidence(
        source="test_rule", rule_id="T001", category=RiskCategory.BUG_RISK,
        severity=Severity.LOW, message="found", line_range=(10, 10),
        snippet="x = 1", confidence=1.0,
    )


def _router(registry: ToolRegistry) -> ToolRouterNode:
    rag = MagicMock()
    rag.search_codebase.return_value = []
    return ToolRouterNode(registry, rag)


def test_router_records_clean_and_evidence_outcomes_per_hunk():
    registry = ToolRegistry()
    registry.register("clean_rule", lambda hunk: [])
    registry.register("finding_rule", lambda hunk: [_evidence()])
    hunk = _hunk()
    state = AgentState(
        hunks=[hunk],
        pending_tools_by_hunk={"a.py:10": ["clean_rule", "finding_rule"]},
    )

    result = _router(registry)(state)

    assert result["executed_tools_by_hunk"] == {"a.py:10": ["clean_rule", "finding_rule"]}
    assert [(o.rule, o.status) for o in result["rule_outcomes"]] == [
        ("clean_rule", RuleOutcomeStatus.CLEAN),
        ("finding_rule", RuleOutcomeStatus.EVIDENCE_PRODUCED),
    ]
    assert len(result["evidence_pool"]) == 1
    assert result["pending_tools_by_hunk"] == {}


def test_router_records_degraded_for_llm_assisted_failure():
    registry = ToolRegistry()

    def unavailable(hunk: HunkInfo) -> list[Evidence]:
        from src.rules.llm_assisted import LLMAnalysisDegraded
        raise LLMAnalysisDegraded("response JSON was truncated")

    registry.register("llm_assisted", unavailable)
    result = _router(registry)(AgentState(
        hunks=[_hunk()], pending_tools_by_hunk={"a.py:10": ["llm_assisted"]},
    ))

    outcome = result["rule_outcomes"][0]
    assert outcome.status is RuleOutcomeStatus.DEGRADED
    assert outcome.detail == "response JSON was truncated"
    assert result["evidence_pool"] == []
    # Degraded rules must NOT appear in executed_tools_by_hunk so that
    # the anti-idle check allows a retry.
    assert result["executed_tools_by_hunk"] == {}


def test_router_records_failed_for_non_llm_rule_error():
    registry = ToolRegistry()
    registry.register("broken_rule", lambda hunk: (_ for _ in ()).throw(ValueError("bad rule")))

    result = _router(registry)(AgentState(
        hunks=[_hunk()], pending_tools_by_hunk={"a.py:10": ["broken_rule"]},
    ))

    outcome = result["rule_outcomes"][0]
    assert outcome.status is RuleOutcomeStatus.FAILED
    assert "ValueError: bad rule" in (outcome.detail or "")
    assert result["evidence_pool"] == []


def test_reflection_keeps_targeted_degraded_retry_but_skips_clean_hunk():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": True,
        "additional_tools_by_hunk": {
            "a.py:10": ["unused_import"],
            "b.py:20": ["unused_import"],
        },
        "reason": "retry semantic checks",
        "coverage_assessment": "partial",
    }
    state = AgentState(
        hunks=[_hunk("a.py", 10), _hunk("b.py", 20)],
        executed_tools_by_hunk={
            "a.py:10": ["unused_import"],
            "b.py:20": ["unused_import"],
        },
        rule_outcomes=[
            RuleOutcome(hunk_key="a.py:10", rule="unused_import",
                        status=RuleOutcomeStatus.DEGRADED, detail="timeout"),
            RuleOutcome(hunk_key="b.py:20", rule="unused_import",
                        status=RuleOutcomeStatus.CLEAN),
        ],
    )

    result = ReflectionNode(llm=llm)(state)

    assert result["needs_more_analysis"] is True
    assert result["pending_tools_by_hunk"] == {"a.py:10": ["unused_import"]}


def test_reflection_digest_distinguishes_clean_degraded_and_unexamined():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "needs_more_analysis": False, "additional_tools_by_hunk": {},
        "reason": "done", "coverage_assessment": "done",
    }
    state = AgentState(
        hunks=[_hunk("clean.py", 1), _hunk("degraded.py", 2), _hunk("new.py", 3)],
        rule_outcomes=[
            RuleOutcome(hunk_key="clean.py:1", rule="llm_assisted",
                        status=RuleOutcomeStatus.CLEAN, detail="No semantic risks found."),
            RuleOutcome(hunk_key="degraded.py:2", rule="llm_assisted",
                        status=RuleOutcomeStatus.DEGRADED, detail="timeout"),
        ],
    )

    ReflectionNode(llm=llm)(state)
    prompt = llm.chat_json.call_args[0][1]
    assert "clean.py:1" in prompt and "clean" in prompt
    assert "degraded.py:2" in prompt and "degraded" in prompt
    assert "new.py:3" in prompt and "unexamined" in prompt
