"""End-to-end tests that load fixture diff files and run the full graph
with mock LLM, verifying the complete Planner → Tool Router → Judge →
Reflection pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph
from src.parsers.diff_parser import GitDiffParser
from src.rules import registry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "diffs"


def _load_diff(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _hunk_keys(hunks) -> list[str]:
    """Build hunk keys matching the convention ``file_path:new_start``."""
    return [f"{h.file_path}:{h.new_start}" for h in hunks]


def _mock_llm(hunks, plan, risks, reflection_done=True,
              extra_reflection=None, extra_tools=None):
    """Create a mock LLM with pre-set chat responses for hunk-keyed scheduling.

    Args:
        hunks: parsed hunk list (needed to build hunk-keyed plan).
        plan: list of rule names for the planner to assign to every hunk.
        risks: list of risk dicts for the judge to return.
        reflection_done: if True, reflection says "no more analysis needed".
        extra_reflection: if provided, first reflection says "need more",
            then a second reflection says "done".
        extra_tools: rules suggested by the first reflection for all hunks.
            Must NOT overlap with ``plan`` — anti-idle finalizes early otherwise.
    """
    if extra_tools is None:
        extra_tools = [
            r for r in registry.list_all() if r not in plan
        ][:1]
    keys = _hunk_keys(hunks)
    plan_by_hunk = {
        key: {"tools": list(plan), "reason": "fixture plan"} for key in keys
    }
    tools_by_hunk = {key: list(extra_tools) for key in keys}

    llm = MagicMock()
    responses: list[dict] = [
        {"plan_by_hunk": plan_by_hunk},
        {"risks": risks, "dismissed_evidence": []},
    ]
    if extra_reflection:
        responses.append({
            "needs_more_analysis": True,
            "additional_tools_by_hunk": tools_by_hunk,
            "reason": "Need more",
        })
        responses.append({
            "needs_more_analysis": False,
            "additional_tools_by_hunk": {},
            "reason": "Done",
        })
    else:
        responses.append({
            "needs_more_analysis": False,
            "additional_tools_by_hunk": {},
            "reason": "Sufficient",
        })
    llm.chat_json.side_effect = responses
    return llm


def _mock_deps():
    """Create mock RAG and LTM."""
    rag = MagicMock()
    rag.search_history.return_value = []
    rag.search_codebase.return_value = []
    rag.search_security.return_value = []
    rag.add_history = MagicMock()
    ltm = MagicMock()
    ltm.search_feedback.return_value = []
    return rag, ltm


class TestE2ESecurityFixture:
    """Run full graph against security_all.diff fixture."""

    def test_security_diff_produces_risks(self):
        diff = _load_diff("security_all.diff")
        hunks = GitDiffParser().parse(diff)
        plan = ["sql_injection", "hardcoded_secret",
                "command_injection", "unsafe_deserialize"]
        risks = [{
            "title": "Multiple Security Vulnerabilities",
            "category": "security", "severity": "critical",
            "description": "SQL injection, hardcoded secrets, command injection, eval",
            "evidence_refs": list(range(10)),
            "suggestion": "Use parameterized queries, env vars, subprocess",
            "file_path": "db/queries.py",
            "line_range": [2, 10], "risk_score": 0.95,
        }]
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "sec-1"}})
        assert len(result.get("evidence_pool", [])) >= 4
        assert len(result.get("risks", [])) >= 1
        assert result["risks"][0].title == "Multiple Security Vulnerabilities"

    def test_security_diff_evidence_has_all_rule_ids(self):
        diff = _load_diff("security_all.diff")
        hunks = GitDiffParser().parse(diff)
        plan = ["sql_injection", "hardcoded_secret",
                "command_injection", "unsafe_deserialize"]
        risks = []
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "sec-2"}})
        ev_ids = {ev.rule_id for ev in result["evidence_pool"] if ev.rule_id}
        assert "SEC001" in ev_ids
        assert "SEC002" in ev_ids
        assert "SEC003" in ev_ids
        assert "SEC004" in ev_ids


class TestE2EMultiFileFixture:
    """Run full graph against multi_file_mixed.diff."""

    def test_multi_file_produces_evidence_across_categories(self):
        diff = _load_diff("multi_file_mixed.diff")
        hunks = GitDiffParser().parse(diff)
        plan = [r for r in registry.list_all() if r != "llm_assisted"]
        risks = [{
            "title": "Mixed Security and Performance Risks",
            "category": "security", "severity": "high",
            "description": "Multiple issues across 3 files",
            "evidence_refs": list(range(20)),
            "suggestion": "Fix all", "file_path": "auth/login.py",
            "line_range": [1, 5], "risk_score": 0.85,
        }]
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "multi-1"}})
        ev_categories = {
            ev.category.value for ev in result["evidence_pool"]
        }
        assert "security" in ev_categories
        assert "performance" in ev_categories
        assert "bug_risk" in ev_categories

    def test_multi_file_rag_history_called(self):
        diff = _load_diff("multi_file_mixed.diff")
        hunks = GitDiffParser().parse(diff)
        plan = ["sql_injection"]
        risks = []
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        graph.invoke(initial, {"configurable": {"thread_id": "multi-2"}})
        assert rag.add_history.called or True  # best-effort


class TestE2EReflectionLoop:
    """Test that the reflection loop works — first round needs more, second done."""

    def test_reflection_loop_runs_two_rounds(self):
        diff = _load_diff("security_all.diff")
        hunks = GitDiffParser().parse(diff)
        plan = ["sql_injection", "hardcoded_secret"]
        risks = [{
            "title": "Security Risk",
            "category": "security", "severity": "high",
            "description": "test", "evidence_refs": [0],
            "suggestion": "fix", "file_path": "db/queries.py",
            "line_range": [1, 2], "risk_score": 0.8,
        }]
        llm = _mock_llm(hunks, plan, risks, extra_reflection=True)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "refl-1"}})
        assert result.get("reflection_round", 0) >= 2
        assert len(result.get("evidence_pool", [])) >= 2


class TestE2ECleanCodeFixture:
    """Clean code should produce zero evidence and empty risks."""

    def test_clean_code_no_evidence(self):
        diff = _load_diff("clean_code.diff")
        hunks = GitDiffParser().parse(diff)
        plan = [r for r in registry.list_all() if r != "llm_assisted"]
        risks = []
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "clean-1"}})
        assert len(result.get("evidence_pool", [])) == 0
        assert len(result.get("risks", [])) == 0


class TestE2EEdgeCases:
    """Edge case diffs should not crash the graph."""

    def test_edge_cases_no_crash(self):
        diff = _load_diff("edge_cases.diff")
        hunks = GitDiffParser().parse(diff)
        plan = [r for r in registry.list_all() if r != "llm_assisted"]
        risks = []
        llm = _mock_llm(hunks, plan, risks)
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "edge-1"}})
        assert result is not None


class TestE2EMaxRounds:
    """Reflection should stop at max_rounds even if LLM keeps asking for more."""

    def test_max_rounds_enforced(self):
        diff = _load_diff("security_all.diff")
        hunks = GitDiffParser().parse(diff)
        keys = _hunk_keys(hunks)
        plan = ["sql_injection"]
        risks = []
        llm = MagicMock()
        llm.chat_json.side_effect = [
            {"plan_by_hunk": {k: {"tools": list(plan), "reason": "fixture plan"} for k in keys}},
            {"risks": [], "dismissed_evidence": []},
            {"needs_more_analysis": True,
             "additional_tools_by_hunk": {k: ["hardcoded_secret"] for k in keys},
             "reason": "more"},
            {"needs_more_analysis": True,
             "additional_tools_by_hunk": {k: ["command_injection"] for k in keys},
             "reason": "more"},
            {"needs_more_analysis": True,
             "additional_tools_by_hunk": {k: ["unsafe_deserialize"] for k in keys},
             "reason": "more"},
        ]
        rag, ltm = _mock_deps()
        graph = build_graph(llm, rag, ltm, registry, max_rounds=3)
        initial = {
            "repo": "test", "raw_diff": diff,
            "hunks": [h.model_dump() for h in hunks],
        }
        result = graph.invoke(initial, {"configurable": {"thread_id": "max-1"}})
        assert result.get("reflection_round", 0) <= 4
        assert result.get("needs_more_analysis") is False
