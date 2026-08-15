import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AgentState


SAMPLE_DIFF = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,6 @@
 def foo():
-    pass
+    password = "sk-1234567890abcdef"
+    eval(user_input)
+    os.system("rm -rf /")
+    return password
"""


def test_graph_runs_end_to_end():
    from src.graph import build_graph
    from src.rules import registry

    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {
            "summary": "test", "plan": ["hardcoded_secret", "unsafe_deserialize", "command_injection"],
            "risk_areas": [],
        },
        {
            "risks": [{
                "title": "Security Issues",
                "category": "security", "severity": "critical",
                "description": "Multiple security issues",
                "evidence_refs": [0, 1, 2],
                "suggestion": "Fix them", "file_path": "test.py",
                "line_range": [2, 4], "risk_score": 0.95,
            }],
            "overall_risk_score": 0.95,
        },
        {
            "needs_more_analysis": False,
            "additional_tools_needed": [],
            "reason": "Sufficient", "coverage_assessment": "100%",
        },
    ]
    mock_rag = MagicMock()
    mock_rag.search_history.return_value = []
    mock_rag.search_codebase.return_value = []
    mock_rag.search_security.return_value = []
    mock_rag.add_history = MagicMock()
    mock_ltm = MagicMock()
    mock_ltm.get_feedback.return_value = []

    graph = build_graph(mock_llm, mock_rag, mock_ltm, registry, max_rounds=3)
    
    from src.parsers.diff_parser import GitDiffParser
    hunks = GitDiffParser().parse(SAMPLE_DIFF)
    initial_state = {
        "repo": "test",
        "raw_diff": SAMPLE_DIFF,
        "hunks": [h.model_dump() for h in hunks],
    }

    result = graph.invoke(initial_state, {"configurable": {"thread_id": "test-1"}})
    assert result is not None
    assert len(result.get("evidence_pool", [])) >= 1


def test_graph_checkpointer_persists_state():
    """Short-term memory: graph runs with SqliteSaver produce checkpoints
    retrievable via get_state, and a report is always produced through
    the reporter node."""
    import tempfile, os
    from langgraph.checkpoint.sqlite import SqliteSaver
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan": ["hardcoded_secret"], "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": False, "additional_tools_needed": [],
         "reason": "ok", "coverage_assessment": "100%"},
    ]
    mock_rag = MagicMock()
    mock_rag.search_history.return_value = []
    mock_rag.search_codebase.return_value = []
    mock_rag.search_security.return_value = []
    mock_rag.add_history = MagicMock()
    mock_ltm = MagicMock()
    mock_ltm.get_feedback.return_value = []

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "cp.db")
        with SqliteSaver.from_conn_string(db) as cp:
            graph = build_graph(
                mock_llm, mock_rag, mock_ltm, registry,
                max_rounds=3, checkpointer=cp,
            )
            cfg = {"configurable": {"thread_id": "test-cp"}}
            diff = (
                "diff --git a/x.py b/x.py\n"
                "@@ -0,0 +1,2 @@\n"
                "+def f():\n"
                "+    password = 'sk-12345'\n"
            )
            hunks = GitDiffParser().parse(diff)
            result = graph.invoke(
                {"repo": "t", "raw_diff": diff,
                 "hunks": [h.model_dump() for h in hunks]},
                cfg,
            )
            # report always present (reporter node guarantee)
            assert result["report"] is not None
            # state retrievable after run == checkpoint persisted
            snap = graph.get_state(cfg)
            assert snap.values.get("report") is not None


def test_graph_round_cap_still_produces_report_and_is_observable():
    """needs_more=True at the round cap must still yield a report (via
    reporter), and the final state must preserve the observation signal:
    needs_more_analysis=True (terminal True means the diff was
    under-analysed at the round cap)."""
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan": ["hardcoded_secret"], "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        # reflections keep asking for more with NEW rules until cap
        {"needs_more_analysis": True, "additional_tools_needed": ["magic_number"],
         "reason": "more", "coverage_assessment": "40%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True, "additional_tools_needed": ["long_line"],
         "reason": "more", "coverage_assessment": "60%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True, "additional_tools_needed": ["naming_violation"],
         "reason": "still not enough", "coverage_assessment": "70%"},
    ]
    mock_rag = MagicMock()
    mock_rag.search_history.return_value = []
    mock_rag.search_codebase.return_value = []
    mock_rag.search_security.return_value = []
    mock_rag.add_history = MagicMock()
    mock_ltm = MagicMock()
    mock_ltm.get_feedback.return_value = []

    graph = build_graph(mock_llm, mock_rag, mock_ltm, registry, max_rounds=3)
    diff = (
        "diff --git a/y.py b/y.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def f():\n"
        "+    password = 'sk-999'\n"
    )
    hunks = GitDiffParser().parse(diff)
    result = graph.invoke(
        {"repo": "t", "raw_diff": diff, "hunks": [h.model_dump() for h in hunks]},
        {"recursion_limit": 30},
    )
    # report still produced despite round-capped needs_more=True
    assert result["report"] is not None
    # observability preserved in final state
    assert result["needs_more_analysis"] is True
    assert result["reflection_round"] == 3


def test_graph_reflection_notes_not_duplicated():
    """Regression: GraphState.reflection_notes uses operator.add (delta
    accumulation), so ReflectionNode must return ONLY the new note. It
    used to return the full list, duplicating every earlier note on each
    round (e.g. two rounds yielded [n1, n1, n2])."""
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan": ["hardcoded_secret"], "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True, "additional_tools_needed": ["magic_number"],
         "reason": "more", "coverage_assessment": "40%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": False, "reason": "done", "coverage_assessment": "95%"},
    ]
    mock_rag = MagicMock()
    mock_rag.search_history.return_value = []
    mock_rag.search_codebase.return_value = []
    mock_rag.search_security.return_value = []
    mock_rag.add_history = MagicMock()
    mock_ltm = MagicMock()
    mock_ltm.get_feedback.return_value = []
    mock_ltm.get_feedback_by_rule_across_files.return_value = []

    graph = build_graph(mock_llm, mock_rag, mock_ltm, registry, max_rounds=3)
    diff = (
        "diff --git a/y.py b/y.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def f():\n"
        "+    password = 'sk-999'\n"
    )
    hunks = GitDiffParser().parse(diff)
    result = graph.invoke(
        {"repo": "t", "raw_diff": diff, "hunks": [h.model_dump() for h in hunks]},
        {"recursion_limit": 30},
    )
    notes = result["reflection_notes"]
    assert len(notes) == 2
    assert len(set(notes)) == 2  # no duplicates
    assert notes[0].startswith("Round 1:")
    assert notes[1].startswith("Round 2:")
