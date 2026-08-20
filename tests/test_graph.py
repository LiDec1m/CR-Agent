import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _hunk_key(diff):
    """Parse the first hunk's file_path:new_start from a diff string."""
    from src.parsers.diff_parser import GitDiffParser
    hunks = GitDiffParser().parse(diff)
    return f"{hunks[0].file_path}:{hunks[0].new_start}"


def test_graph_runs_end_to_end():
    from src.graph import build_graph
    from src.rules import registry

    key = _hunk_key(SAMPLE_DIFF)
    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "test",
         "plan_by_hunk": {key: ["hardcoded_secret", "unsafe_deserialize", "command_injection"]},
         "risk_areas": []},
        {"risks": [{"title": "Security Issues", "category": "security",
                    "severity": "critical", "description": "Multiple",
                    "evidence_refs": [0, 1, 2], "suggestion": "Fix",
                    "file_path": "test.py", "line_range": [2, 4], "risk_score": 0.95}],
         "overall_risk_score": 0.95},
        {"needs_more_analysis": False, "additional_tools_by_hunk": {},
         "reason": "Sufficient", "coverage_assessment": "100%"},
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
        "repo": "test", "raw_diff": SAMPLE_DIFF,
        "hunks": [h.model_dump() for h in hunks],
    }
    result = graph.invoke(initial_state, {"configurable": {"thread_id": "test-1"}})
    assert result is not None
    assert len(result.get("evidence_pool", [])) >= 1


def test_graph_checkpointer_persists_state():
    import tempfile
    from langgraph.checkpoint.sqlite import SqliteSaver
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    diff = (
        "diff --git a/x.py b/x.py\n@@ -0,0 +1,2 @@\n"
        "+def f():\n+    password = 'sk-12345'\n"
    )
    hunks = GitDiffParser().parse(diff)
    key = f"{hunks[0].file_path}:{hunks[0].new_start}"
    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan_by_hunk": {key: ["hardcoded_secret"]}, "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": False, "additional_tools_by_hunk": {},
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
            graph = build_graph(mock_llm, mock_rag, mock_ltm, registry,
                                max_rounds=3, checkpointer=cp)
            cfg = {"configurable": {"thread_id": "test-cp"}}
            result = graph.invoke(
                {"repo": "t", "raw_diff": diff,
                 "hunks": [h.model_dump() for h in hunks]}, cfg)
            assert result["report"] is not None
            snap = graph.get_state(cfg)
            assert snap.values.get("report") is not None


def test_graph_round_cap_still_produces_report_and_is_observable():
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    diff = (
        "diff --git a/y.py b/y.py\n@@ -0,0 +1,2 @@\n"
        "+def f():\n+    password = 'sk-999'\n"
    )
    hunks = GitDiffParser().parse(diff)
    key = f"{hunks[0].file_path}:{hunks[0].new_start}"
    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan_by_hunk": {key: ["hardcoded_secret"]}, "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True,
         "additional_tools_by_hunk": {key: ["magic_number"]},
         "reason": "more", "coverage_assessment": "40%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True,
         "additional_tools_by_hunk": {key: ["long_line"]},
         "reason": "more", "coverage_assessment": "60%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True,
         "additional_tools_by_hunk": {key: ["naming_violation"]},
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
    result = graph.invoke(
        {"repo": "t", "raw_diff": diff, "hunks": [h.model_dump() for h in hunks]},
        {"recursion_limit": 30},
    )
    assert result["report"] is not None
    assert result["needs_more_analysis"] is True
    assert result["reflection_round"] == 3


def test_graph_reflection_notes_not_duplicated():
    from src.graph import build_graph
    from src.rules import registry
    from src.parsers.diff_parser import GitDiffParser

    diff = (
        "diff --git a/y.py b/y.py\n@@ -0,0 +1,2 @@\n"
        "+def f():\n+    password = 'sk-999'\n"
    )
    hunks = GitDiffParser().parse(diff)
    key = f"{hunks[0].file_path}:{hunks[0].new_start}"
    mock_llm = MagicMock()
    mock_llm.chat_json.side_effect = [
        {"summary": "t", "plan_by_hunk": {key: ["hardcoded_secret"]}, "risk_areas": []},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": True,
         "additional_tools_by_hunk": {key: ["magic_number"]},
         "reason": "more", "coverage_assessment": "40%"},
        {"risks": [], "overall_risk_score": 0.0},
        {"needs_more_analysis": False, "additional_tools_by_hunk": {},
         "reason": "done", "coverage_assessment": "95%"},
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
    result = graph.invoke(
        {"repo": "t", "raw_diff": diff, "hunks": [h.model_dump() for h in hunks]},
        {"recursion_limit": 30},
    )
    notes = result["reflection_notes"]
    assert len(notes) == 2
    assert len(set(notes)) == 2
    assert notes[0].startswith("Round 1:")
    assert notes[1].startswith("Round 2:")
