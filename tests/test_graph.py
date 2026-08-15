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
