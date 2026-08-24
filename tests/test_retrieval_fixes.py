"""Tests for retrieval-layer fixes: OR-joined FTS, exact-path codebase
lookup, planner history dedup/slimming, and embedding-cost removal."""

import json
import os
import sqlite3
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.retriever import RAGRetriever


class _FakeEmbedding:
    """Deterministic embedding client for tests (no API calls)."""

    def __init__(self):
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Simple bag-of-words hashing so similarity is controllable.
        vec = [0.0] * 16
        for token in text.lower().split():
            vec[hash(token) % 16] += 1.0
        return vec


def _make_retriever(tmp_path):
    client = _FakeEmbedding()
    rag = RAGRetriever(str(tmp_path / "rag.db"), client)
    return rag, client


# ---------------------------------------------------------------------------
# A: FTS query is OR-joined (bm25 ranks by overlap count, not all-hits)
# ---------------------------------------------------------------------------

def test_fts_query_or_joined():
    q = RAGRetriever._build_fts_query("def parse_sql query")
    assert q == '"def" OR "parse_sql" OR "query"'


def test_fts_or_semantics_finds_partial_overlap(tmp_path):
    """AND semantics: a row containing 2 of 3 query tokens would NOT
    match. OR semantics: it matches and ranks below a 3/3 row."""
    rag, _ = _make_retriever(tmp_path)
    rag.add_history("t1", "a.py", "added parse_sql helper", ["SQL helper"], ["security"], 0.5)
    rag.add_history("t2", "a.py", "parse_sql query timeout", ["SQL timeout"], ["security"], 0.5)

    results = rag.search_history("parse_sql query", file_pattern=None)
    # Both rows are candidates under OR; full-overlap row ranks first.
    assert len(results) >= 2
    assert results[0]["diff_summary"] == "parse_sql query timeout"


def test_fts_query_long_input_no_truncation():
    """No token truncation: every token participates in the OR chain."""
    text = " ".join(f"tok{i}" for i in range(200))
    q = RAGRetriever._build_fts_query(text)
    assert q.count(" OR ") == 199


# ---------------------------------------------------------------------------
# C: search_codebase is an exact-path lookup
# ---------------------------------------------------------------------------

def _index_one_file(rag, file_path, symbols):
    conn = sqlite3.connect(rag._db_path)
    for sym in symbols:
        cur = conn.execute(
            "INSERT INTO codebase_index (file_path, symbol_name, symbol_type,"
            " line_range, content, imports, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_path, sym["name"], sym["type"], sym["lines"],
             sym["content"], json.dumps([]), "2026-01-01"),
        )
        conn.execute(
            "INSERT INTO codebase_index_fts (rowid, file_path, symbol_name,"
            " content) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, file_path, sym["name"], sym["content"]),
        )
    conn.commit()
    conn.close()


def test_search_codebase_returns_all_file_symbols(tmp_path):
    """A file with 8 symbols returns all 8: the old FTS+rank LIMIT 5
    dropped 3 of them arbitrarily, silently breaking _enrich_hunk."""
    rag, _ = _make_retriever(tmp_path)
    syms = [
        {"name": f"fn_{i}", "type": "function", "lines": f"{i * 10}-{i * 10 + 5}",
         "content": f"def fn_{i}(): pass"}
        for i in range(8)
    ]
    _index_one_file(rag, "src/big.py", syms)
    results = rag.search_codebase("src/big.py")
    diff_file = [r for r in results if r["source"] == "diff_file"]
    assert len(diff_file) == 8
    assert {r["symbol_name"] for r in diff_file} == {f"fn_{i}" for i in range(8)}


def test_search_codebase_exact_path_no_cross_file_leak(tmp_path):
    """A different file whose CONTENT happens to contain the queried
    path tokens must not leak into the results (old FTS MATCH could)."""
    rag, _ = _make_retriever(tmp_path)
    _index_one_file(rag, "src/llm/client.py", [
        {"name": "chat_json", "type": "function", "lines": "1-10",
         "content": "def chat_json(): pass"},
    ])
    # This file's content mentions "client.py" verbatim.
    _index_one_file(rag, "src/notes.py", [
        {"name": "notes", "type": "module", "lines": "1-5",
         "content": "# see also src/llm/client.py chat_json"},
    ])
    results = rag.search_codebase("src/llm/client.py")
    diff_file = {r["file_path"] for r in results if r["source"] == "diff_file"}
    assert diff_file == {"src/llm/client.py"}


def test_search_codebase_unknown_path_returns_empty(tmp_path):
    rag, _ = _make_retriever(tmp_path)
    assert rag.search_codebase("no/such/file.py") == []


# ---------------------------------------------------------------------------
# B + D: planner history recall — path-aligned query, shared embeds, dedup
# ---------------------------------------------------------------------------

def _make_hunk(file_path="a.py", new_start=1, code="x = 1"):
    from src.models import ChangeType, DiffLine, HunkInfo
    return HunkInfo(
        file_path=file_path, old_start=0, old_count=0,
        new_start=new_start, new_count=1,
        lines=[DiffLine(content=code, change_type=ChangeType.ADDED)],
    )


def test_planner_history_query_includes_file_path():
    """The embed query is `file_path added_code`, aligning with the
    write-side embed text which starts with the file path."""
    from src.nodes.planner import PlannerNode

    llm = MagicMock()
    llm.chat_json.return_value = {"plan_by_hunk": {}}
    rag = MagicMock()
    rag.search_history.return_value = []
    planner = PlannerNode(llm, rag)
    planner(AgentStateHunks([_make_hunk("src/a.py", code="def f(): pass")]))
    query = rag.search_history.call_args[0][0]
    assert query.startswith("src/a.py ")
    assert "def f(): pass" in query


def AgentStateHunks(hunks):
    from src.models import AgentState
    return AgentState(hunks=hunks)


def test_planner_identical_hunks_share_one_embed_call():
    from src.nodes.planner import PlannerNode

    llm = MagicMock()
    llm.chat_json.return_value = {"plan_by_hunk": {}}
    rag = MagicMock()
    rag.search_history.return_value = []
    planner = PlannerNode(llm, rag)
    planner(AgentStateHunks([
        _make_hunk("a.py", new_start=1, code="same code"),
        _make_hunk("a.py", new_start=10, code="same code"),
    ]))
    assert rag.search_history.call_count == 1


def test_planner_same_code_different_files_no_cache_share():
    """The embed-share key includes file_path: identical code in two
    files must recall each file's history separately."""
    from src.nodes.planner import PlannerNode

    llm = MagicMock()
    llm.chat_json.return_value = {"plan_by_hunk": {}}
    rag = MagicMock()
    rag.search_history.return_value = []
    planner = PlannerNode(llm, rag)
    planner(AgentStateHunks([
        _make_hunk("a.py", new_start=1, code="same code"),
        _make_hunk("b.py", new_start=1, code="same code"),
    ]))
    assert rag.search_history.call_count == 2


def test_planner_history_dedup_and_slim_prompt():
    """Same history row recalled for two hunks appears once in the
    prompt, carrying only the five LLM-relevant fields."""
    from src.nodes.planner import PlannerNode

    llm = MagicMock()
    llm.chat_json.return_value = {"plan_by_hunk": {}}
    row = {
        "id": 7, "thread_id": "th-1", "file_path": "a.py",
        "diff_summary": "added login retry", "risk_titles": ["Hardcoded secret"],
        "risk_categories": ["security"], "overall_score": 0.6,
        "created_at": "2026-08-01T00:00:00",
    }
    rag = MagicMock()
    rag.search_history.return_value = [row]
    planner = PlannerNode(llm, rag)
    # Two different hunks (same file): search_history is called twice
    # (different added_code), each returning the same row.
    planner(AgentStateHunks([
        _make_hunk("a.py", new_start=1, code="code one"),
        _make_hunk("a.py", new_start=10, code="code two"),
    ]))
    prompt = llm.chat_json.call_args[0][1]
    history_block = json.loads(
        prompt.split("Historical risks:\n")[1].split("\n\n")[0]
    )
    assert len(history_block) == 1  # deduped
    assert set(history_block[0].keys()) == {
        "file_path", "diff_summary", "risk_titles",
        "risk_categories", "overall_score",
    }
    assert "thread_id" not in prompt
    assert "created_at" not in prompt


# ---------------------------------------------------------------------------
# F: indexer writes no symbol embeddings
# ---------------------------------------------------------------------------

def test_index_file_full_writes_null_embedding(tmp_path):
    from src.rag.indexer import CodebaseIndexer

    client = MagicMock()  # embed_batch must never be called
    indexer = CodebaseIndexer(str(tmp_path / "cb.db"), client)
    indexer.init_tables()
    source = (
        "import os\n"
        "def alpha():\n"
        "    return 1\n"
    )
    count = indexer.index_file_full("m.py", source)
    assert count >= 1
    client.embed_batch.assert_not_called()
    conn = sqlite3.connect(str(tmp_path / "cb.db"))
    embeddings = [
        r[0] for r in conn.execute("SELECT embedding FROM codebase_index")
    ]
    conn.close()
    assert all(e is None for e in embeddings)
