import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.retriever import RAGRetriever
from src.rag.indexer import SecurityKnowledgeLoader


def _get_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def test_retriever_init_creates_tables():
    db_path = _get_db_path()
    mock_embedding = MagicMock()
    retriever = RAGRetriever(db_path, mock_embedding)
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "past_risks" in tables
    assert "security_knowledge" in tables
    assert "codebase_index" in tables
    os.unlink(db_path)


def test_add_and_search_history():
    db_path = _get_db_path()
    mock_embedding = MagicMock()
    mock_embedding.embed.return_value = [0.1, 0.2, 0.3]
    retriever = RAGRetriever(db_path, mock_embedding)
    retriever.add_history(
        thread_id="t1", file_path="auth/login.py",
        diff_summary="Modified login function with SQL query",
        risk_titles=["SQL injection"], risk_categories=["security"],
        overall_score=0.8,
    )
    results = retriever.search_history(
        query="login SQL query", file_pattern="auth/*", top_k=5
    )
    assert len(results) >= 1
    assert "SQL injection" in results[0]["risk_titles"]
    os.unlink(db_path)


def test_search_security_with_rule_id():
    db_path = _get_db_path()
    mock_embedding = MagicMock()
    mock_embedding.embed.return_value = [0.1, 0.2, 0.3]
    retriever = RAGRetriever(db_path, mock_embedding)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO security_knowledge (title, category, rule_id, content, best_practice, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("SQL Injection Prevention", "security", "SEC001",
         "Never concatenate strings for SQL queries", "Use parameterized queries",
         json.dumps([0.1, 0.2, 0.3])),
    )
    conn.commit()
    conn.close()
    results = retriever.search_security(
        query="SQL injection", rule_ids=["SEC001"], top_k=5
    )
    assert len(results) >= 1
    assert "parameterized" in results[0]["best_practice"].lower()
    os.unlink(db_path)


def test_security_knowledge_loader():
    db_path = _get_db_path()
    loader = SecurityKnowledgeLoader(db_path)
    loader.init_tables()
    knowledge = [
        {
            "title": "SQL Injection",
            "category": "security",
            "rule_id": "SEC001",
            "content": "String concatenation in SQL is dangerous",
            "best_practice": "Use parameterized queries",
            "references": ["https://owasp.org/sql-injection"],
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(knowledge, f)
        json_path = f.name
    loader.load_from_json(json_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM security_knowledge")
    count = cursor.fetchone()[0]
    conn.close()
    os.unlink(json_path)
    os.unlink(db_path)
    assert count == 1
