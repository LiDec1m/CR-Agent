import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.long_term import LongTermMemory


def _get_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def test_long_term_init_tables():
    db_path = _get_db_path()
    mem = LongTermMemory(db_path)
    mem.init_tables()
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "feedback" in tables
    os.unlink(db_path)


def test_add_and_get_feedback():
    db_path = _get_db_path()
    mem = LongTermMemory(db_path)
    mem.init_tables()
    mem.add_feedback(
        thread_id="t1", file_pattern="auth/*",
        rule_id="SEC001", feedback_type="false_positive",
        content="This is parameterized, not injection",
    )
    results = mem.get_feedback("auth/*")
    assert len(results) == 1
    assert results[0]["rule_id"] == "SEC001"
    assert results[0]["feedback_type"] == "false_positive"
    os.unlink(db_path)


def test_get_all_feedback():
    db_path = _get_db_path()
    mem = LongTermMemory(db_path)
    mem.init_tables()
    mem.add_feedback("t1", "a/*", "SEC001", "confirmed", "yes")
    mem.add_feedback("t2", "b/*", "SEC002", "false_positive", "no")
    results = mem.get_all_feedback()
    assert len(results) == 2
    os.unlink(db_path)
