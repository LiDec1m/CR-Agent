"""Long-term memory: feedback table for human-in-the-loop corrections."""

from __future__ import annotations

import sqlite3


class LongTermMemory:
    """CRUD operations for the feedback table."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def init_tables(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                file_pattern TEXT NOT NULL,
                rule_id TEXT,
                feedback_type TEXT NOT NULL,
                feedback_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()
        conn.close()

    def add_feedback(self, thread_id: str, file_pattern: str, rule_id: str,
                     feedback_type: str, content: str) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO feedback (thread_id, file_pattern, rule_id, feedback_type, feedback_content) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, file_pattern, rule_id, feedback_type, content),
        )
        conn.commit()
        conn.close()

    def get_feedback(self, file_pattern: str) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        glob_pattern = file_pattern.replace("*", "%")
        rows = conn.execute(
            "SELECT * FROM feedback WHERE file_pattern LIKE ?",
            (glob_pattern,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_feedback_by_rule_across_files(
        self,
        rule_ids: list[str],
        exclude_file_pattern: str | None = None,
    ) -> list[dict]:
        """Retrieve feedback by rule_id across ALL files.

        This solves the cross-file feedback problem: if SEC001 was a
        false_positive for judge.py, the same feedback should also
        inform analysis of tool_router.py when SEC001 is triggered.

        Args:
            rule_ids: Rule IDs to match.
            exclude_file_pattern: Optionally exclude feedback from this
                file pattern to avoid duplication (since file-specific
                feedback is already fetched via get_feedback).
        """
        if not rule_ids:
            return []
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in rule_ids)
        sql = (
            f"SELECT * FROM feedback "
            f"WHERE rule_id IN ({placeholders})"
        )
        params: list = list(rule_ids)
        if exclude_file_pattern:
            glob_pattern = exclude_file_pattern.replace("*", "%")
            sql += " AND file_pattern NOT LIKE ?"
            params.append(glob_pattern)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_feedback(self) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM feedback").fetchall()
        conn.close()
        return [dict(r) for r in rows]
