"""Long-term memory: feedback table for human-in-the-loop corrections."""

from __future__ import annotations

import sqlite3


def normalize_file_pattern(pattern: str) -> str:
    """Normalize a user-supplied file pattern to equality/prefix semantics.

    search_feedback matches with equality or prefix (no LIKE wildcards:
    ``_`` in filenames would need escaping and silently mismatches). To
    keep ingestion and recall symmetric, wildcards are folded at the
    storage boundary: a trailing ``/*`` or ``/**`` becomes a directory
    prefix (``src/auth/*`` -> ``src/auth/``), stray ``*`` are stripped
    (``*.py`` -> ``.py`` is meaningless as a prefix, so an asterisk-only
    tail collapses to the literal part), and whitespace is trimmed.
    """
    pattern = (pattern or "").strip()
    if not pattern:
        return ""
    # Collapse directory-tail wildcards to a directory prefix.
    for suffix in ("/**", "/*"):
        if pattern.endswith(suffix):
            pattern = pattern[: -len(suffix)] + "/"
            break
    # A dangling wildcard mid-pattern cannot be expressed as a prefix;
    # drop it and keep the literal head (best-effort).
    pattern = pattern.replace("*", "")
    return pattern


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
        # External-content FTS index over feedback_content. The base table
        # schema is unchanged; the index is fully derivable from it and can
        # be rebuilt at any time. 'missing'-type rows stay indexed too —
        # type exclusion happens at query time (search_feedback).
        conn.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS feedback_fts USING fts5(
                feedback_content, content='feedback', content_rowid='id')"""
        )
        # Re-sync the index so pre-existing feedback rows are searchable.
        conn.execute("INSERT INTO feedback_fts(feedback_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()

    def add_feedback(self, thread_id: str, file_pattern: str, rule_id: str,
                     feedback_type: str, content: str) -> None:
        # Normalize at the storage boundary so every stored pattern works
        # under search_feedback's equality/prefix semantics.
        file_pattern = normalize_file_pattern(file_pattern)
        conn = sqlite3.connect(self._db_path)
        cur = conn.execute(
            "INSERT INTO feedback (thread_id, file_pattern, rule_id, feedback_type, feedback_content) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, file_pattern, rule_id, feedback_type, content),
        )
        # Keep the external-content FTS index in sync with the new row.
        conn.execute(
            "INSERT INTO feedback_fts(rowid, feedback_content) VALUES (?, ?)",
            (cur.lastrowid, content),
        )
        conn.commit()
        conn.close()

    def search_feedback(
        self,
        file_path: str,
        symbol_names: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Recall Judge-relevant feedback for one evidence file.

        Two-stage recall (fixed contract, consumed only by the Judge):

        1. SQL filter — keep rows whose ``file_pattern`` matches the
           evidence ``file_path`` by equality or prefix, excluding
           ``feedback_type = 'missing'`` (missed-risk feedback concerns
           the Planner/Reflection, not the Judge's adjudication).
        2. FTS match — within the filtered set, match the
           evidence-involved ``symbol_names`` against
           ``feedback_content`` (quoted phrases, OR-joined), ranked by
           bm25, ``LIMIT`` applied.

        Returns [] when no symbols are supplied (symbol matching is the
        core of the second stage) or when the FTS index is unavailable.
        """
        clean = [name.strip() for name in symbol_names if name and name.strip()]
        if not clean:
            return []
        # Double embedded quotes per the FTS5 phrase-quoting rule.
        match_query = " OR ".join(
            '"' + name.replace('"', '""') + '"' for name in clean
        )
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT f.* FROM feedback f
                JOIN feedback_fts ON f.id = feedback_fts.rowid
                WHERE f.feedback_type != 'missing'
                  AND (? = f.file_pattern
                       OR substr(?, 1, length(f.file_pattern)) = f.file_pattern)
                  AND feedback_fts MATCH ?
                ORDER BY bm25(feedback_fts)
                LIMIT ?
                """,
                (file_path, file_path, match_query, limit),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def get_all_feedback(self) -> list[dict]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM feedback").fetchall()
        conn.close()
        return [dict(r) for r in rows]
