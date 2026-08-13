"""RAG Retriever with hybrid search (embedding + FTS5 + RRF fusion)."""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any


class RAGRetriever:
    """Hybrid retrieval over past risks, security knowledge, and codebase.

    Combines embedding-based semantic search with FTS5 keyword search,
    then fuses the rankings using Reciprocal Rank Fusion (RRF).
    """

    # RRF constant
    _RRF_K = 60

    def __init__(self, db_path: str, embedding_client: Any) -> None:
        self._db_path = db_path
        self._embedding_client = embedding_client
        self._init_tables()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            # -- past_risks --
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS past_risks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    diff_summary TEXT,
                    risk_titles TEXT,
                    risk_categories TEXT,
                    overall_score REAL,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_past_risks_file "
                "ON past_risks(file_path)"
            )

            # -- security_knowledge --
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    category TEXT,
                    rule_id TEXT,
                    content TEXT,
                    best_practice TEXT,
                    "references" TEXT,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_security_knowledge_rule "
                "ON security_knowledge(rule_id)"
            )

            # -- codebase_index --
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS codebase_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    symbol_name TEXT,
                    symbol_type TEXT,
                    line_range TEXT,
                    content TEXT,
                    imports TEXT,
                    imported_by TEXT,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_codebase_file "
                "ON codebase_index(file_path)"
            )

            # -- FTS5 virtual tables --
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS past_risks_fts "
                "USING fts5(diff_summary, risk_titles, risk_categories, "
                "file_path, content='past_risks', content_rowid='id')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS security_knowledge_fts "
                "USING fts5(title, content, best_practice, "
                "content='security_knowledge', content_rowid='id')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS codebase_index_fts "
                "USING fts5(file_path, symbol_name, content, "
                "content='codebase_index', content_rowid='id')"
            )

            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_history(
        self,
        thread_id: str,
        file_path: str,
        diff_summary: str,
        risk_titles: list[str],
        risk_categories: list[str],
        overall_score: float,
    ) -> None:
        """Insert a past-risk record and compute its embedding."""
        embedding = self._embedding_client.embed(
            f"{file_path} {diff_summary} {' '.join(risk_titles)}"
        )
        embedding_str = json.dumps(embedding)
        risk_titles_str = json.dumps(risk_titles)
        risk_categories_str = json.dumps(risk_categories)
        created_at = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO past_risks
                    (thread_id, file_path, diff_summary, risk_titles,
                     risk_categories, overall_score, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    file_path,
                    diff_summary,
                    risk_titles_str,
                    risk_categories_str,
                    overall_score,
                    embedding_str,
                    created_at,
                ),
            )
            rowid = cursor.lastrowid

            # Sync FTS5 external-content table
            conn.execute(
                "INSERT INTO past_risks_fts(rowid, diff_summary, risk_titles, "
                "risk_categories, file_path) VALUES (?, ?, ?, ?, ?)",
                (rowid, diff_summary, risk_titles_str, risk_categories_str, file_path),
            )
            conn.commit()
        finally:
            conn.close()

    def add_security_knowledge(
        self,
        title: str,
        category: str,
        rule_id: str,
        content: str,
        best_practice: str,
        references: list[str],
        embedding: list[float] | None = None,
    ) -> None:
        """Insert a security-knowledge record."""
        if embedding is None:
            embedding = self._embedding_client.embed(f"{title} {content}")
        embedding_str = json.dumps(embedding)
        references_str = json.dumps(references)
        created_at = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO security_knowledge
                    (title, category, rule_id, content, best_practice,
                     "references", embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    title,
                    category,
                    rule_id,
                    content,
                    best_practice,
                    references_str,
                    embedding_str,
                    created_at,
                ),
            )
            rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO security_knowledge_fts(rowid, title, content, "
                "best_practice) VALUES (?, ?, ?, ?)",
                (rowid, title, content, best_practice),
            )
            conn.commit()
        finally:
            conn.close()

    def add_codebase_entry(
        self,
        file_path: str,
        symbol_name: str,
        symbol_type: str,
        line_range: str,
        content: str,
        imports: list[str],
        imported_by: list[str],
        embedding: list[float] | None = None,
    ) -> None:
        """Insert a codebase-index record."""
        if embedding is None:
            embedding = self._embedding_client.embed(
                f"{file_path} {symbol_name} {content}"
            )
        embedding_str = json.dumps(embedding)
        imports_str = json.dumps(imports)
        imported_by_str = json.dumps(imported_by)
        created_at = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                """
                INSERT INTO codebase_index
                    (file_path, symbol_name, symbol_type, line_range, content,
                     imports, imported_by, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    symbol_name,
                    symbol_type,
                    line_range,
                    content,
                    imports_str,
                    imported_by_str,
                    embedding_str,
                    created_at,
                ),
            )
            rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO codebase_index_fts(rowid, file_path, symbol_name, "
                "content) VALUES (?, ?, ?, ?)",
                (rowid, file_path, symbol_name, content),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Search operations (hybrid: embedding + FTS5 + RRF)
    # ------------------------------------------------------------------

    def search_history(
        self,
        query: str,
        file_pattern: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search past risks by embedding + FTS5 + RRF."""
        query_embedding = self._embedding_client.embed(query)

        # --- Embedding (semantic) candidates ---
        embedding_results = self._embedding_search_past_risks(
            query_embedding, file_pattern, top_k * 3
        )

        # --- FTS5 (keyword) candidates ---
        fts_results = self._fts_search_past_risks(query, file_pattern, top_k * 3)

        # --- RRF fusion ---
        fused = self._rrf_fuse(embedding_results, fts_results, self._RRF_K, top_k)

        # Hydrate full records
        return self._hydrate_past_risks(fused)

    def search_security(
        self,
        query: str,
        rule_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search security knowledge by embedding + FTS5 + RRF."""
        query_embedding = self._embedding_client.embed(query)

        # --- Embedding (semantic) candidates ---
        embedding_results = self._embedding_search_security(
            query_embedding, rule_ids, top_k * 3
        )

        # --- FTS5 (keyword) candidates ---
        fts_results = self._fts_search_security(query, rule_ids, top_k * 3)

        # --- RRF fusion ---
        fused = self._rrf_fuse(embedding_results, fts_results, self._RRF_K, top_k)

        # Hydrate full records
        return self._hydrate_security(fused)

    def search_codebase(
        self,
        file_path: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """FTS5 search codebase index by file path / symbol keywords."""
        conn = sqlite3.connect(self._db_path)
        try:
            # Build FTS5 query from file_path components
            fts_query = self._build_fts_query(file_path)
            if not fts_query:
                return []

            rows = conn.execute(
                """
                SELECT c.id, c.file_path, c.symbol_name, c.symbol_type,
                       c.line_range, c.content, c.imports, c.imported_by
                FROM codebase_index_fts f
                JOIN codebase_index c ON c.id = f.rowid
                WHERE codebase_index_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()

            results: list[dict[str, Any]] = []
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "file_path": row[1],
                        "symbol_name": row[2],
                        "symbol_type": row[3],
                        "line_range": row[4],
                        "content": row[5],
                        "imports": json.loads(row[6]) if row[6] else [],
                        "imported_by": json.loads(row[7]) if row[7] else [],
                    }
                )
            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Embedding search helpers
    # ------------------------------------------------------------------

    def _embedding_search_past_risks(
        self,
        query_embedding: list[float],
        file_pattern: str | None,
        limit: int,
    ) -> list[int]:
        """Return row IDs of past_risks ranked by cosine similarity."""
        conn = sqlite3.connect(self._db_path)
        try:
            sql = "SELECT id, file_path, embedding FROM past_risks"
            params: list[Any] = []
            if file_pattern:
                sql += " WHERE file_path LIKE ?"
                params.append(file_pattern.replace("*", "%"))
            sql += " LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()

            scored: list[tuple[float, int]] = []
            for row in rows:
                row_id, file_path, embedding_str = row
                if not embedding_str:
                    continue
                emb = json.loads(embedding_str)
                sim = self._cosine_similarity(query_embedding, emb)
                scored.append((sim, row_id))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [row_id for _, row_id in scored]
        finally:
            conn.close()

    def _embedding_search_security(
        self,
        query_embedding: list[float],
        rule_ids: list[str] | None,
        limit: int,
    ) -> list[int]:
        """Return row IDs of security_knowledge ranked by cosine similarity."""
        conn = sqlite3.connect(self._db_path)
        try:
            if rule_ids:
                placeholders = ",".join("?" * len(rule_ids))
                sql = (
                    f"SELECT id, embedding FROM security_knowledge "
                    f"WHERE rule_id IN ({placeholders}) LIMIT ?"
                )
                params: list[Any] = list(rule_ids) + [limit]
            else:
                sql = "SELECT id, embedding FROM security_knowledge LIMIT ?"
                params = [limit]

            rows = conn.execute(sql, params).fetchall()

            scored: list[tuple[float, int]] = []
            for row in rows:
                row_id, embedding_str = row
                if not embedding_str:
                    continue
                emb = json.loads(embedding_str)
                sim = self._cosine_similarity(query_embedding, emb)
                scored.append((sim, row_id))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [row_id for _, row_id in scored]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # FTS5 search helpers
    # ------------------------------------------------------------------

    def _fts_search_past_risks(
        self,
        query: str,
        file_pattern: str | None,
        limit: int,
    ) -> list[int]:
        """Return row IDs from past_risks_fts ranked by FTS5."""
        conn = sqlite3.connect(self._db_path)
        try:
            fts_query = self._build_fts_query(query)
            if not fts_query:
                return []

            # Combine query terms with optional file pattern
            pattern_clause = ""
            params: list[Any] = [fts_query]
            if file_pattern:
                # Use LIKE on the joined table for file filtering
                pattern_clause = " AND c.file_path LIKE ?"
                params.append(file_pattern.replace("*", "%"))

            sql = (
                "SELECT c.id FROM past_risks_fts f "
                "JOIN past_risks c ON c.id = f.rowid "
                "WHERE past_risks_fts MATCH ?"
                + pattern_clause
                + " ORDER BY rank LIMIT ?"
            )
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            return [row[0] for row in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def _fts_search_security(
        self,
        query: str,
        rule_ids: list[str] | None,
        limit: int,
    ) -> list[int]:
        """Return row IDs from security_knowledge_fts ranked by FTS5."""
        conn = sqlite3.connect(self._db_path)
        try:
            fts_query = self._build_fts_query(query)
            if not fts_query:
                return []

            rule_clause = ""
            params: list[Any] = [fts_query]
            if rule_ids:
                placeholders = ",".join("?" * len(rule_ids))
                rule_clause = f" AND c.rule_id IN ({placeholders})"
                params.extend(rule_ids)

            sql = (
                "SELECT c.id FROM security_knowledge_fts f "
                "JOIN security_knowledge c ON c.id = f.rowid "
                "WHERE security_knowledge_fts MATCH ?"
                + rule_clause
                + " ORDER BY rank LIMIT ?"
            )
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()
            return [row[0] for row in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        embedding_results: list[int],
        fts_results: list[int],
        k: int = 60,
        top_k: int = 10,
    ) -> list[int]:
        """Reciprocal Rank Fusion: score(id) = sum(1/(k+rank+1))."""
        scores: dict[int, float] = {}

        for rank, row_id in enumerate(embedding_results):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank + 1)

        for rank, row_id in enumerate(fts_results):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank + 1)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return sorted_ids[:top_k]

    # ------------------------------------------------------------------
    # Hydration helpers
    # ------------------------------------------------------------------

    def _hydrate_past_risks(self, row_ids: list[int]) -> list[dict[str, Any]]:
        if not row_ids:
            return []
        conn = sqlite3.connect(self._db_path)
        try:
            placeholders = ",".join("?" * len(row_ids))
            rows = conn.execute(
                f"SELECT id, thread_id, file_path, diff_summary, risk_titles, "
                f"risk_categories, overall_score, created_at "
                f"FROM past_risks WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

            row_map = {row[0]: row for row in rows}
            results: list[dict[str, Any]] = []
            for row_id in row_ids:
                row = row_map.get(row_id)
                if row is None:
                    continue
                results.append(
                    {
                        "id": row[0],
                        "thread_id": row[1],
                        "file_path": row[2],
                        "diff_summary": row[3],
                        "risk_titles": json.loads(row[4]) if row[4] else [],
                        "risk_categories": json.loads(row[5]) if row[5] else [],
                        "overall_score": row[6],
                        "created_at": row[7],
                    }
                )
            return results
        finally:
            conn.close()

    def _hydrate_security(self, row_ids: list[int]) -> list[dict[str, Any]]:
        if not row_ids:
            return []
        conn = sqlite3.connect(self._db_path)
        try:
            placeholders = ",".join("?" * len(row_ids))
            rows = conn.execute(
                f"SELECT id, title, category, rule_id, content, best_practice, "
                f'"references", created_at '
                f"FROM security_knowledge WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()

            row_map = {row[0]: row for row in rows}
            results: list[dict[str, Any]] = []
            for row_id in row_ids:
                row = row_map.get(row_id)
                if row is None:
                    continue
                results.append(
                    {
                        "id": row[0],
                        "title": row[1],
                        "category": row[2],
                        "rule_id": row[3],
                        "content": row[4],
                        "best_practice": row[5],
                        "references": json.loads(row[6]) if row[6] else [],
                        "created_at": row[7],
                    }
                )
            return results
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _build_fts_query(text: str) -> str:
        """Build an FTS5 MATCH query: wrap each token in double quotes."""
        tokens = text.replace('"', " ").split()
        if not tokens:
            return ""
        return " ".join(f'"{t}"' for t in tokens)
