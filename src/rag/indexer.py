"""Security knowledge loader and codebase indexer for the RAG subsystem."""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SecurityKnowledgeLoader:
    """Load security-knowledge entries into the SQLite database."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def init_tables(self) -> None:
        """Create the security_knowledge table + FTS5 virtual table."""
        conn = sqlite3.connect(self._db_path)
        try:
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
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS security_knowledge_fts "
                "USING fts5(title, content, best_practice, "
                "content='security_knowledge', content_rowid='id')"
            )
            conn.commit()
        finally:
            conn.close()

    def load_from_json(
        self,
        json_path: str,
        embedding_client: Any | None = None,
    ) -> int:
        """Load knowledge entries from a JSON file.

        Returns the number of entries inserted.
        """
        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)

        conn = sqlite3.connect(self._db_path)
        try:
            count = 0
            for entry in entries:
                title = entry.get("title", "")
                category = entry.get("category", "")
                rule_id = entry.get("rule_id", "")
                content = entry.get("content", "")
                best_practice = entry.get("best_practice", "")
                references = entry.get("references", [])
                references_str = json.dumps(references)

                if embedding_client is not None:
                    embedding_str = json.dumps(
                        embedding_client.embed(f"{title} {content}")
                    )
                else:
                    embedding_str = None

                created_at = datetime.now(timezone.utc).isoformat()

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
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()


class CodebaseIndexer:
    """Index Python source files into the codebase_index table.

    Uses ``ast`` to extract function and class symbols.
    """

    def __init__(self, db_path: str, embedding_client: Any) -> None:
        self._db_path = db_path
        self._embedding_client = embedding_client

    def init_tables(self) -> None:
        """Create the codebase_index table + FTS5 virtual table."""
        conn = sqlite3.connect(self._db_path)
        try:
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
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS codebase_index_fts "
                "USING fts5(file_path, symbol_name, content, "
                "content='codebase_index', content_rowid='id')"
            )
            conn.commit()
        finally:
            conn.close()

    def index_file(self, file_path: str, content: str) -> int:
        """Parse a Python file and index its symbols.

        Returns the number of symbols indexed.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None

        imports = self._extract_imports(content) if tree is not None else []

        symbols = (
            self._extract_symbols_from_source(tree, content)
            if tree is not None
            else []
        )

        # If no symbols were found, index the entire file as one entry
        if not symbols:
            symbols = [
                {
                    "name": Path(file_path).stem,
                    "type": "module",
                    "lines": "1-end",
                    "content": content,
                }
            ]

        conn = sqlite3.connect(self._db_path)
        try:
            count = 0
            for sym in symbols:
                symbol_name = sym["name"]
                symbol_type = sym["type"]
                line_range = sym["lines"]
                sym_content = sym.get("content", "")

                embedding = self._embedding_client.embed(
                    f"{file_path} {symbol_name} {sym_content[:500]}"
                )
                embedding_str = json.dumps(embedding)
                imports_str = json.dumps(imports)
                created_at = datetime.now(timezone.utc).isoformat()

                cursor = conn.execute(
                    """
                    INSERT INTO codebase_index
                        (file_path, symbol_name, symbol_type, line_range,
                         content, imports, imported_by, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        symbol_name,
                        symbol_type,
                        line_range,
                        sym_content,
                        imports_str,
                        "[]",  # imported_by — populated later by cross-reference
                        embedding_str,
                        created_at,
                    ),
                )
                rowid = cursor.lastrowid

                conn.execute(
                    "INSERT INTO codebase_index_fts(rowid, file_path, "
                    "symbol_name, content) VALUES (?, ?, ?, ?)",
                    (rowid, file_path, symbol_name, sym_content),
                )
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_symbols(tree: ast.Module) -> list[dict[str, Any]]:
        """Extract function and class definitions from an AST."""
        symbols: list[dict[str, Any]] = []
        source_lines: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = getattr(node, "end_lineno", node.lineno)
                line_range = f"{node.lineno}-{end_line}"
                # Try to get the source segment for the function
                sym_content = ast.get_source_segment(
                    "\n".join(source_lines), node
                ) or ""
                symbols.append(
                    {
                        "name": node.name,
                        "type": "function",
                        "lines": line_range,
                        "content": sym_content,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                end_line = getattr(node, "end_lineno", node.lineno)
                line_range = f"{node.lineno}-{end_line}"
                sym_content = ast.get_source_segment(
                    "\n".join(source_lines), node
                ) or ""
                symbols.append(
                    {
                        "name": node.name,
                        "type": "class",
                        "lines": line_range,
                        "content": sym_content,
                    }
                )

        return symbols

    def index_file_full(self, file_path: str, content: str) -> int:
        """Parse a Python file and index its symbols (full-source version).

        This variant passes the original source to ``ast.get_source_segment``
        so that function/class bodies are captured correctly.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None

        imports = self._extract_imports(content) if tree is not None else []

        symbols = (
            self._extract_symbols_from_source(tree, content) if tree is not None else []
        )

        if not symbols:
            symbols = [
                {
                    "name": Path(file_path).stem,
                    "type": "module",
                    "lines": "1-end",
                    "content": content,
                }
            ]

        conn = sqlite3.connect(self._db_path)
        try:
            count = 0
            for sym in symbols:
                symbol_name = sym["name"]
                symbol_type = sym["type"]
                line_range = sym["lines"]
                sym_content = sym.get("content", "")

                embedding = self._embedding_client.embed(
                    f"{file_path} {symbol_name} {sym_content[:500]}"
                )
                embedding_str = json.dumps(embedding)
                imports_str = json.dumps(imports)
                created_at = datetime.now(timezone.utc).isoformat()

                cursor = conn.execute(
                    """
                    INSERT INTO codebase_index
                        (file_path, symbol_name, symbol_type, line_range,
                         content, imports, imported_by, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        symbol_name,
                        symbol_type,
                        line_range,
                        sym_content,
                        imports_str,
                        "[]",
                        embedding_str,
                        created_at,
                    ),
                )
                rowid = cursor.lastrowid

                conn.execute(
                    "INSERT INTO codebase_index_fts(rowid, file_path, "
                    "symbol_name, content) VALUES (?, ?, ?, ?)",
                    (rowid, file_path, symbol_name, sym_content),
                )
                count += 1

            conn.commit()
            return count
        finally:
            conn.close()

    @staticmethod
    def _extract_symbols_from_source(
        tree: ast.Module, source: str
    ) -> list[dict[str, Any]]:
        """Extract symbols using the full source text for body extraction."""
        symbols: list[dict[str, Any]] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_line = getattr(node, "end_lineno", node.lineno)
                line_range = f"{node.lineno}-{end_line}"
                sym_content = ast.get_source_segment(source, node) or ""
                symbols.append(
                    {
                        "name": node.name,
                        "type": "function",
                        "lines": line_range,
                        "content": sym_content,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                end_line = getattr(node, "end_lineno", node.lineno)
                line_range = f"{node.lineno}-{end_line}"
                sym_content = ast.get_source_segment(source, node) or ""
                symbols.append(
                    {
                        "name": node.name,
                        "type": "class",
                        "lines": line_range,
                        "content": sym_content,
                    }
                )

        return symbols

    @staticmethod
    def _extract_imports(content: str) -> list[str]:
        """Extract import module paths from Python source."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
