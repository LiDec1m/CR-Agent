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

    def clear(self) -> None:
        """Clear all security knowledge entries (table + FTS index)."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("DELETE FROM security_knowledge_fts")
            conn.execute("DELETE FROM security_knowledge")
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name='security_knowledge'"
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

    TODO: Multi-language AST support.
      Currently only Python files are supported via the stdlib ``ast`` module.
      To support JS/TS/Java/Go, introduce a pluggable parser interface:

      1. Replace ``ast.parse()`` with a ``LanguageParser`` abstraction that
         delegates to ``tree-sitter`` (supports 30+ languages) or
         ``libcst`` (precise Python/JS CST).
      2. Map language-specific AST node types:
         - Python: ``ast.FunctionDef`` / ``ast.ClassDef``
         - JS/TS: ``FunctionDeclaration`` / ``ClassDeclaration``
         - Java: ``MethodDeclaration`` / ``ClassDeclaration``
         - Go: ``FunctionDecl`` / ``TypeDeclaration``
      3. Update ``_extract_imports()`` to handle each language's import syntax
         (e.g. JS ``import/require``, Java ``import``, Go ``import``).
      4. Update ``cr-agent index`` CLI to walk non-``.py`` files.
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

    def clear_index(self) -> None:
        """Clear all codebase index entries (table + FTS index).

        Call this before re-indexing an entire codebase to avoid duplicates.
        Use ``delete_by_file()`` instead if you only want to re-index specific
        files while keeping the rest of the index intact.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("DELETE FROM codebase_index_fts")
            conn.execute("DELETE FROM codebase_index")
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name='codebase_index'"
            )
            conn.commit()
        finally:
            conn.close()

    def delete_by_file(self, file_path: str) -> int:
        """Delete all index entries for a specific file.

        Use this before re-indexing a single file to avoid duplicates,
        while preserving the rest of the index. Returns the number of
        records deleted.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            # Get rowids to delete from FTS table too
            rowids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM codebase_index WHERE file_path = ?",
                    (file_path,),
                ).fetchall()
            ]
            if not rowids:
                return 0
            placeholders = ",".join("?" for _ in rowids)
            conn.execute(
                f"DELETE FROM codebase_index_fts "
                f"WHERE rowid IN ({placeholders})",
                rowids,
            )
            conn.execute(
                "DELETE FROM codebase_index WHERE file_path = ?",
                (file_path,),
            )
            conn.commit()
            return len(rowids)
        finally:
            conn.close()

    def resolve_imports(self) -> int:
        """Resolve direct imports to file paths and populate resolved_imports.

        Iterates all indexed files, resolves their direct imports to file
        paths, and writes the resolved paths to the ``resolved_imports``
        field so that ``search_codebase`` can use them directly without
        re-resolving at query time.

        Only handles direct absolute imports (e.g. ``src.rag.retriever``).
        Relative imports (``from . import x``), aliased imports
        (``import x as y``), and star imports are not resolved.

        Returns the number of resolved import links.
        """
        conn = sqlite3.connect(self._db_path)
        try:
            indexed_paths = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT file_path FROM codebase_index"
                ).fetchall()
            }

            resolved_map: dict[int, list[str]] = {}
            links = 0

            for row in conn.execute(
                "SELECT id, file_path, imports FROM codebase_index"
            ).fetchall():
                rid = row[0]
                importer_path = row[1]
                imports_str = row[2]
                if not imports_str:
                    continue
                try:
                    imports_list = json.loads(imports_str)
                except (json.JSONDecodeError, TypeError):
                    continue

                resolved_list: list[str] = []
                for imp in imports_list:
                    resolved = self._resolve_import_to_path(imp, indexed_paths)
                    if resolved and resolved != importer_path:
                        resolved_list.append(resolved)
                        links += 1

                if resolved_list:
                    resolved_map[rid] = resolved_list

            self._ensure_resolved_imports_column(conn)

            for rid, resolved_list in resolved_map.items():
                conn.execute(
                    "UPDATE codebase_index SET resolved_imports = ? "
                    "WHERE id = ?",
                    (json.dumps(resolved_list), rid),
                )
            for row in conn.execute(
                "SELECT id FROM codebase_index WHERE resolved_imports IS NULL"
            ).fetchall():
                conn.execute(
                    "UPDATE codebase_index SET resolved_imports = '[]' "
                    "WHERE id = ?",
                    (row[0],),
                )

            conn.commit()
            return links
        finally:
            conn.close()

    @staticmethod
    def _ensure_resolved_imports_column(conn: sqlite3.Connection) -> None:
        """Add resolved_imports column if it doesn't exist."""
        try:
            conn.execute(
                "ALTER TABLE codebase_index "
                "ADD COLUMN resolved_imports TEXT DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            # Column already exists
            pass

    @staticmethod
    def _resolve_import_to_path(
        module_path: str, indexed_paths: set[str]
    ) -> str | None:
        """Resolve a Python import path to an indexed file path.

        Handles ``src.rag.retriever`` → ``src/rag/retriever.py``.
        Also handles ``from src.rag.retriever import RAGRetriever``
        by stripping the last component if no direct match is found.
        Returns None if no match is found in indexed_paths.
        """
        if not module_path:
            return None

        # Convert dotted path to file path
        file_path = module_path.replace(".", "/") + ".py"

        if file_path in indexed_paths:
            return file_path

        # Try stripping the last component (e.g. src.rag.retriever.RAGRetriever
        # → src.rag.retriever → src/rag/retriever.py)
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            parent_path = parts[0].replace(".", "/") + ".py"
            if parent_path in indexed_paths:
                return parent_path

        return None

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

        # Batch-compute all symbol embeddings in a single API call.
        # Each text mirrors the query-side format used by search_codebase:
        # file path + symbol name + truncated content.
        embedding_texts = [
            f"{file_path} {sym['name']} {sym.get('content', '')[:500]}"
            for sym in symbols
        ]
        embeddings = self._embedding_client.embed_batch(embedding_texts)
        imports_str = json.dumps(imports)

        conn = sqlite3.connect(self._db_path)
        try:
            count = 0
            for sym, embedding in zip(symbols, embeddings):
                symbol_name = sym["name"]
                symbol_type = sym["type"]
                line_range = sym["lines"]
                sym_content = sym.get("content", "")
                embedding_str = json.dumps(embedding)
                created_at = datetime.now(timezone.utc).isoformat()

                cursor = conn.execute(
                    """
                    INSERT INTO codebase_index
                        (file_path, symbol_name, symbol_type, line_range,
                         content, imports, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        symbol_name,
                        symbol_type,
                        line_range,
                        sym_content,
                        imports_str,
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
