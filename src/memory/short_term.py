"""Short-term memory: LangGraph SqliteSaver checkpointer wrapper."""

from __future__ import annotations

from typing import Any


class ShortTermMemory:
    """Wraps LangGraph SqliteSaver for state persistence."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def get_checkpointer(self) -> Any:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(self._db_path)
