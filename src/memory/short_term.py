"""Short-term memory: LangGraph SqliteSaver checkpointer wrapper."""

from __future__ import annotations

import sqlite3
from typing import Any


class _SerdeCompat:
    """Adapter bridging SqliteSaver 2.x calls to langgraph-checkpoint 4.x.

    langgraph-checkpoint-sqlite 2.x still calls the removed
    ``JsonPlusSerializer.dumps``/``.loads`` API, while the installed
    langgraph-checkpoint 4.x only provides ``dumps_typed``/``loads_typed``.
    Metadata payloads are plain dicts, which ``dumps_typed`` encodes with
    the fixed "msgpack" type tag, so the translation is lossless.
    """

    def __init__(self, serde: Any) -> None:
        self._serde = serde

    def dumps(self, obj: Any) -> bytes:
        type_tag, payload = self._serde.dumps_typed(obj)
        assert type_tag == "msgpack", f"unexpected type tag: {type_tag}"
        return payload

    def loads(self, data: bytes) -> Any:
        return self._serde.loads_typed(("msgpack", data))


class _SaverContext:
    """Owns the DB connection and exposes the saver via ``with``.

    ``SqliteSaver.from_conn_string`` is a @contextmanager generator; calling
    ``__enter__`` on it and dropping the context object lets GC close the
    generator (and the connection) at an arbitrary point. Owning an explicit
    ``sqlite3.Connection`` avoids that trap entirely.
    """

    def __init__(self, db_path: str) -> None:
        from langgraph.checkpoint.sqlite import SqliteSaver

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._saver = SqliteSaver(self._conn)
        # Patch only when the underlying serializer lacks the legacy API
        # (langgraph-checkpoint >= 3.0 removed it).
        if not hasattr(self._saver.jsonplus_serde, "dumps"):
            self._saver.jsonplus_serde = _SerdeCompat(self._saver.jsonplus_serde)

    def __enter__(self) -> Any:
        return self._saver

    def __exit__(self, *exc_info: Any) -> None:
        self._conn.close()


class ShortTermMemory:
    """Wraps LangGraph SqliteSaver for state persistence."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def get_checkpointer(self) -> Any:
        return _SaverContext(self._db_path)
