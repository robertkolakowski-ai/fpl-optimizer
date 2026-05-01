"""Persistent shared cache backed by SQLite.

Best-practice replacement for the in-memory dict cache:
  - Survives gunicorn restarts and multiple worker processes
  - Atomic via SQLite WAL mode
  - Generates ETags for HTTP-level browser caching (304 Not Modified)

Usage:
    from .cache import sqlite_cache_get, sqlite_cache_set, etag_for

    val = sqlite_cache_get("fpl_bootstrap", ttl_seconds=300)
    if val is None:
        val = expensive_fetch()
        sqlite_cache_set("fpl_bootstrap", val)

    # Flask response:
    etag = etag_for(val)
    if request.headers.get("If-None-Match") == etag:
        return ("", 304)
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "private, max-age=60"
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _data_dir() -> Path:
    base = os.environ.get("FPL_DATA_DIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data"


def _db_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "cache.db"


_DB_INITIALIZED = False


def _conn() -> sqlite3.Connection:
    global _DB_INITIALIZED
    conn = sqlite3.connect(_db_path(), timeout=10)
    if not _DB_INITIALIZED:
        # WAL mode for concurrent readers + atomic writes across workers
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                fetched_at REAL NOT NULL,
                etag TEXT NOT NULL
            )
        """)
        conn.commit()
        _DB_INITIALIZED = True
    return conn


def etag_for(value: Any) -> str:
    """Stable, deterministic ETag for any JSON-serializable value."""
    serialized = json.dumps(value, sort_keys=True, default=str).encode()
    return f'W/"{hashlib.md5(serialized).hexdigest()}"'


def sqlite_cache_get(key: str, ttl_seconds: int = 300) -> Any | None:
    """Returns cached value if present and within TTL, else None."""
    with _LOCK, _conn() as conn:
        row = conn.execute(
            "SELECT value_json, fetched_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return None
    value_json, fetched_at = row
    if (time.time() - fetched_at) > ttl_seconds:
        return None
    try:
        return json.loads(value_json)
    except Exception:
        return None


def sqlite_cache_get_with_etag(key: str, ttl_seconds: int = 300) -> tuple[Any | None, str | None]:
    """Returns (value, etag) — useful for endpoints that want to send 304."""
    with _LOCK, _conn() as conn:
        row = conn.execute(
            "SELECT value_json, fetched_at, etag FROM cache WHERE key = ?", (key,)
        ).fetchone()
    if not row:
        return None, None
    value_json, fetched_at, etag = row
    if (time.time() - fetched_at) > ttl_seconds:
        return None, None
    try:
        return json.loads(value_json), etag
    except Exception:
        return None, None


def sqlite_cache_set(key: str, value: Any) -> str:
    """Stores `value` under `key`, returns its ETag."""
    serialized = json.dumps(value, default=str)
    etag = etag_for(value)
    with _LOCK, _conn() as conn:
        conn.execute(
            "INSERT INTO cache (key, value_json, fetched_at, etag) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value_json=excluded.value_json, fetched_at=excluded.fetched_at, etag=excluded.etag",
            (key, serialized, time.time(), etag),
        )
    return etag


def sqlite_cache_invalidate(key_prefix: str = "") -> int:
    """Drops all entries (or those matching a prefix). Returns count deleted."""
    with _LOCK, _conn() as conn:
        if key_prefix:
            cur = conn.execute("DELETE FROM cache WHERE key LIKE ?", (key_prefix + "%",))
        else:
            cur = conn.execute("DELETE FROM cache")
        return cur.rowcount


def sqlite_cache_stats() -> dict[str, Any]:
    with _LOCK, _conn() as conn:
        rows = conn.execute(
            "SELECT key, fetched_at, length(value_json) AS sz FROM cache ORDER BY fetched_at DESC"
        ).fetchall()
    now = time.time()
    return {
        "n_keys": len(rows),
        "total_bytes": sum(r[2] for r in rows),
        "entries": [
            {
                "key": r[0],
                "age_seconds": int(now - r[1]),
                "size_bytes": r[2],
            }
            for r in rows[:50]
        ],
    }
