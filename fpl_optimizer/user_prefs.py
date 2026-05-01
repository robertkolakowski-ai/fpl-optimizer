"""Lightweight user preferences store, keyed by FPL team ID.

Pragmatic alternative to full magic-link auth: the FPL team ID is already a
public identifier the user pastes in. We store per-team-id preferences
(risk mode, watched players, alert thresholds) in SQLite — keeps the user
state across sessions without an account system.

When/if real auth is added (#5 full version), migrate this table by mapping
team_id → user_id.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
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
    return d / "user_prefs.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_prefs (
            team_id INTEGER PRIMARY KEY,
            prefs_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watched_players (
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, player_id)
        )
    """)
    return conn


_DEFAULT_PREFS: dict[str, Any] = {
    "risk_mode": "balanced",
    "alert_price_change": True,
    "alert_injury": True,
    "alert_rotation": False,
    "language": "no",
}


def get_prefs(team_id: int) -> dict[str, Any]:
    with _LOCK, _conn() as conn:
        row = conn.execute(
            "SELECT prefs_json FROM user_prefs WHERE team_id = ?", (team_id,)
        ).fetchone()
    if not row:
        return dict(_DEFAULT_PREFS)
    try:
        prefs = json.loads(row["prefs_json"])
    except Exception:
        return dict(_DEFAULT_PREFS)
    out = dict(_DEFAULT_PREFS)
    out.update(prefs)
    return out


def set_prefs(team_id: int, prefs: dict[str, Any]) -> None:
    merged = get_prefs(team_id)
    merged.update(prefs)
    with _LOCK, _conn() as conn:
        conn.execute(
            "INSERT INTO user_prefs (team_id, prefs_json) VALUES (?, ?) "
            "ON CONFLICT(team_id) DO UPDATE SET prefs_json = excluded.prefs_json, "
            "updated_at = CURRENT_TIMESTAMP",
            (team_id, json.dumps(merged)),
        )


def get_watched(team_id: int) -> list[int]:
    with _LOCK, _conn() as conn:
        rows = conn.execute(
            "SELECT player_id FROM watched_players WHERE team_id = ? ORDER BY added_at",
            (team_id,),
        ).fetchall()
    return [r["player_id"] for r in rows]


def add_watched(team_id: int, player_id: int) -> None:
    with _LOCK, _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watched_players (team_id, player_id) VALUES (?, ?)",
            (team_id, player_id),
        )


def remove_watched(team_id: int, player_id: int) -> None:
    with _LOCK, _conn() as conn:
        conn.execute(
            "DELETE FROM watched_players WHERE team_id = ? AND player_id = ?",
            (team_id, player_id),
        )


def list_known_team_ids() -> list[int]:
    with _LOCK, _conn() as conn:
        rows = conn.execute("SELECT team_id FROM user_prefs").fetchall()
    return [r["team_id"] for r in rows]
