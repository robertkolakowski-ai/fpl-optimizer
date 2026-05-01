"""Persistent score-history snapshots.

Captures top-N composite_score per gameweek so we can later compare model
predictions to actual FPL points (the foundation for #2 backtest and #1
historical persistence).

Storage mirrors predictions_log.py:
  - JSON file at ``$FPL_DATA_DIR/score_history.json`` (env override, default <repo>/data/)
  - Atomic write via temp file + rename
  - Render free filesystem is ephemeral; rely on GH Actions cron to commit back
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _data_dir() -> Path:
    base = os.environ.get("FPL_DATA_DIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data"


def _store_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "score_history.json"


_DEFAULT_STATE: dict[str, Any] = {"version": 1, "snapshots": []}


def _load() -> dict[str, Any]:
    p = _store_path()
    if not p.exists():
        return {"version": 1, "snapshots": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("snapshots", [])
        return data
    except Exception:
        try:
            p.rename(p.with_suffix(".json.corrupt"))
        except Exception:
            pass
        return {"version": 1, "snapshots": []}


def _save(state: dict[str, Any]) -> None:
    p = _store_path()
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


_state: dict[str, Any] = _load()


def get_state() -> dict[str, Any]:
    with _LOCK:
        return {
            "version": _state.get("version", 1),
            "snapshots": list(_state.get("snapshots", [])),
        }


def record_snapshot(
    gw: int,
    players: list,
    *,
    snapshot_at: str | None = None,
    top_n: int = 50,
) -> int:
    """Snapshot top-N players by composite_score for a GW.

    Idempotent on `gw` — re-recording replaces any existing snapshot for that GW.
    Returns count of player rows persisted.
    """
    ranked = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_n]
    rows = [
        {
            "id": p.id,
            "name": p.name,
            "team": p.team,
            "position": p.position,
            "cost": p.cost,
            "composite_score": round(p.composite_score, 4),
            "form": p.form,
            "ppg": p.points_per_game,
            "ep_next": p.ep_next,
            "fixture_difficulty": round(p.fixture_difficulty, 3),
            "actual_points": None,  # filled in later by update_actuals
        }
        for p in ranked
    ]
    with _LOCK:
        _state["snapshots"] = [s for s in _state["snapshots"] if s.get("gameweek") != gw]
        _state["snapshots"].append({
            "gameweek": gw,
            "snapshot_at": snapshot_at,
            "players": rows,
        })
        _save(_state)
        return len(rows)


def update_actuals(gw: int, player_actual_points: dict[int, int]) -> int:
    """Fill in actual GW points for previously snapshotted players. Returns count updated."""
    with _LOCK:
        updated = 0
        for snap in _state["snapshots"]:
            if snap.get("gameweek") != gw:
                continue
            for row in snap.get("players", []):
                if row.get("actual_points") is None:
                    pts = player_actual_points.get(row["id"])
                    if pts is not None:
                        row["actual_points"] = pts
                        updated += 1
        if updated:
            _save(_state)
        return updated


def get_snapshot(gw: int) -> dict | None:
    with _LOCK:
        for snap in _state["snapshots"]:
            if snap.get("gameweek") == gw:
                return dict(snap)
        return None


def list_gameweeks() -> list[int]:
    with _LOCK:
        return sorted({s.get("gameweek") for s in _state["snapshots"] if s.get("gameweek")})
