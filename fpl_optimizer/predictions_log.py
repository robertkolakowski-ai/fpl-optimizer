"""Persistent predictions log.

Stores model recommendations per GW so we can compute real hit-rate over time.
Two streams:
  1. Fixture-level: 1X2 / O2.5 / BTTS (from predictions.py output)
  2. FPL-recommendation level: captain pick, top differential, recommended transfer

Storage:
  - JSON file at ``$FPL_DATA_DIR/predictions_log.json`` (env-overridable, default <repo>/data/)
  - Loaded on import; saved after every mutation (atomic via temp file + rename)
  - Render free filesystem is ephemeral on redeploy. The GH Actions cron job
    (see .github/workflows/predictions-snapshot.yml) is the canonical persistence
    layer — it commits the JSON back to the repo nightly.
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
    # Default: <repo_root>/data — repo_root is parent of fpl_optimizer/
    return Path(__file__).resolve().parent.parent / "data"


def _store_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "predictions_log.json"


_DEFAULT_STATE: dict[str, Any] = {
    "version": 1,
    "fixtures": [],          # list[dict] — same shape as in-memory _prediction_history
    "fpl_recommendations": [],  # list[dict] — per-GW recommendations
}


def _load() -> dict[str, Any]:
    p = _store_path()
    if not p.exists():
        return dict(_DEFAULT_STATE, fixtures=[], fpl_recommendations=[])
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Sanity: fill in missing keys
        for k, v in _DEFAULT_STATE.items():
            data.setdefault(k, v if not isinstance(v, list) else [])
        return data
    except Exception:
        # Corrupted file — start fresh, but keep a backup
        try:
            p.rename(p.with_suffix(".json.corrupt"))
        except Exception:
            pass
        return dict(_DEFAULT_STATE, fixtures=[], fpl_recommendations=[])


def _save(state: dict[str, Any]) -> None:
    p = _store_path()
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


# In-process cache (single-process Flask is fine; if we ever go multi-worker,
# disk is still the source of truth — just reload before mutating).
_state: dict[str, Any] = _load()


def get_state() -> dict[str, Any]:
    """Return a deep-ish copy safe for serialization."""
    with _LOCK:
        return {
            "version": _state.get("version", 1),
            "fixtures": list(_state.get("fixtures", [])),
            "fpl_recommendations": list(_state.get("fpl_recommendations", [])),
        }


def record_fixture_predictions(gw: int, predictions: list) -> int:
    """Record fixture-level predictions for a GW. Returns count added (0 if already present)."""
    with _LOCK:
        existing = {(p.get("gameweek"), p.get("fixture_id")) for p in _state["fixtures"]}
        added = 0
        for pred in predictions:
            key = (gw, pred.fixture_id)
            if key in existing:
                continue
            _state["fixtures"].append({
                "gameweek": gw,
                "fixture_id": pred.fixture_id,
                "home_team": pred.home_team_short,
                "away_team": pred.away_team_short,
                "home_xg": pred.home_xg,
                "away_xg": pred.away_xg,
                "predicted_1x2": "1" if pred.home_win_prob > max(pred.draw_prob, pred.away_win_prob)
                                else "X" if pred.draw_prob > pred.away_win_prob else "2",
                "home_win_prob": pred.home_win_prob,
                "draw_prob": pred.draw_prob,
                "away_win_prob": pred.away_win_prob,
                "predicted_over_25": pred.over_25 > 0.5,
                "over_25_prob": pred.over_25,
                "predicted_btts": pred.btts_yes > 0.5,
                "btts_prob": pred.btts_yes,
                "actual_home_score": None,
                "actual_away_score": None,
                "actual_1x2": None,
                "actual_over_25": None,
                "actual_btts": None,
            })
            added += 1
        if added:
            _save(_state)
        return added


def update_fixture_actuals(fixture_map: dict) -> int:
    """Fill in actual results for finished fixtures. Returns count updated."""
    with _LOCK:
        updated = 0
        for entry in _state["fixtures"]:
            if entry["actual_1x2"] is not None:
                continue
            f = fixture_map.get(entry["fixture_id"])
            if not f or not getattr(f, "finished", False) or getattr(f, "home_score", None) is None:
                continue
            entry["actual_home_score"] = f.home_score
            entry["actual_away_score"] = f.away_score
            if f.home_score > f.away_score:
                entry["actual_1x2"] = "1"
            elif f.home_score == f.away_score:
                entry["actual_1x2"] = "X"
            else:
                entry["actual_1x2"] = "2"
            entry["actual_over_25"] = (f.home_score + f.away_score) > 2.5
            entry["actual_btts"] = f.home_score > 0 and f.away_score > 0
            updated += 1
        if updated:
            _save(_state)
        return updated


def record_fpl_recommendation(
    gw: int,
    *,
    captain_id: int | None = None,
    captain_name: str | None = None,
    captain_expected: float | None = None,
    top_transfer: dict | None = None,
    optimal_squad_ids: list[int] | None = None,
    differential_pick_id: int | None = None,
    differential_pick_name: str | None = None,
    snapshot_at: str | None = None,
) -> bool:
    """Record (or update) the model's recommendations for a GW.

    Idempotent on (gw, snapshot_at) — re-recording for the same GW *replaces*
    the latest snapshot for that GW (we keep only the deadline-time snapshot
    in production; multiple snapshots per GW are allowed for debugging).
    """
    with _LOCK:
        # Replace any existing recommendation for this GW (deadline-time snapshot)
        _state["fpl_recommendations"] = [r for r in _state["fpl_recommendations"] if r.get("gameweek") != gw]
        _state["fpl_recommendations"].append({
            "gameweek": gw,
            "snapshot_at": snapshot_at,
            "captain_id": captain_id,
            "captain_name": captain_name,
            "captain_expected": captain_expected,
            "captain_actual": None,        # filled by update_fpl_actuals
            "top_transfer": top_transfer,  # {"out_id":..,"in_id":..,"out_name":..,"in_name":..,"delta_xpts":..}
            "transfer_actual_delta": None,
            "optimal_squad_ids": optimal_squad_ids or [],
            "differential_pick_id": differential_pick_id,
            "differential_pick_name": differential_pick_name,
            "differential_actual": None,
        })
        _save(_state)
        return True


def update_fpl_actuals(gw: int, player_actual_points: dict[int, int]) -> bool:
    """Fill in actual GW points for previously recorded FPL recommendations.

    Args:
        gw: Gameweek to update.
        player_actual_points: {player_id: actual_points_in_gw, ...}
    """
    with _LOCK:
        changed = False
        for r in _state["fpl_recommendations"]:
            if r.get("gameweek") != gw:
                continue
            if r.get("captain_id") and r.get("captain_actual") is None:
                pts = player_actual_points.get(r["captain_id"])
                if pts is not None:
                    r["captain_actual"] = pts
                    changed = True
            tt = r.get("top_transfer")
            if tt and r.get("transfer_actual_delta") is None:
                in_pts = player_actual_points.get(tt.get("in_id"))
                out_pts = player_actual_points.get(tt.get("out_id"))
                if in_pts is not None and out_pts is not None:
                    r["transfer_actual_delta"] = in_pts - out_pts
                    changed = True
            if r.get("differential_pick_id") and r.get("differential_actual") is None:
                pts = player_actual_points.get(r["differential_pick_id"])
                if pts is not None:
                    r["differential_actual"] = pts
                    changed = True
        if changed:
            _save(_state)
        return changed


def compute_hit_rate() -> dict[str, Any]:
    """Aggregate hit-rate statistics across all logged predictions."""
    with _LOCK:
        fixtures = list(_state["fixtures"])
        recs = list(_state["fpl_recommendations"])

    # Fixture-level
    settled_fx = [f for f in fixtures if f.get("actual_1x2")]
    onextwo_correct = sum(1 for f in settled_fx if f["predicted_1x2"] == f["actual_1x2"])
    o25_correct = sum(1 for f in settled_fx if f["predicted_over_25"] == f["actual_over_25"])
    btts_correct = sum(1 for f in settled_fx if f["predicted_btts"] == f["actual_btts"])

    # FPL-rec level
    cap_settled = [r for r in recs if r.get("captain_actual") is not None]
    cap_total_actual = sum(r["captain_actual"] for r in cap_settled)
    cap_total_expected = sum((r.get("captain_expected") or 0) for r in cap_settled)

    transfers_settled = [r for r in recs if r.get("transfer_actual_delta") is not None]
    transfers_winning = sum(1 for r in transfers_settled if r["transfer_actual_delta"] > 0)

    return {
        "fixtures": {
            "total_settled": len(settled_fx),
            "onextwo_correct": onextwo_correct,
            "onextwo_pct": round(100 * onextwo_correct / len(settled_fx), 1) if settled_fx else None,
            "over25_correct": o25_correct,
            "over25_pct": round(100 * o25_correct / len(settled_fx), 1) if settled_fx else None,
            "btts_correct": btts_correct,
            "btts_pct": round(100 * btts_correct / len(settled_fx), 1) if settled_fx else None,
        },
        "fpl_recommendations": {
            "captains_settled": len(cap_settled),
            "captain_actual_avg": round(cap_total_actual / len(cap_settled), 2) if cap_settled else None,
            "captain_expected_avg": round(cap_total_expected / len(cap_settled), 2) if cap_settled else None,
            "transfers_settled": len(transfers_settled),
            "transfers_winning_pct": round(100 * transfers_winning / len(transfers_settled), 1) if transfers_settled else None,
        },
        "data_dir": str(_data_dir()),
    }
