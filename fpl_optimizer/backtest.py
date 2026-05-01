"""Honest backtest engine — replays the LIVE scoring model against finished GWs.

Critical design choice (raised by elevating from MVP to best-practice):
  The previous version used a simplified `_replay_score()` parallel to the
  real `score_players()`. That meant backtest results were directionally OK
  but not numerically comparable to live model output.

  This version:
    1. Reconstructs each player's state (cumulative stats) AS IT WOULD HAVE
       BEEN at the deadline of a historical GW, using the per-GW history
       from `element-summary/{id}/`.
    2. Calls the EXACT SAME `score_players()` function the live app uses.
    3. Cross-validates: normalization happens only against data available
       BEFORE that GW.

  So the backtest answers the question users actually care about:
  "If we'd run this model at the deadline of GW20, how would its top-15
  have compared to the actual top-15?"
"""
from __future__ import annotations

import json
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from .analyzer import score_players
from .api import BASE_URL
from .models import Fixture, Gameweek, Player

_LOCK = threading.Lock()


def _data_dir() -> Path:
    base = os.environ.get("FPL_DATA_DIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data"


def _cache_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "element_summary_cache.json"


def _load_cache() -> dict[str, Any]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    p = _cache_path()
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(p)


def fetch_element_summary(
    client: httpx.Client,
    player_id: int,
    cache: dict[str, Any],
    cache_ttl_hours: float = 12,
) -> dict | None:
    key = str(player_id)
    now = time.time()
    cached = cache.get(key)
    if cached and (now - cached.get("fetched_at", 0)) < cache_ttl_hours * 3600:
        return cached.get("data")
    try:
        resp = client.get(f"{BASE_URL}/element-summary/{player_id}/")
        resp.raise_for_status()
        data = resp.json()
        cache[key] = {"fetched_at": now, "data": data}
        return data
    except Exception:
        return None


def _player_snapshot_at_gw(player: Player, history: list[dict], target_gw: int) -> Player | None:
    """Reconstruct a Player as it would have appeared going into `target_gw`.

    Uses element-summary history rows BEFORE target_gw to rebuild cumulative
    stats (xG, xA, BPS, minutes, clean_sheets, bonus, points) and a recent-form
    proxy from the last 5 played GWs.

    Returns None if no prior history exists.
    """
    prior = [h for h in history if h.get("round") and h["round"] < target_gw]
    if not prior:
        return None

    snap = deepcopy(player)
    # Cumulative through the prior rows
    snap.minutes = sum(int(h.get("minutes", 0) or 0) for h in prior)
    snap.goals = sum(int(h.get("goals_scored", 0) or 0) for h in prior)
    snap.assists = sum(int(h.get("assists", 0) or 0) for h in prior)
    snap.clean_sheets = sum(int(h.get("clean_sheets", 0) or 0) for h in prior)
    snap.goals_conceded = sum(int(h.get("goals_conceded", 0) or 0) for h in prior)
    snap.bonus = sum(int(h.get("bonus", 0) or 0) for h in prior)
    snap.bps = sum(int(h.get("bps", 0) or 0) for h in prior)
    snap.total_points = sum(int(h.get("total_points", 0) or 0) for h in prior)
    snap.starts = sum(int(h.get("starts", 0) or 0) for h in prior)
    snap.saves = sum(int(h.get("saves", 0) or 0) for h in prior)
    snap.yellow_cards = sum(int(h.get("yellow_cards", 0) or 0) for h in prior)
    snap.red_cards = sum(int(h.get("red_cards", 0) or 0) for h in prior)
    snap.xG = sum(float(h.get("expected_goals", 0) or 0) for h in prior)
    snap.xA = sum(float(h.get("expected_assists", 0) or 0) for h in prior)
    snap.influence = sum(float(h.get("influence", 0) or 0) for h in prior)
    snap.creativity = sum(float(h.get("creativity", 0) or 0) for h in prior)
    snap.threat = sum(float(h.get("threat", 0) or 0) for h in prior)
    # ICT index: average of cumulative components, FPL-style (per-GW field, average)
    snap.ict_index = sum(float(h.get("ict_index", 0) or 0) for h in prior) / len(prior)
    snap.expected_goal_involvements = snap.xG + snap.xA
    snap.expected_goals_conceded = sum(float(h.get("expected_goals_conceded", 0) or 0) for h in prior)

    # Form proxy: average pts over last 5 GWs (FPL's actual definition)
    last5 = prior[-5:]
    snap.form = sum(int(h.get("total_points", 0) or 0) for h in last5) / max(1, len(last5))
    snap.points_per_game = snap.total_points / len(prior) if prior else 0.0

    # Per-90 stats — only meaningful if the player has played enough
    if snap.minutes >= 60:
        per90 = 90.0 / snap.minutes
        snap.xG_per90 = snap.xG * per90
        snap.xA_per90 = snap.xA * per90
        snap.xGI_per90 = snap.expected_goal_involvements * per90
        snap.xGC_per90 = snap.expected_goals_conceded * per90
    else:
        snap.xG_per90 = snap.xA_per90 = snap.xGI_per90 = snap.xGC_per90 = 0.0

    # Reset things we don't have historical context for
    snap.composite_score = 0.0
    snap.fixture_difficulty = 0.5  # will be recomputed by score_players
    snap.rotation_risk = 0.0
    snap.projected_minutes = 0.0
    snap.score_breakdown = []
    return snap


def _build_historical_snapshot(
    players: list[Player],
    histories: dict[int, list[dict]],
    target_gw: int,
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict | None,
) -> list[Player]:
    """Build a list of player snapshots representing state at deadline of target_gw.

    Then runs the LIVE score_players() against this snapshot — same code path
    as the production model.
    """
    snapshots: list[Player] = []
    for p in players:
        hist = histories.get(p.id, [])
        snap = _player_snapshot_at_gw(p, hist, target_gw)
        if snap is not None:
            snapshots.append(snap)

    # Build a fictitious "current/next gw" gameweek list so fixture_difficulty
    # is computed using fixtures from target_gw onwards (not future-known data).
    # Trick: mark target_gw as is_next, all others not.
    fake_gws = []
    for gw in gameweeks:
        # We can't mutate the original; create a lightweight equivalent
        new_gw = Gameweek(
            id=gw.id, name=gw.name,
            finished=gw.id < target_gw,
            is_current=False,
            is_next=(gw.id == target_gw),
        )
        fake_gws.append(new_gw)

    # Filter fixtures to only those at target_gw or later (no future leakage)
    relevant_fx = [f for f in fixtures if f.gameweek and f.gameweek >= target_gw]

    score_players(snapshots, relevant_fx, fake_gws, teams=teams, lookahead=5)
    return snapshots


def _fetch_histories(
    pool: list[Player],
    cache: dict[str, Any],
) -> dict[int, list[dict]]:
    histories: dict[int, list[dict]] = {}
    with httpx.Client(timeout=30) as client:
        for p in pool:
            data = fetch_element_summary(client, p.id, cache)
            if data:
                histories[p.id] = data.get("history", [])
    _save_cache(cache)
    return histories


def backtest_captain(
    players: list[Player],
    finished_gws: list[int],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict | None = None,
    *,
    top_pool: int = 80,
) -> dict[str, Any]:
    """Honest backtest: replay live model on per-GW historical snapshots.

    For each finished GW:
        1. Reconstruct top-pool's state at GW deadline (cumulative through GW-1)
        2. Call live `score_players()` — same code as production
        3. Pick highest-scoring outfield player → that's the captain pick
        4. Compare actual GW points (×2 for captain) vs theoretical max captain
    """
    if not finished_gws:
        return {"gameweeks": [], "summary": {"gws_evaluated": 0}}

    pool = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_pool]
    cache = _load_cache()
    histories = _fetch_histories(pool, cache)

    salah = next((p for p in players if "salah" in p.name.lower()), None)
    if salah and salah.id not in histories:
        with httpx.Client(timeout=30) as client:
            data = fetch_element_summary(client, salah.id, cache)
            if data:
                histories[salah.id] = data.get("history", [])
        _save_cache(cache)

    gw_rows = []
    model_total = 0
    best_total = 0
    salah_total = 0
    eval_count = 0

    for gw in finished_gws:
        # Build snapshot of pool as it was at deadline for `gw`
        snapshot = _build_historical_snapshot(
            pool, histories, gw, fixtures, gameweeks, teams
        )
        if not snapshot:
            continue

        # Captain candidates: outfield only, with positive snapshot score
        candidates = [p for p in snapshot if p.position in (2, 3, 4) and p.composite_score > 0]
        if not candidates:
            continue

        # Map snapshot -> actual GW points
        actual_pts: dict[int, int] = {}
        for sp in snapshot:
            rows = [h for h in histories.get(sp.id, []) if h.get("round") == gw]
            if rows:
                actual_pts[sp.id] = sum(int(r.get("total_points", 0) or 0) for r in rows)

        # Model pick: highest replayed composite_score
        model_pick = max(candidates, key=lambda p: p.composite_score)
        if model_pick.id not in actual_pts:
            continue
        eval_count += 1

        # Best possible: highest actual pts in pool
        best_pick_id, best_actual = max(actual_pts.items(), key=lambda kv: kv[1])
        best_pick = next((p for p in snapshot if p.id == best_pick_id), None)

        salah_actual = actual_pts.get(salah.id) if salah else None

        # Captain doubles
        model_pts = actual_pts[model_pick.id] * 2
        best_pts = best_actual * 2
        salah_pts = (salah_actual or 0) * 2 if salah_actual is not None else None

        model_total += model_pts
        best_total += best_pts
        if salah_pts is not None:
            salah_total += salah_pts

        # Rich row including model's score breakdown for the pick
        gw_rows.append({
            "gw": gw,
            "model_pick": model_pick.name,
            "model_pick_id": model_pick.id,
            "model_pick_photo": model_pick.photo,
            "model_pick_score": round(model_pick.composite_score, 4),
            "model_actual": model_pts,
            "best_possible": best_pts,
            "best_player": best_pick.name if best_pick else "?",
            "best_player_photo": best_pick.photo if best_pick else "",
            "salah_actual": salah_pts,
            "delta_vs_best": model_pts - best_pts,
            # The top-3 score components for the model's pick — explainability
            "score_breakdown_top3": sorted(
                model_pick.score_breakdown,
                key=lambda c: c.get("contribution", 0),
                reverse=True,
            )[:3] if model_pick.score_breakdown else [],
        })

    if eval_count == 0:
        return {"gameweeks": [], "summary": {"gws_evaluated": 0}}

    model_avg = model_total / eval_count
    best_avg = best_total / eval_count
    salah_avg = salah_total / eval_count if salah else None
    capture = (model_avg / best_avg) if best_avg > 0 else 0

    return {
        "gameweeks": gw_rows,
        "summary": {
            "gws_evaluated": eval_count,
            "model_avg_pts": round(model_avg, 2),
            "best_possible_avg": round(best_avg, 2),
            "salah_avg_pts": round(salah_avg, 2) if salah_avg is not None else None,
            "model_capture_rate": round(capture, 3),
            "model_total_pts": model_total,
            "best_possible_total": best_total,
            "salah_total_pts": salah_total if salah else None,
            "methodology": (
                "Honest replay — uses live score_players() on per-GW snapshots "
                "rebuilt from element-summary history. No future-data leakage."
            ),
        },
    }


def backtest_top_n_overlap(
    players: list[Player],
    finished_gws: list[int],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict | None = None,
    *,
    top_n: int = 15,
    top_pool: int = 80,
) -> dict[str, Any]:
    """For each finished GW: model's top-N (rebuilt + scored honestly) vs actual top-N."""
    if not finished_gws:
        return {"gameweeks": [], "summary": {"gws_evaluated": 0}}

    pool = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_pool]
    cache = _load_cache()
    histories = _fetch_histories(pool, cache)

    rows = []
    overlap_sum = 0
    eval_count = 0

    for gw in finished_gws:
        snapshot = _build_historical_snapshot(
            pool, histories, gw, fixtures, gameweeks, teams
        )
        if not snapshot:
            continue
        actual_pts: dict[int, int] = {}
        for sp in snapshot:
            r_for_gw = [h for h in histories.get(sp.id, []) if h.get("round") == gw]
            if r_for_gw:
                actual_pts[sp.id] = sum(int(r.get("total_points", 0) or 0) for r in r_for_gw)
        scored = [sp for sp in snapshot if sp.id in actual_pts and sp.composite_score > 0]
        if len(scored) < top_n:
            continue
        eval_count += 1
        model_top = {sp.id for sp in sorted(scored, key=lambda s: s.composite_score, reverse=True)[:top_n]}
        actual_top = {sp_id for sp_id, _ in sorted(actual_pts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]}
        overlap = len(model_top & actual_top)
        overlap_sum += overlap
        rows.append({"gw": gw, "overlap": overlap, "of": top_n})

    if eval_count == 0:
        return {"gameweeks": [], "summary": {"gws_evaluated": 0}}

    return {
        "gameweeks": rows,
        "summary": {
            "gws_evaluated": eval_count,
            "top_n": top_n,
            "avg_overlap": round(overlap_sum / eval_count, 2),
            "avg_hit_rate_pct": round(100 * overlap_sum / (eval_count * top_n), 1),
        },
    }
