"""Live in-play BPS-projected bonus points.

While matches are live, the FPL ``/event/<gw>/live/`` endpoint returns each
player's running BPS (Bonus Point System) score. Final bonus = top BPS in
each fixture gets 3, second 2, third 1 (ties resolved per FPL rules).

This module:
  - Pulls live data
  - Groups players by fixture
  - For each fixture, ranks BPS and projects 3/2/1 bonus before they're
    officially awarded by FPL (which usually waits until ~5 min after FT)

This is exactly the kind of mid-match info that gives an edge — captain
swap decisions, autosub anticipation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import httpx

from .api import fetch_live_gameweek, BASE_URL


def project_bonus(
    gameweek: int,
    fixtures: list,
) -> list[dict[str, Any]]:
    """Return projected bonus per fixture for an in-play GW.

    Returns:
        [{
            "fixture_id": int,
            "home_team": int, "away_team": int,
            "started": bool, "finished": bool,
            "minutes": int,
            "bonus_projection": [
                {"player_id": int, "bps": int, "projected_bonus": 3|2|1, ...}
            ]
        }, ...]
    """
    with httpx.Client(timeout=20) as client:
        live = fetch_live_gameweek(client, gameweek)

    # Build player_id → BPS + minutes from live response
    elements = (live or {}).get("elements", [])
    player_bps: dict[int, dict[str, Any]] = {}
    for el in elements:
        stats = el.get("stats") or {}
        explain = el.get("explain") or []
        # explain is a list of per-fixture stat lists
        for ex in explain:
            fid = ex.get("fixture")
            if not fid:
                continue
            fstats = ex.get("stats") or []
            bps_entry = next((s for s in fstats if s.get("identifier") == "bps"), None)
            mins_entry = next((s for s in fstats if s.get("identifier") == "minutes"), None)
            bps = (bps_entry or {}).get("value", 0) or 0
            mins = (mins_entry or {}).get("value", 0) or 0
            player_bps[(el["id"], fid)] = {
                "player_id": el["id"],
                "fixture_id": fid,
                "bps": bps,
                "minutes": mins,
                "total_points": stats.get("total_points", 0),
            }

    # Group by fixture
    by_fixture: dict[int, list[dict]] = defaultdict(list)
    for entry in player_bps.values():
        by_fixture[entry["fixture_id"]].append(entry)

    fixture_map = {f.id: f for f in fixtures}
    results = []
    for fid, players in by_fixture.items():
        fx = fixture_map.get(fid)
        if not fx:
            continue
        # Sort by BPS desc, only players with minutes > 0
        playing = sorted(
            [p for p in players if p["minutes"] > 0],
            key=lambda p: p["bps"],
            reverse=True,
        )
        # FPL bonus logic: 3-2-1 with ties tied (e.g. two players tied for top get 3 each, next 1)
        proj = _project_bonus_with_ties(playing)
        for entry, bonus in zip(playing, proj):
            entry["projected_bonus"] = bonus

        results.append({
            "fixture_id": fid,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "started": fx.started,
            "finished": fx.finished,
            "home_score": fx.home_score,
            "away_score": fx.away_score,
            "bonus_projection": playing[:6],  # top-6 BPS
        })
    return results


def _project_bonus_with_ties(sorted_players: list[dict]) -> list[int]:
    """FPL bonus distribution with ties.

    Rules: top BPS gets 3, second 2, third 1. Ties:
      - n players tied for 1st → all get 3
      - 1 unique 1st, n tied for 2nd → all tied get 2
      - 1 unique 1st, 1 unique 2nd, n tied for 3rd → all tied get 1
    """
    if not sorted_players:
        return []
    bonuses = [0] * len(sorted_players)
    bps_values = [p["bps"] for p in sorted_players]
    # Distinct BPS values in descending order
    distinct = []
    for v in bps_values:
        if not distinct or distinct[-1] != v:
            distinct.append(v)
    bonus_levels = [3, 2, 1]
    for i, lvl in enumerate(bonus_levels):
        if i >= len(distinct):
            break
        v = distinct[i]
        for j, p_bps in enumerate(bps_values):
            if p_bps == v:
                bonuses[j] = lvl
    return bonuses
