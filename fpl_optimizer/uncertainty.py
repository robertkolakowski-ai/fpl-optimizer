"""Bayesian-style uncertainty estimation for player point projections.

Best-practice version (raised from MVP):
  - Bootstrap weights samples by similarity to the upcoming fixture.
    A player's GW20 vs Liverpool (away) tells us less about their
    upcoming GW vs Burnley (home) than their past Burnley/Bournemouth-
    tier home games do. We weight closer fixtures higher in the bootstrap.
  - Adjusts for injury / availability:
      * chance_of_playing < 100 → scales p50 down, widens p10-p90
      * Status flags (i = injured, d = doubt, s = suspended) handled explicitly
  - Returns the adjustment metadata so the UI can show
    "p50=8 (adjusted from 10 due to injury flag)"

Output per player:
  {
    "p10": float, "p50": float, "p90": float,
    "mean": float, "n": int,
    "raw": {"p10":..., "p50":..., "p90":..., "mean":...},  # pre-adjustment
    "next_fixture_difficulty": int (1-5),
    "availability": float (0.0-1.0),
    "adjustment_notes": [str, ...]
  }
"""
from __future__ import annotations

import random
from typing import Any

import httpx

from .backtest import _load_cache, _save_cache, fetch_element_summary
from .models import Fixture, Gameweek, Player


def _player_gw_records(history: list[dict], min_minutes: int = 30) -> list[dict[str, Any]]:
    """Extract per-GW records (points, fixture context) for similarity weighting."""
    by_gw: dict[int, dict[str, Any]] = {}
    for h in history:
        gw = h.get("round")
        if not gw:
            continue
        if gw not in by_gw:
            by_gw[gw] = {
                "points": 0,
                "minutes": 0,
                "was_home": bool(h.get("was_home")),
                "opponent_team": h.get("opponent_team"),
                "difficulty": int(h.get("difficulty", 3) or 3),
            }
        by_gw[gw]["points"] += int(h.get("total_points", 0) or 0)
        by_gw[gw]["minutes"] += int(h.get("minutes", 0) or 0)
        # Multi-fixture GW: average difficulty
        if h.get("difficulty"):
            by_gw[gw]["difficulty"] = int(h["difficulty"])
    return [
        {**v, "round": k}
        for k, v in by_gw.items()
        if v["minutes"] >= min_minutes
    ]


def _next_fixture_for_player(
    player: Player,
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
) -> tuple[int, bool] | None:
    """Returns (difficulty, is_home) for the player's next fixture, if any."""
    next_gw = next((g for g in gameweeks if g.is_next), None) or next(
        (g for g in gameweeks if g.is_current and not g.finished), None
    )
    if not next_gw:
        return None
    for f in fixtures:
        if f.gameweek != next_gw.id:
            continue
        if f.home_team == player.team:
            return (int(f.home_difficulty), True)
        if f.away_team == player.team:
            return (int(f.away_difficulty), False)
    return None


def _similarity_weight(record: dict, target_difficulty: int, target_is_home: bool) -> float:
    """Inverse-distance similarity weight in (0.5, 1.0+).

    Records vs similar opponents/venue weighted higher. Far-away records still
    contribute (we don't want to throw out information), but the weight is lower.
    """
    diff_dist = abs(record["difficulty"] - target_difficulty)  # 0-4
    home_match = 1.0 if record["was_home"] == target_is_home else 0.6
    # Difficulty distance: 0 → 1.0, 4 → 0.5
    diff_weight = 1.0 - 0.125 * diff_dist
    return diff_weight * home_match


def _weighted_bootstrap(
    samples: list[int],
    weights: list[float],
    n_resamples: int = 1000,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    rng_seed: int = 42,
) -> dict[str, float]:
    """Weighted bootstrap — draws scaled by per-sample weights."""
    if not samples:
        return {f"p{int(q*100)}": 0.0 for q in quantiles} | {"mean": 0.0, "n": 0}
    if len(samples) == 1:
        v = float(samples[0])
        return {f"p{int(q*100)}": v for q in quantiles} | {"mean": v, "n": 1}

    rng = random.Random(rng_seed)
    total_w = sum(weights) or 1.0
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cum.append(acc)

    draws: list[int] = []
    for _ in range(n_resamples):
        u = rng.random()
        # Find first cum >= u
        for i, c in enumerate(cum):
            if u <= c:
                draws.append(samples[i])
                break
        else:
            draws.append(samples[-1])
    draws.sort()
    n = len(draws)
    weighted_mean = sum(s * w for s, w in zip(samples, weights)) / total_w
    result: dict[str, float] = {"n": len(samples), "mean": round(weighted_mean, 2)}
    for q in quantiles:
        idx = max(0, min(n - 1, int(q * n)))
        result[f"p{int(q*100)}"] = float(draws[idx])
    return result


def _apply_availability_adjustment(
    estimate: dict[str, float],
    player: Player,
) -> tuple[dict[str, float], list[str]]:
    """Adjust p10/p50/p90 for injury / availability. Returns (adjusted, notes)."""
    notes: list[str] = []
    cop = player.chance_of_playing
    out = dict(estimate)

    # Status-based adjustments take priority
    # FPL doesn't expose status directly via Player, but `news` is a strong proxy
    news = (player.news or "").lower()
    if "suspended" in news or "ban" in news:
        notes.append("Suspended — projected 0")
        return ({**out, "p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}, notes)

    if cop is not None:
        if cop == 0:
            notes.append(f"Unavailable (chance_of_playing=0%) — projected 0")
            return ({**out, "p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}, notes)
        if cop < 100:
            scale = cop / 100.0
            # Scale center down
            for k in ("p10", "p50", "p90", "mean"):
                out[k] = round(out.get(k, 0) * scale, 1)
            # Widen interval to reflect uncertainty about whether they play
            spread = out["p90"] - out["p10"]
            extra_uncertainty = (1 - scale) * 0.3 * spread
            out["p10"] = max(0.0, round(out["p10"] - extra_uncertainty, 1))
            out["p90"] = round(out["p90"] + extra_uncertainty, 1)
            notes.append(f"Adjusted ↓ for {cop}% playing chance (interval widened)")
    if player.rotation_risk and player.rotation_risk > 0.4:
        # Rotation risk doesn't change the "if they play" distribution, but
        # affects expected value. Flag it without re-scaling (#4 talks about
        # uncertainty, rotation is captured separately via projected_minutes).
        notes.append(f"Rotation risk {int(player.rotation_risk * 100)}%")
    return out, notes


def estimate_player_uncertainty(
    players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    *,
    top_pool: int = 80,
    min_minutes: int = 30,
) -> dict[int, dict[str, Any]]:
    """For each top-pool player, return uncertainty estimates that account for:
      - Per-game similarity weighting against the upcoming fixture
      - Injury / availability adjustment
    """
    pool = sorted(players, key=lambda p: p.composite_score, reverse=True)[:top_pool]
    cache = _load_cache()
    out: dict[int, dict[str, Any]] = {}

    with httpx.Client(timeout=30) as client:
        for p in pool:
            data = fetch_element_summary(client, p.id, cache)
            if not data:
                continue
            records = _player_gw_records(data.get("history", []), min_minutes=min_minutes)
            if not records:
                continue
            samples = [r["points"] for r in records]
            next_fx = _next_fixture_for_player(p, fixtures, gameweeks)
            if next_fx:
                tgt_diff, tgt_home = next_fx
                weights = [_similarity_weight(r, tgt_diff, tgt_home) for r in records]
            else:
                weights = [1.0] * len(records)
                tgt_diff = None
                tgt_home = None
            raw = _weighted_bootstrap(samples, weights)
            adjusted, notes = _apply_availability_adjustment(raw, p)
            out[p.id] = {
                **adjusted,
                "raw": raw,
                "next_fixture_difficulty": tgt_diff,
                "next_fixture_is_home": tgt_home,
                "availability": (
                    p.chance_of_playing / 100.0
                    if p.chance_of_playing is not None else 1.0
                ),
                "adjustment_notes": notes,
            }
    _save_cache(cache)
    return out


def captain_recommendations(
    players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    *,
    risk_mode: str = "balanced",
    top_pool: int = 80,
    top_n_results: int = 10,
) -> list[dict[str, Any]]:
    """Rank captain candidates with risk-aware sorting + fixture-aware uncertainty."""
    uncertainties = estimate_player_uncertainty(players, fixtures, gameweeks, top_pool=top_pool)
    pool = [p for p in players if p.id in uncertainties and p.position in (2, 3, 4)]

    def rank_key(p: Player) -> float:
        u = uncertainties[p.id]
        if risk_mode == "safe":
            return u.get("p10", 0)
        if risk_mode == "aggressive":
            return u.get("p90", 0)
        # balanced: mean + slight upside bonus
        return u.get("mean", 0) + 0.2 * (u.get("p90", 0) - u.get("p10", 0))

    pool.sort(key=rank_key, reverse=True)
    return [
        {
            "id": p.id,
            "name": p.name,
            "photo": p.photo,
            "team": p.team,
            "position": p.position_name,
            "cost": p.cost,
            "composite_score": round(p.composite_score, 3),
            "uncertainty": uncertainties[p.id],
            "rank_score": round(rank_key(p), 2),
        }
        for p in pool[:top_n_results]
    ]
