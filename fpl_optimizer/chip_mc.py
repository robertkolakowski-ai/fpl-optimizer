"""Monte Carlo chip strategy with proper Dynamic Programming allocation.

Best-practice version (raised from MVP):
  - For each MC path, optimal chip allocation is solved via DP over
    (chips_used_set, gw) — not greedy per-chip pluck-best.
  - DP correctly handles constraints:
      * Each chip max 1 use per season
      * At most 1 chip per GW
      * (FPL 2024+ rule) Two Wildcards: one per half-season; we model this as
        WC1 valid in GW≤19, WC2 valid in GW≥20
  - Includes "do nothing" baseline so chip values are EXTRA pts above baseline.
  - Aggregates over N paths: per (chip, gw), what % of optimal allocations
    placed the chip there + average gain when played.
"""
from __future__ import annotations

import random
from typing import Any

from .models import Fixture, Gameweek, Player
from .uncertainty import estimate_player_uncertainty


CHIP_NAMES_DEFAULT = ("WC1", "WC2", "BB", "TC", "FH")


def _sample_player_pts(uncertainty: dict[str, Any], rng: random.Random) -> float:
    """Sample a single GW realization from a player's bootstrap quantiles."""
    p10 = uncertainty.get("p10", 0)
    p50 = uncertainty.get("p50", 0)
    p90 = uncertainty.get("p90", 0)
    u = rng.random()
    if u < 0.1:
        return p10 * (u / 0.1)
    if u < 0.5:
        return p10 + ((u - 0.1) / 0.4) * (p50 - p10)
    if u < 0.9:
        return p50 + ((u - 0.5) / 0.4) * (p90 - p50)
    return p90 * 1.1


def _per_gw_metrics(
    pool: list[Player],
    uncertainty: dict[int, dict[str, Any]],
    remaining_gws: list[int],
    rng: random.Random,
) -> dict[int, dict[str, float]]:
    """For one MC path: sample per-player per-GW pts, return aggregate metrics per GW."""
    metrics: dict[int, dict[str, float]] = {}
    for gw in remaining_gws:
        sampled = sorted(
            (_sample_player_pts(uncertainty[p.id], rng) for p in pool if p.id in uncertainty),
            reverse=True,
        )
        if not sampled:
            metrics[gw] = {"top_pick": 0, "top_xi": 0, "bench": 0, "baseline": 0}
            continue
        top_pick = sampled[0]
        top_xi = sum(sampled[:11])
        # Bench = positions 12-15 of best-15
        bench_sum = sum(sampled[11:15]) if len(sampled) >= 15 else 0
        # Baseline: an "average squad" — assume ~50 pts per GW
        baseline_xi = 50.0
        metrics[gw] = {
            "top_pick": top_pick,
            "top_xi": top_xi,
            "bench": bench_sum,
            "baseline": baseline_xi,
        }
    return metrics


def _chip_value(chip: str, gw: int, m: dict[str, float]) -> float:
    """Extra points (above baseline) from playing chip at this GW for this MC path."""
    if chip == "TC":
        # Triple captain: extra captain (already double-counted) → 1× more captain pts
        return m["top_pick"]
    if chip == "BB":
        # Bench plays: extra bench pts
        return m["bench"]
    if chip in ("WC1", "WC2"):
        # WC: rebuilds squad → assume top-XI vs your current avg (baseline_xi)
        return max(0, m["top_xi"] - m["baseline"])
    if chip == "FH":
        # FH: same as WC for a single GW (then revert)
        return max(0, m["top_xi"] - m["baseline"])
    return 0


def _chip_allowed_in_gw(chip: str, gw: int) -> bool:
    """FPL 2024+ rules: WC1 in first half (GW≤19), WC2 in second (GW≥20)."""
    if chip == "WC1":
        return gw <= 19
    if chip == "WC2":
        return gw >= 20
    return True


def _solve_dp(
    chips_available: list[str],
    remaining_gws: list[int],
    metrics: dict[int, dict[str, float]],
) -> tuple[float, dict[str, int | None]]:
    """DP over (used_chips_bitmask, gw_index). Returns (total_extra, allocation).

    allocation[chip] = gw or None.
    """
    n_chips = len(chips_available)
    n_gws = len(remaining_gws)
    chip_to_bit = {c: 1 << i for i, c in enumerate(chips_available)}

    # memo[(used_mask, gw_idx)] = (best_extra, best_choice)
    # best_choice: ("play", chip) | ("skip", None) | terminal
    memo: dict[tuple[int, int], tuple[float, tuple[str, str | None]]] = {}

    def solve(used: int, gw_idx: int) -> tuple[float, tuple[str, str | None]]:
        if gw_idx == n_gws:
            return 0.0, ("end", None)
        key = (used, gw_idx)
        if key in memo:
            return memo[key]
        gw = remaining_gws[gw_idx]
        m = metrics[gw]

        # Option 1: skip this GW
        skip_val, _ = solve(used, gw_idx + 1)
        best = (skip_val, ("skip", None))

        # Option 2: play any unused chip allowed in this GW
        for chip in chips_available:
            bit = chip_to_bit[chip]
            if used & bit:
                continue
            if not _chip_allowed_in_gw(chip, gw):
                continue
            value = _chip_value(chip, gw, m)
            future, _ = solve(used | bit, gw_idx + 1)
            total = value + future
            if total > best[0]:
                best = (total, ("play", chip))
        memo[key] = best
        return best

    total_extra, _ = solve(0, 0)

    # Reconstruct allocation
    allocation: dict[str, int | None] = {c: None for c in chips_available}
    used = 0
    for gw_idx, gw in enumerate(remaining_gws):
        _, action = memo[(used, gw_idx)]
        if action[0] == "play":
            chip = action[1]
            allocation[chip] = gw
            used |= chip_to_bit[chip]
    return total_extra, allocation


def simulate_chips(
    players: list[Player],
    remaining_gws: list[int],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    *,
    chips_available: list[str] | None = None,
    n_simulations: int = 500,
    top_pool: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    """Run MC with DP-based optimal chip allocation per path.

    Args:
        chips_available: defaults to ("WC1","WC2","BB","TC","FH"). Pass a
            subset if some chips are already used.
    """
    if not remaining_gws:
        return {"error": "No remaining gameweeks"}
    chips_available = chips_available or list(CHIP_NAMES_DEFAULT)

    uncertainty = estimate_player_uncertainty(players, fixtures, gameweeks, top_pool=top_pool)
    pool = [p for p in players if p.id in uncertainty]
    pool.sort(key=lambda p: p.composite_score, reverse=True)
    pool = pool[:top_pool]

    rng = random.Random(seed)

    # chip_wins[chip][gw] = number of MC paths where DP placed this chip here
    chip_wins: dict[str, dict[int, int]] = {c: {g: 0 for g in remaining_gws} for c in chips_available}
    chip_gains: dict[str, dict[int, list[float]]] = {c: {g: [] for g in remaining_gws} for c in chips_available}
    chip_unused: dict[str, int] = {c: 0 for c in chips_available}
    total_extra_path: list[float] = []

    for _ in range(n_simulations):
        metrics = _per_gw_metrics(pool, uncertainty, remaining_gws, rng)
        total, alloc = _solve_dp(chips_available, remaining_gws, metrics)
        total_extra_path.append(total)
        for chip, gw in alloc.items():
            if gw is None:
                chip_unused[chip] += 1
                continue
            chip_wins[chip][gw] += 1
            chip_gains[chip][gw].append(_chip_value(chip, gw, metrics[gw]))

    result: dict[str, Any] = {
        "n_simulations": n_simulations,
        "remaining_gws": remaining_gws,
        "expected_extra_points": round(sum(total_extra_path) / max(1, n_simulations), 1),
        "chips": {},
    }
    for chip in chips_available:
        per_gw = []
        for gw in remaining_gws:
            wins = chip_wins[chip][gw]
            gains = chip_gains[chip][gw]
            per_gw.append({
                "gw": gw,
                "play_probability": round(100 * wins / n_simulations, 1),
                "avg_gain_when_played": round(sum(gains) / len(gains), 2) if gains else 0,
                "n_paths": wins,
            })
        per_gw.sort(key=lambda x: x["play_probability"], reverse=True)
        result["chips"][chip] = {
            "best_gw": per_gw[0]["gw"] if per_gw and per_gw[0]["play_probability"] > 0 else None,
            "best_gw_probability": per_gw[0]["play_probability"] if per_gw else 0,
            "unused_probability": round(100 * chip_unused[chip] / n_simulations, 1),
            "ranked": per_gw[:5],
        }
    return result
