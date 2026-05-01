"""Mini-league intelligence with proper rank-EV simulation.

Best-practice version (raised from MVP):
  - Pulls each league member's full picks for the GW
  - Simulates remaining-season trajectories per member using uncertainty (#4)
  - For YOUR team: evaluates candidate transfers by simulating both:
      (a) your current squad's rank distribution at season end
      (b) post-transfer squad's rank distribution
    and reporting the change.
  - Output is actionable: "Salah → Saka: +3.2 EP per GW, moves you from
    median rank 4 to median rank 2 in 41% of paths".

Pragmatic shortcuts (called out so we know what we'd improve next):
  - Other members are assumed to keep current squad (no opponent transfers).
    A real version would also model their probable transfers.
  - We simulate EP per remaining GW from each player's bootstrap p10/p50/p90,
    not actual fixtures-vs-actual-fixtures. Good enough for relative ranking.
"""
from __future__ import annotations

import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

from .api import (
    fetch_league_standings,
    fetch_user_picks_full,
)
from .models import Fixture, Gameweek, Player
from .uncertainty import estimate_player_uncertainty


# Inline sample helper (mirrors chip_mc) — keeps modules independent
def _sample_pts(uncertainty: dict[str, Any], rng: random.Random) -> float:
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


def _fetch_member_picks(client: httpx.Client, user_id: int, gw: int) -> dict | None:
    try:
        return fetch_user_picks_full(client, user_id, gw)
    except Exception:
        return None


def _simulate_remaining_season(
    squad_player_ids: list[int],
    starting_xi_ids: set[int],
    captain_id: int | None,
    uncertainty: dict[int, dict[str, Any]],
    n_paths: int,
    n_gws_remaining: int,
    rng: random.Random,
    starting_total: int = 0,
) -> list[float]:
    """Simulate end-of-season total points distribution for a squad.

    Returns list of N path-final totals.
    """
    finals = []
    for _ in range(n_paths):
        total = starting_total
        for _gw in range(n_gws_remaining):
            for pid in squad_player_ids:
                if pid not in uncertainty:
                    continue
                pts = _sample_pts(uncertainty[pid], rng)
                if pid in starting_xi_ids:
                    if captain_id and pid == captain_id:
                        total += 2 * pts
                    else:
                        total += pts
                # bench pts ignored (only auto-subs would change this)
        finals.append(total)
    return finals


def league_intelligence(
    league_id: int,
    gw: int,
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    all_players: list[Player],
    your_team_id: int | None = None,
    *,
    max_members: int = 50,
    simulate_rank_ev: bool = True,
    n_paths: int = 200,
) -> dict[str, Any]:
    """Aggregate league picks + simulate rank distribution if `your_team_id` provided."""
    with httpx.Client(timeout=30) as client:
        standings = fetch_league_standings(client, league_id)
        members = (standings.get("standings") or {}).get("results", [])[:max_members]
        member_ids = [m["entry"] for m in members]
        member_starting_pts = {m["entry"]: int(m.get("total", 0)) for m in members}

        picks_by_member: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_member_picks, client, mid, gw): mid
                for mid in member_ids
            }
            for fut in as_completed(futures):
                mid = futures[fut]
                data = fut.result()
                if data:
                    picks_by_member[mid] = data

    if not picks_by_member:
        return {"error": "Could not fetch any member picks"}

    # Aggregate ownership
    ownership: Counter[int] = Counter()
    captains: Counter[int] = Counter()
    for mid, data in picks_by_member.items():
        for pick in data.get("picks", []):
            ownership[pick["element"]] += 1
            if pick.get("is_captain"):
                captains[pick["element"]] += 1

    n_members = len(picks_by_member)

    your_picks_data = picks_by_member.get(your_team_id) if your_team_id else None
    your_player_ids: list[int] = []
    your_starting_xi_ids: set[int] = set()
    your_captain_id: int | None = None
    if your_picks_data:
        for pick in your_picks_data.get("picks", []):
            your_player_ids.append(pick["element"])
            if pick.get("position", 16) <= 11:
                your_starting_xi_ids.add(pick["element"])
            if pick.get("is_captain"):
                your_captain_id = pick["element"]

    # Differentials
    your_differentials = []
    their_differentials = []
    if your_player_ids:
        for pid in set(your_player_ids):
            own_pct = 100 * ownership[pid] / n_members
            if own_pct < 25:
                your_differentials.append({"player_id": pid, "league_own_pct": round(own_pct, 1)})
        for pid, count in ownership.most_common():
            if pid in your_player_ids:
                continue
            own_pct = 100 * count / n_members
            if own_pct >= 30:
                their_differentials.append({"player_id": pid, "league_own_pct": round(own_pct, 1)})
            if len(their_differentials) >= 10:
                break

    captain_breakdown = [
        {"player_id": pid, "captain_pct": round(100 * count / n_members, 1)}
        for pid, count in captains.most_common(10)
    ]

    out: dict[str, Any] = {
        "league_id": league_id,
        "gameweek": gw,
        "members_analyzed": n_members,
        "ownership": [
            {"player_id": pid, "count": count, "pct": round(100 * count / n_members, 1)}
            for pid, count in ownership.most_common(40)
        ],
        "captain_breakdown": captain_breakdown,
        "your_differentials": sorted(your_differentials, key=lambda d: d["league_own_pct"]),
        "their_differentials": their_differentials,
        "your_captain_id": your_captain_id,
    }

    # Rank-EV simulation
    if simulate_rank_ev and your_team_id and your_player_ids:
        n_remaining = len([g for g in gameweeks if not g.finished and g.id >= gw])
        if n_remaining > 0:
            uncertainty = estimate_player_uncertainty(
                all_players, fixtures, gameweeks, top_pool=120
            )
            rng = random.Random(7)

            # Simulate finals for each member
            member_finals: dict[int, list[float]] = {}
            for mid, data in picks_by_member.items():
                squad_ids = []
                starting = set()
                cap_id = None
                for pick in data.get("picks", []):
                    squad_ids.append(pick["element"])
                    if pick.get("position", 16) <= 11:
                        starting.add(pick["element"])
                    if pick.get("is_captain"):
                        cap_id = pick["element"]
                start_pts = member_starting_pts.get(mid, 0)
                member_finals[mid] = _simulate_remaining_season(
                    squad_ids, starting, cap_id, uncertainty,
                    n_paths=n_paths, n_gws_remaining=n_remaining, rng=rng,
                    starting_total=start_pts,
                )

            # For each path, compute your rank vs members
            your_finals = member_finals.get(your_team_id, [])
            your_ranks = []
            for i in range(min(n_paths, len(your_finals))):
                yours = your_finals[i]
                rank = 1 + sum(1 for mid, fl in member_finals.items()
                               if mid != your_team_id and fl[i] > yours)
                your_ranks.append(rank)
            if your_ranks:
                your_ranks.sort()
                median_rank = your_ranks[len(your_ranks) // 2]
                p10_rank = your_ranks[max(0, int(0.1 * len(your_ranks)))]
                p90_rank = your_ranks[min(len(your_ranks) - 1, int(0.9 * len(your_ranks)))]
                # Probability of winning league
                wins = sum(1 for r in your_ranks if r == 1)
                top3 = sum(1 for r in your_ranks if r <= 3)
                out["rank_simulation"] = {
                    "n_paths": len(your_ranks),
                    "median_rank": median_rank,
                    "p10_rank": p10_rank,
                    "p90_rank": p90_rank,
                    "win_probability": round(100 * wins / len(your_ranks), 1),
                    "top3_probability": round(100 * top3 / len(your_ranks), 1),
                    "n_remaining_gws": n_remaining,
                }
    return out


def evaluate_transfer_rank_ev(
    league_id: int,
    your_team_id: int,
    out_player_id: int,
    in_player_id: int,
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    all_players: list[Player],
    *,
    n_paths: int = 300,
    max_members: int = 30,
) -> dict[str, Any]:
    """Compare YOUR rank distribution with vs without a candidate transfer.

    Returns expected rank gain + probability that transfer improves rank.
    """
    next_gw = next((g for g in gameweeks if g.is_next), None) or next(
        (g for g in gameweeks if g.is_current and not g.finished), None
    )
    if not next_gw:
        return {"error": "No active GW"}
    gw = next_gw.id

    with httpx.Client(timeout=30) as client:
        standings = fetch_league_standings(client, league_id)
        members = (standings.get("standings") or {}).get("results", [])[:max_members]
        member_ids = [m["entry"] for m in members]
        member_starting_pts = {m["entry"]: int(m.get("total", 0)) for m in members}

        picks_by_member: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_member_picks, client, mid, gw): mid for mid in member_ids}
            for fut in as_completed(futures):
                mid = futures[fut]
                data = fut.result()
                if data:
                    picks_by_member[mid] = data

    if your_team_id not in picks_by_member:
        return {"error": "Could not fetch your team"}

    your_data = picks_by_member[your_team_id]
    your_squad = [p["element"] for p in your_data.get("picks", [])]
    your_starting = {p["element"] for p in your_data.get("picks", []) if p.get("position", 16) <= 11}
    your_captain = next((p["element"] for p in your_data.get("picks", []) if p.get("is_captain")), None)

    # Build post-transfer squad
    post_squad = [in_player_id if pid == out_player_id else pid for pid in your_squad]
    post_starting = set(post_squad[:11]) if out_player_id in your_starting else your_starting
    if out_player_id in your_starting:
        post_starting = (your_starting - {out_player_id}) | {in_player_id}
    post_captain = your_captain if your_captain != out_player_id else in_player_id

    n_remaining = len([g for g in gameweeks if not g.finished and g.id >= gw])
    if n_remaining == 0:
        return {"error": "No remaining GWs"}

    uncertainty = estimate_player_uncertainty(all_players, fixtures, gameweeks, top_pool=150)
    rng = random.Random(11)

    # Simulate finals for everyone
    finals_with: dict[int, list[float]] = {}
    finals_without: dict[int, list[float]] = {}
    for mid, data in picks_by_member.items():
        squad_ids = [p["element"] for p in data.get("picks", [])]
        starting = {p["element"] for p in data.get("picks", []) if p.get("position", 16) <= 11}
        cap_id = next((p["element"] for p in data.get("picks", []) if p.get("is_captain")), None)
        start_pts = member_starting_pts.get(mid, 0)
        seed_rng = random.Random(rng.randrange(1 << 30))
        if mid == your_team_id:
            finals_without[mid] = _simulate_remaining_season(
                squad_ids, starting, cap_id, uncertainty,
                n_paths=n_paths, n_gws_remaining=n_remaining,
                rng=random.Random(11),  # Fixed seed for both — paired comparison
                starting_total=start_pts,
            )
            finals_with[mid] = _simulate_remaining_season(
                post_squad, post_starting, post_captain, uncertainty,
                n_paths=n_paths, n_gws_remaining=n_remaining,
                rng=random.Random(11),
                starting_total=start_pts,
            )
        else:
            shared = _simulate_remaining_season(
                squad_ids, starting, cap_id, uncertainty,
                n_paths=n_paths, n_gws_remaining=n_remaining,
                rng=seed_rng, starting_total=start_pts,
            )
            finals_with[mid] = shared
            finals_without[mid] = shared

    def compute_ranks(finals_per_member: dict[int, list[float]]) -> list[int]:
        ranks = []
        ys = finals_per_member[your_team_id]
        for i in range(n_paths):
            r = 1 + sum(1 for mid, fl in finals_per_member.items() if mid != your_team_id and fl[i] > ys[i])
            ranks.append(r)
        return ranks

    ranks_without = compute_ranks(finals_without)
    ranks_with = compute_ranks(finals_with)

    delta = [w - wo for w, wo in zip(ranks_with, ranks_without)]
    improved = sum(1 for d in delta if d < 0)  # lower rank = better
    worsened = sum(1 for d in delta if d > 0)

    avg_pts_with = sum(finals_with[your_team_id]) / n_paths
    avg_pts_without = sum(finals_without[your_team_id]) / n_paths

    return {
        "n_paths": n_paths,
        "n_remaining_gws": n_remaining,
        "ep_delta": round(avg_pts_with - avg_pts_without, 2),
        "median_rank_without": sorted(ranks_without)[n_paths // 2],
        "median_rank_with": sorted(ranks_with)[n_paths // 2],
        "rank_improves_pct": round(100 * improved / n_paths, 1),
        "rank_worsens_pct": round(100 * worsened / n_paths, 1),
        "avg_rank_change": round(sum(delta) / n_paths, 2),  # negative = better
    }
