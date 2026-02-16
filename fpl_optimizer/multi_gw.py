"""Multi-gameweek transfer planner and chip optimizer.

Uses a greedy rolling-horizon approach:
For each GW in the planning window, re-score players for that specific GW,
then find the best single transfer considering free transfers and hit costs.
"""

from __future__ import annotations

from copy import deepcopy

from .analyzer import score_players
from .models import Fixture, Gameweek, Player


POINTS_PER_HIT = 4


def plan_transfers(
    squad_players: list[Player],
    all_players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict,
    bank: float,
    horizon: int = 6,
    free_transfers: int = 1,
) -> list[dict]:
    """Plan transfers across multiple gameweeks using greedy approach.

    Returns a list of dicts, one per GW in the horizon.
    """
    current_squad_ids = [p.id for p in squad_players]
    current_bank = bank
    ft = free_transfers

    current_gw = next((gw for gw in gameweeks if gw.is_next), None)
    if current_gw is None:
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
    if current_gw is None:
        return []

    player_map = {p.id: p for p in all_players}
    plan = []
    cumulative = 0.0

    for gw_offset in range(horizon):
        gw_id = current_gw.id + gw_offset
        gw_obj = next((g for g in gameweeks if g.id == gw_id), None)
        if gw_obj is None:
            break

        # Score players for this specific GW
        temp_gws = [Gameweek(id=gw_id, name=f"GW{gw_id}", finished=False,
                             is_current=False, is_next=True)]
        players_copy = deepcopy(all_players)
        score_players(players_copy, fixtures, temp_gws, teams, lookahead=1)
        pm = {p.id: p for p in players_copy}

        # Evaluate current squad xPts
        squad_xpts = sum(pm[pid].composite_score for pid in current_squad_ids if pid in pm)

        # Find best single transfer
        best_gain = 0.0
        best_out_id = None
        best_in_id = None

        team_counts: dict[int, int] = {}
        for pid in current_squad_ids:
            p = player_map.get(pid)
            if p:
                team_counts[p.team] = team_counts.get(p.team, 0) + 1

        for out_id in current_squad_ids:
            out_p = pm.get(out_id) or player_map.get(out_id)
            if not out_p:
                continue
            remaining_budget = current_bank + out_p.sell_value

            for candidate in players_copy:
                if candidate.id in current_squad_ids:
                    continue
                if candidate.position != out_p.position:
                    continue
                if candidate.cost > remaining_budget:
                    continue
                # Team constraint
                cand_team_count = team_counts.get(candidate.team, 0)
                if candidate.team != out_p.team and cand_team_count >= 3:
                    continue

                gain = candidate.composite_score - (pm.get(out_id) or out_p).composite_score
                if gain > best_gain:
                    best_gain = gain
                    best_out_id = out_id
                    best_in_id = candidate.id

        # Decide: make transfer or hold
        hit_cost = 0 if ft > 0 else POINTS_PER_HIT
        net_gain = best_gain - hit_cost if best_out_id else 0

        gw_plan: dict = {
            "gameweek": gw_id,
            "gameweek_name": gw_obj.name if gw_obj else f"GW{gw_id}",
            "free_transfers": ft,
            "squad_xpts": round(squad_xpts, 2),
        }

        if net_gain > 0.5 and best_out_id and best_in_id:
            out_p = player_map.get(best_out_id)
            in_p = player_map.get(best_in_id)
            gw_plan["transfer_out"] = {
                "id": out_p.id, "name": out_p.name,
                "position_name": out_p.position_name,
                "cost": out_p.cost,
            } if out_p else None
            gw_plan["transfer_in"] = {
                "id": in_p.id, "name": in_p.name,
                "position_name": in_p.position_name,
                "cost": in_p.cost,
            } if in_p else None
            gw_plan["hit_cost"] = hit_cost
            gw_plan["expected_points_gain"] = round(net_gain, 2)

            # Apply transfer
            current_squad_ids = [pid for pid in current_squad_ids if pid != best_out_id]
            current_squad_ids.append(best_in_id)
            current_bank = current_bank + out_p.sell_value - in_p.cost
            ft = min(max(ft - 1, 0) + 1, 5)
        else:
            gw_plan["transfer_out"] = None
            gw_plan["transfer_in"] = None
            gw_plan["hit_cost"] = 0
            gw_plan["expected_points_gain"] = 0
            ft = min(ft + 1, 5)

        cumulative += gw_plan["expected_points_gain"]
        gw_plan["cumulative_gain"] = round(cumulative, 2)
        plan.append(gw_plan)

    return plan


def recommend_chips(
    squad_players: list[Player],
    all_players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict,
) -> dict:
    """Recommend optimal GW to play each chip.

    Returns dict with keys: bench_boost, triple_captain, free_hit, wildcard.
    """
    current_gw = next((gw for gw in gameweeks if gw.is_next), None)
    if current_gw is None:
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
    if current_gw is None:
        return {}

    squad_ids = {p.id for p in squad_players}
    squad_teams = {p.team for p in squad_players}
    player_map = {p.id: p for p in all_players}
    teams_dict = {tid: t.short_name for tid, t in teams.items()}

    # Count fixtures per team per GW
    gw_team_fixtures: dict[int, dict[int, int]] = {}
    for f in fixtures:
        if f.gameweek is None or f.gameweek < current_gw.id:
            continue
        gw_team_fixtures.setdefault(f.gameweek, {})
        gw_team_fixtures[f.gameweek][f.home_team] = gw_team_fixtures[f.gameweek].get(f.home_team, 0) + 1
        gw_team_fixtures[f.gameweek][f.away_team] = gw_team_fixtures[f.gameweek].get(f.away_team, 0) + 1

    results = {}
    window = range(current_gw.id, min(current_gw.id + 10, 39))

    # --- BENCH BOOST: Best on DGW with strong bench ---
    best_bb = {"gw": None, "score": 0, "reason": "", "dgw_teams": 0}
    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        dgw_count = sum(1 for pid in squad_ids
                        if player_map.get(pid) and tf.get(player_map[pid].team, 0) >= 2)
        total_fixtures = sum(1 for pid in squad_ids
                             if player_map.get(pid) and tf.get(player_map[pid].team, 0) >= 1)
        score = dgw_count * 3 + total_fixtures
        if score > best_bb["score"]:
            best_bb = {"gw": gw_id, "score": score, "dgw_teams": dgw_count,
                       "reason": f"{dgw_count} DGW players, {total_fixtures}/15 with fixtures"}

    results["bench_boost"] = best_bb

    # --- TRIPLE CAPTAIN: Highest xPts player in user's squad in a DGW ---
    best_tc = {"gw": None, "score": 0, "reason": "", "player": ""}
    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        for pid in squad_ids:
            p = player_map.get(pid)
            if not p:
                continue
            if tf.get(p.team, 0) >= 2:
                tc_score = p.ep_next * 3 if p.ep_next else p.form * 2
                if tc_score > best_tc["score"]:
                    best_tc = {"gw": gw_id, "score": round(tc_score, 1),
                               "player": p.name,
                               "reason": f"{p.name} ({teams_dict.get(p.team, '???')}) DGW"}

    if not best_tc["gw"]:
        # Fallback: highest ep_next player in squad
        squad_in_map = [player_map[pid] for pid in squad_ids if pid in player_map]
        top = max(squad_in_map, key=lambda p: p.ep_next, default=None)
        if top:
            best_tc = {"gw": current_gw.id, "score": round(top.ep_next * 2, 1),
                       "player": top.name,
                       "reason": f"{top.name} highest xPts in squad (no DGW found)"}
    results["triple_captain"] = best_tc

    # --- FREE HIT: Best for BGW where squad has fewest fixtures ---
    best_fh = {"gw": None, "score": 0, "reason": "", "missing": 0}
    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        missing = sum(1 for pid in squad_ids
                      if player_map.get(pid) and tf.get(player_map[pid].team, 0) == 0)
        if missing >= 3 and missing > best_fh["missing"]:
            best_fh = {"gw": gw_id, "score": missing * 2,
                       "missing": missing,
                       "reason": f"{missing} squad players without fixture"}
    if not best_fh["gw"]:
        best_fh["reason"] = "No significant blank GWs detected"
    results["free_hit"] = best_fh

    # --- WILDCARD: GW where squad fixture difficulty spikes ---
    best_wc = {"gw": None, "score": 0, "reason": ""}
    for gw_id in window:
        temp_gws = [Gameweek(id=gw_id, name=f"GW{gw_id}", finished=False,
                             is_current=False, is_next=True)]
        players_copy = deepcopy(all_players)
        score_players(players_copy, fixtures, temp_gws, teams, lookahead=3)
        pm = {p.id: p for p in players_copy}
        squad_score = sum(pm[pid].composite_score for pid in squad_ids if pid in pm)
        # Best possible squad score
        from .optimizer import select_squad
        try:
            optimal = select_squad(players_copy, budget=200)  # High budget = unconstrained
            optimal_score = sum(p.composite_score for p in optimal.starting)
        except Exception:
            optimal_score = squad_score

        gap = optimal_score - squad_score
        if gap > best_wc["score"]:
            best_wc = {"gw": gw_id, "score": round(gap, 2),
                       "reason": f"Squad gap to optimal: {gap:.1f} score points"}

    results["wildcard"] = best_wc

    return results
