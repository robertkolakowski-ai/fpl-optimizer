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
    """Recommend optimal GW to play each chip with verbal advice.

    Implements 16 rules: hard rules (availability, budget, team limits),
    soft rules (form, fixtures, minutes), strategic rules (chip-specific),
    and risk rules (scenario robustness, diversification, timing).
    """
    current_gw = next((gw for gw in gameweeks if gw.is_next), None)
    if current_gw is None:
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
    if current_gw is None:
        return {}

    last_gw_id = max((gw.id for gw in gameweeks), default=38)
    gws_remaining = last_gw_id - current_gw.id + 1

    squad_ids = {p.id for p in squad_players}
    player_map = {p.id: p for p in all_players}
    teams_dict = {tid: t.short_name for tid, t in teams.items()}

    # ===== HARD RULE 1: Availability filter (<75% = excluded) =====
    def is_available(p: Player) -> bool:
        return p.chance_of_playing is None or p.chance_of_playing >= 75

    def availability_weight(p: Player) -> float:
        if p.chance_of_playing is None:
            return 1.0
        return p.chance_of_playing / 100.0

    # ===== RULE 2: Minutes probability =====
    def has_reliable_minutes(p: Player) -> bool:
        """Player must have >70 projected minutes (non-rotation, fit)."""
        if p.projected_minutes > 0:
            return p.projected_minutes >= 70
        # Fallback: check starts ratio
        approx_apps = p.minutes / 90.0 if p.minutes > 0 else 0
        if approx_apps >= 3 and p.starts > 0:
            return (p.starts / approx_apps) >= 0.65
        return p.minutes > 0 and p.rotation_risk < 0.4

    # ===== RULE 3: DGW/BGW fixture mapping =====
    gw_team_fixtures: dict[int, dict[int, int]] = {}
    gw_fixture_details: dict[int, list[Fixture]] = {}
    for f in fixtures:
        if f.gameweek is None or f.gameweek < current_gw.id:
            continue
        gw_team_fixtures.setdefault(f.gameweek, {})
        gw_team_fixtures[f.gameweek][f.home_team] = gw_team_fixtures[f.gameweek].get(f.home_team, 0) + 1
        gw_team_fixtures[f.gameweek][f.away_team] = gw_team_fixtures[f.gameweek].get(f.away_team, 0) + 1
        gw_fixture_details.setdefault(f.gameweek, []).append(f)

    # ===== RULE 4: Fixture difficulty per GW for squad =====
    def squad_avg_difficulty(gw_id: int) -> float:
        fxs = gw_fixture_details.get(gw_id, [])
        diffs = []
        for pid in squad_ids:
            p = player_map.get(pid)
            if not p:
                continue
            for f in fxs:
                if f.home_team == p.team:
                    diffs.append(f.home_difficulty)
                elif f.away_team == p.team:
                    diffs.append(f.away_difficulty)
        return sum(diffs) / len(diffs) if diffs else 3.0

    # ===== RULE 13: Risk diversification — count exposure =====
    def match_exposure(gw_id: int) -> dict:
        """How many squad players are in each match."""
        fxs = gw_fixture_details.get(gw_id, [])
        exposure: dict[str, int] = {}
        for f in fxs:
            key = f"{teams_dict.get(f.home_team, '?')} vs {teams_dict.get(f.away_team, '?')}"
            count = sum(1 for pid in squad_ids
                        if player_map.get(pid) and player_map[pid].team in (f.home_team, f.away_team))
            if count > 0:
                exposure[key] = count
        return exposure

    # ===== Identify starting XI and bench =====
    sorted_squad = sorted(squad_players, key=lambda p: p.ep_next, reverse=True)
    gks = [p for p in sorted_squad if p.position == 1]
    outfield = [p for p in sorted_squad if p.position != 1]
    starting_xi = gks[:1] + outfield[:10]
    starting_xi_ids = {p.id for p in starting_xi}
    bench_players = [p for p in squad_players if p.id not in starting_xi_ids]

    squad_unavailable = [p for p in squad_players if not is_available(p)]
    bench_unavailable = [p for p in bench_players if not is_available(p)]
    bench_rotation = [p for p in bench_players if not has_reliable_minutes(p) and is_available(p)]

    window = range(current_gw.id, min(current_gw.id + 10, last_gw_id + 1))

    results = {}

    # ================================================================
    # BENCH BOOST — maximize 15 players scoring
    # ================================================================
    best_bb = {"gw": None, "score": -999, "reason": "", "dgw_teams": 0,
               "confidence": "low", "advice": [], "warnings": []}

    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        advice = []
        warnings = []
        score = 0

        # Count DGW and fixture coverage
        dgw_players = [player_map[pid] for pid in squad_ids
                       if player_map.get(pid) and tf.get(player_map[pid].team, 0) >= 2]
        players_with_fixture = [player_map[pid] for pid in squad_ids
                                if player_map.get(pid) and tf.get(player_map[pid].team, 0) >= 1]

        # HARD: all 15 must have a fixture
        if len(players_with_fixture) < 15:
            blanks = 15 - len(players_with_fixture)
            warnings.append(f"{blanks} player(s) blanking — BB loses value")
            score -= blanks * 5

        # DGW bonus (Rule 3)
        score += len(dgw_players) * 4

        # Bench availability (Rule 1 hard stop)
        bench_fit = [p for p in bench_players if is_available(p)]
        bench_unfit = [p for p in bench_players if not is_available(p)]
        if bench_unfit:
            names = ", ".join(p.name for p in bench_unfit)
            warnings.append(f"Bench unavailable: {names}")
            score -= len(bench_unfit) * 8  # Heavy penalty

        # Bench minutes reliability (Rule 2)
        bench_rotation_risk = [p for p in bench_fit if not has_reliable_minutes(p)]
        if bench_rotation_risk:
            names = ", ".join(p.name for p in bench_rotation_risk)
            warnings.append(f"Rotation risk on bench: {names}")
            score -= len(bench_rotation_risk) * 3

        # Bench expected points
        bench_ep = sum(p.ep_next * availability_weight(p) for p in bench_players
                       if tf.get(p.team, 0) >= 1)
        score += bench_ep * 2

        # Fixture difficulty (Rule 4) — easy fixtures better for BB
        avg_diff = squad_avg_difficulty(gw_id)
        if avg_diff <= 2.5:
            score += 3
            advice.append("Favorable overall fixtures for your squad")
        elif avg_diff >= 3.5:
            score -= 2
            warnings.append("Tough fixtures reduce BB ceiling")

        # Diversification (Rule 13)
        expo = match_exposure(gw_id)
        over_exposed = {k: v for k, v in expo.items() if v >= 6}
        if over_exposed:
            for match, count in over_exposed.items():
                warnings.append(f"{count} players in {match} — heavy single-match exposure")
                score -= 2

        # Timing (Rule 14)
        if gws_remaining <= 3:
            advice.append("Late season — use BB now or lose it")
            score += 2

        if score > best_bb["score"]:
            # Build advice
            gw_advice = list(advice)
            if len(dgw_players) >= 5:
                gw_advice.insert(0, f"Strong DGW: {len(dgw_players)} players with double fixtures")
            elif len(dgw_players) > 0:
                gw_advice.insert(0, f"Partial DGW: only {len(dgw_players)} with double fixtures")
            else:
                gw_advice.insert(0, "No DGW — consider waiting for a double gameweek")

            if len(bench_fit) == len(bench_players) and not bench_rotation_risk:
                gw_advice.append("All bench players fit and nailed — ideal for BB")

            bench_ep_str = f"Bench expected points: {bench_ep:.1f}"
            gw_advice.append(bench_ep_str)

            confidence = "high" if len(dgw_players) >= 8 and not bench_unfit else \
                         "medium" if len(dgw_players) >= 3 and len(bench_unfit) == 0 else "low"

            best_bb = {
                "gw": gw_id, "score": round(score, 1),
                "dgw_teams": len(dgw_players),
                "reason": f"{len(dgw_players)} DGW players, {len(players_with_fixture)}/15 with fixtures, {len(bench_fit)}/{len(bench_players)} bench fit",
                "confidence": confidence,
                "advice": gw_advice,
                "warnings": list(warnings),
            }

    results["bench_boost"] = best_bb

    # ================================================================
    # TRIPLE CAPTAIN — highest ceiling player
    # ================================================================
    best_tc = {"gw": None, "score": 0, "reason": "", "player": "",
               "confidence": "low", "advice": [], "warnings": []}

    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        for pid in squad_ids:
            p = player_map.get(pid)
            if not p:
                continue
            # HARD RULE 1: availability >= 75%
            if not is_available(p):
                continue
            # RULE 2: must have reliable minutes
            if not has_reliable_minutes(p):
                continue

            n_fixtures = tf.get(p.team, 0)
            if n_fixtures == 0:
                continue

            advice = []
            warnings = []

            # Ceiling score: ep_next * fixtures, weighted by form and xGI
            ceiling = p.ep_next * n_fixtures
            # Boost for high xGI (attacking ceiling)
            ceiling += (p.xG + p.xA) * 0.5 * n_fixtures
            # Boost for form
            ceiling += p.form * 0.3

            # Rule 9: TC wants ceiling > consistency
            if p.position in (3, 4) and (p.xG + p.xA) > 0.3:
                advice.append(f"High attacking ceiling: {p.xG + p.xA:.1f} xGI")
                ceiling += 1

            # Fixture difficulty (Rule 4)
            fxs = gw_fixture_details.get(gw_id, [])
            player_diffs = []
            for f in fxs:
                if f.home_team == p.team:
                    player_diffs.append(f.home_difficulty)
                elif f.away_team == p.team:
                    player_diffs.append(f.away_difficulty)
            avg_p_diff = sum(player_diffs) / len(player_diffs) if player_diffs else 3
            if avg_p_diff <= 2.0:
                ceiling += 2
                advice.append(f"Easy fixture(s) (FDR {avg_p_diff:.0f})")
            elif avg_p_diff >= 4.0:
                ceiling -= 2
                warnings.append(f"Tough fixture(s) (FDR {avg_p_diff:.0f})")

            # Home advantage
            home_fixtures = sum(1 for f in fxs if f.home_team == p.team)
            if home_fixtures > 0:
                advice.append(f"{'All' if home_fixtures == n_fixtures else 'Includes'} home fixture(s)")
                ceiling += home_fixtures * 0.5

            # DGW bonus (Rule 3)
            if n_fixtures >= 2:
                advice.insert(0, f"Double gameweek — {n_fixtures} fixtures")
                ceiling *= 1.3

            # Rotation risk warning (Rule 9)
            if p.rotation_risk > 0.3:
                warnings.append(f"Rotation risk ({p.rotation_risk:.0%}) — minutes not guaranteed")
                ceiling *= 0.7

            # Timing (Rule 14)
            if gws_remaining <= 2 and n_fixtures < 2:
                advice.append("Last chance to use TC — consider even without DGW")
                ceiling += 1

            if ceiling > best_tc["score"]:
                tc_advice = list(advice)
                tc_advice.append(f"xPts ceiling: {ceiling:.1f} (base EP: {p.ep_next:.1f})")
                confidence = "high" if n_fixtures >= 2 and avg_p_diff <= 2.5 else \
                             "medium" if n_fixtures >= 2 or (p.form >= 6 and avg_p_diff <= 3) else "low"

                best_tc = {
                    "gw": gw_id, "score": round(ceiling, 1),
                    "player": p.name,
                    "reason": f"{p.name} ({teams_dict.get(p.team, '???')}) — {n_fixtures} fixture(s), FDR {avg_p_diff:.0f}",
                    "confidence": confidence,
                    "advice": tc_advice,
                    "warnings": list(warnings),
                }

    if not best_tc["gw"]:
        # Fallback
        fit_squad = [player_map[pid] for pid in squad_ids
                     if pid in player_map and is_available(player_map[pid]) and has_reliable_minutes(player_map[pid])]
        top = max(fit_squad, key=lambda p: p.ep_next, default=None)
        if top:
            best_tc = {
                "gw": current_gw.id, "score": round(top.ep_next * 2, 1),
                "player": top.name,
                "reason": f"{top.name} highest xPts (no DGW found in window)",
                "confidence": "low",
                "advice": ["No DGW found — consider holding TC for a future double gameweek",
                           f"{top.name} has the highest expected points ({top.ep_next:.1f}) in your squad"],
                "warnings": ["Using TC on a single GW is suboptimal — DGW doubles the value"],
            }
    results["triple_captain"] = best_tc

    # ================================================================
    # FREE HIT — short-term optimization for BGW
    # ================================================================
    best_fh = {"gw": None, "score": -999, "reason": "", "missing": 0,
               "confidence": "low", "advice": [], "warnings": []}

    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        advice = []
        warnings = []
        score = 0

        # Count players without fixtures (BGW impact)
        missing_fixture = [player_map[pid] for pid in squad_ids
                           if player_map.get(pid) and tf.get(player_map[pid].team, 0) == 0]
        # Count injured players with fixtures (also effectively missing)
        injured_with_fixture = [p for p in squad_unavailable if tf.get(p.team, 0) >= 1]

        total_missing = len(missing_fixture) + len(injured_with_fixture)
        score += total_missing * 3

        if missing_fixture:
            advice.append(f"{len(missing_fixture)} player(s) have no fixture — their teams are blanking")
        if injured_with_fixture:
            names = ", ".join(p.name for p in injured_with_fixture)
            advice.append(f"{len(injured_with_fixture)} injured player(s) would miss out: {names}")

        # Rule 6: FH lets you build best possible team for one GW
        # Check how many strong options exist in this GW
        total_gw_fixtures = len(gw_fixture_details.get(gw_id, []))
        if total_gw_fixtures < 8:
            advice.append(f"Reduced fixture list ({total_gw_fixtures} matches) — FH helps navigate")
            score += (10 - total_gw_fixtures) * 2

        # Fixture difficulty — does your squad face tough games?
        avg_diff = squad_avg_difficulty(gw_id)
        if avg_diff >= 3.5:
            advice.append(f"Squad faces tough fixtures (avg FDR {avg_diff:.1f}) — FH can target easier ones")
            score += 2

        # Rule 15: Scenario robustness — if only a few fixtures exist, risk of postponement is higher
        if total_gw_fixtures <= 5:
            warnings.append(f"Only {total_gw_fixtures} fixtures scheduled — postponement risk is higher")
            score -= 1

        # Rule 16: Exit strategy — FH doesn't affect your squad afterwards
        advice.append("Free Hit reverts your team next GW — no structural impact")

        # Rule 14: Timing — save FH for known BGW if possible
        if gws_remaining > 10 and total_missing < 5:
            warnings.append("Early in season — a bigger BGW may come later")
            score -= 2

        if total_missing < 3:
            continue  # Not worth FH unless significant disruption

        confidence = "high" if total_missing >= 7 else \
                     "medium" if total_missing >= 4 else "low"

        if score > best_fh["score"]:
            best_fh = {
                "gw": gw_id, "score": round(score, 1),
                "missing": total_missing,
                "reason": f"{len(missing_fixture)} blanking + {len(injured_with_fixture)} injured = {total_missing} unavailable",
                "confidence": confidence,
                "advice": list(advice),
                "warnings": list(warnings),
            }

    if not best_fh["gw"]:
        best_fh = {
            "gw": None, "score": 0, "missing": 0,
            "reason": "No significant blank gameweeks detected",
            "confidence": "low",
            "advice": ["No BGW with 3+ missing players found in the next 10 GWs",
                       "Hold your Free Hit for when fixture postponements or blanks are announced"],
            "warnings": [],
        }
    results["free_hit"] = best_fh

    # ================================================================
    # WILDCARD — long-term structural rebuild
    # ================================================================
    best_wc = {"gw": None, "score": -999, "reason": "",
               "confidence": "low", "advice": [], "warnings": []}

    # Assess current squad health
    injured_count = len(squad_unavailable)
    rotation_players = [p for p in squad_players if p.rotation_risk >= 0.4 and is_available(p)]
    low_form = [p for p in squad_players if p.form < 2.0 and p.minutes > 200]
    poor_fixtures_ahead = []

    # Check average fixture difficulty over next 5 GWs per player
    for p in squad_players:
        diffs = []
        for gw_off in range(5):
            gw_id = current_gw.id + gw_off
            for f in gw_fixture_details.get(gw_id, []):
                if f.home_team == p.team:
                    diffs.append(f.home_difficulty)
                elif f.away_team == p.team:
                    diffs.append(f.away_difficulty)
        if diffs and sum(diffs) / len(diffs) >= 3.8:
            poor_fixtures_ahead.append(p)

    # Score structural need for WC
    structural_issues = 0
    wc_advice = []
    wc_warnings = []

    if injured_count >= 3:
        structural_issues += injured_count * 2
        names = ", ".join(p.name for p in squad_unavailable[:5])
        wc_advice.append(f"{injured_count} injured/unavailable players: {names}")

    if len(rotation_players) >= 3:
        structural_issues += len(rotation_players)
        wc_advice.append(f"{len(rotation_players)} players with high rotation risk")

    if len(low_form) >= 4:
        structural_issues += len(low_form)
        names = ", ".join(p.name for p in low_form[:5])
        wc_advice.append(f"{len(low_form)} players in poor form (<2.0): {names}")

    if len(poor_fixtures_ahead) >= 6:
        structural_issues += len(poor_fixtures_ahead) - 3
        wc_advice.append(f"{len(poor_fixtures_ahead)} players face tough fixtures over next 5 GWs")

    # Rule 12: Team value awareness
    wc_advice.append("Wildcard resets selling prices — consider team value impact")

    # Rule 14: Timing
    if gws_remaining <= 5:
        wc_warnings.append("Very late to play WC — limited GWs to benefit from restructure")
        structural_issues -= 3
    elif gws_remaining >= 25:
        wc_advice.append("Early season — WC gives maximum remaining GWs to benefit")
    elif gws_remaining >= 15:
        wc_advice.append("Mid-season — good time to restructure if squad has issues")

    # Rule 16: Exit strategy
    wc_advice.append("After WC, plan your team for the next 5-8 GW fixture swing")

    # Find best GW for WC (where structural gap is worst)
    for gw_id in window:
        tf = gw_team_fixtures.get(gw_id, {})
        # Check if a DGW is coming soon — good to WC into it
        future_dgws = []
        for future_gw in range(gw_id, min(gw_id + 5, last_gw_id + 1)):
            ftf = gw_team_fixtures.get(future_gw, {})
            dgw_teams_count = sum(1 for v in ftf.values() if v >= 2)
            if dgw_teams_count >= 3:
                future_dgws.append(future_gw)

        gw_score = structural_issues

        # Bonus: WC before a DGW lets you build the perfect DGW squad
        if future_dgws:
            gw_score += len(future_dgws) * 3
            if gw_id not in future_dgws:
                wc_advice_gw = f"DGW coming in GW{future_dgws[0]} — WC to prepare"
            else:
                wc_advice_gw = f"DGW this week — WC to maximize it"
        else:
            wc_advice_gw = None

        if gw_score > best_wc["score"]:
            gw_advice = list(wc_advice)
            if wc_advice_gw:
                gw_advice.insert(0, wc_advice_gw)

            confidence = "high" if structural_issues >= 8 else \
                         "medium" if structural_issues >= 4 else "low"

            best_wc = {
                "gw": gw_id, "score": round(gw_score, 1),
                "reason": f"{injured_count} injured, {len(rotation_players)} rotation risks, {len(low_form)} poor form",
                "confidence": confidence,
                "advice": gw_advice,
                "warnings": list(wc_warnings),
            }

    if not best_wc["gw"]:
        best_wc["advice"] = ["Your squad structure looks solid — no urgent need for Wildcard",
                             "Save it for when fixture swings or injuries force multiple changes"]
        best_wc["warnings"] = []
    elif structural_issues < 3:
        best_wc["advice"].insert(0, "Squad is in decent shape — consider holding WC for later")

    results["wildcard"] = best_wc

    # ================================================================
    # GENERAL STRATEGIC ADVICE (Rule 14: timing overview)
    # ================================================================
    general_advice = []
    if gws_remaining <= 5:
        general_advice.append("End of season — use any remaining chips before they expire")
    if gws_remaining <= 3:
        general_advice.append("Final GWs — prioritize immediate impact over long-term planning")

    # Rule 10: Transfer cost awareness
    if injured_count >= 3:
        general_advice.append(f"With {injured_count} injured players, a WC or FH may save you multiple hits")

    # Rule 15: Scenario robustness
    general_advice.append("DGW/BGW schedules can change — monitor fixture announcements before committing a chip")

    results["general_advice"] = general_advice
    results["squad_health"] = {
        "injured": injured_count,
        "rotation_risk": len(rotation_players),
        "low_form": len(low_form),
        "poor_fixtures": len(poor_fixtures_ahead),
        "gws_remaining": gws_remaining,
    }

    return results
