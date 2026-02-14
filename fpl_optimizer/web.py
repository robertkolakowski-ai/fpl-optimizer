from __future__ import annotations

import random
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import httpx

from .analyzer import compute_projected_minutes, compute_rotation_risk, score_players
from .api import (
    fetch_entry_history,
    fetch_league_standings,
    fetch_live_gameweek,
    fetch_user_entry,
    fetch_user_picks_full,
    load_data,
    load_user_team,
)
from .models import Player, Squad
from .optimizer import select_squad
from .transfers import suggest_transfers

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))

# In-memory cache for FPL data (avoids re-fetching on every tab switch)
_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached_data():
    """Load FPL data, caching for _CACHE_TTL seconds."""
    now = time.time()
    if _cache.get("data") and now - _cache.get("ts", 0) < _CACHE_TTL:
        return _cache["data"]

    players, teams, gameweeks, fixtures = load_data()
    _cache["data"] = (players, teams, gameweeks, fixtures)
    _cache["ts"] = now
    return players, teams, gameweeks, fixtures


POS_MAP = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
SORT_FIELDS = {
    "name", "team", "position", "cost", "total_points", "minutes",
    "goals", "assists", "clean_sheets", "bonus", "form",
    "points_per_game", "xG", "xA", "ict_index", "composite_score",
    "bps", "xG_per90", "xA_per90", "xGI_per90", "expected_goal_involvements",
    "selected_by_percent", "influence", "creativity", "threat",
    "goals_conceded", "saves", "ep_next",
}


@app.route("/")
def index():
    return render_template("web.html")


@app.route("/api/data")
def api_data():
    """Lightweight endpoint returning teams list for filter dropdowns."""
    try:
        _, teams, gameweeks, _ = _get_cached_data()
        teams_list = [t.to_dict() for t in teams.values()]
        teams_list.sort(key=lambda t: t["name"])
        gw_list = [{"id": gw.id, "name": gw.name, "finished": gw.finished,
                     "is_current": gw.is_current, "is_next": gw.is_next}
                    for gw in gameweeks]
        return jsonify({"teams": teams_list, "gameweeks": gw_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/players")
def api_players():
    """Returns all players with stats, supports filtering and sorting."""
    try:
        players, teams, _, _ = _get_cached_data()

        result = list(players)

        # Filter by position
        pos_filter = request.args.get("position", "").upper()
        if pos_filter and pos_filter in POS_MAP:
            pos_id = POS_MAP[pos_filter]
            result = [p for p in result if p.position == pos_id]

        # Filter by team
        team_filter = request.args.get("team", "")
        if team_filter:
            try:
                team_id = int(team_filter)
                result = [p for p in result if p.team == team_id]
            except ValueError:
                pass

        # Search by name (case-insensitive substring)
        search = request.args.get("search", "").strip().lower()
        if search:
            result = [p for p in result if search in p.name.lower()]

        # Filter by price range
        price_min = request.args.get("price_min", "")
        price_max = request.args.get("price_max", "")
        if price_min:
            try:
                result = [p for p in result if p.cost >= float(price_min)]
            except ValueError:
                pass
        if price_max:
            try:
                result = [p for p in result if p.cost <= float(price_max)]
            except ValueError:
                pass

        # Filter by form range
        form_min = request.args.get("form_min", "")
        if form_min:
            try:
                result = [p for p in result if p.form >= float(form_min)]
            except ValueError:
                pass

        # Filter by ownership range
        own_max = request.args.get("own_max", "")
        if own_max:
            try:
                result = [p for p in result if p.selected_by_percent <= float(own_max)]
            except ValueError:
                pass

        # Sort
        sort_by = request.args.get("sort", "total_points")
        if sort_by not in SORT_FIELDS:
            sort_by = "total_points"
        sort_dir = request.args.get("dir", "desc")
        reverse = sort_dir != "asc"
        result.sort(key=lambda p: getattr(p, sort_by, 0), reverse=reverse)

        # Build response with team names
        teams_dict = {tid: t.short_name for tid, t in teams.items()}
        players_out = []
        for p in result:
            d = p.to_dict()
            d["team_name"] = teams_dict.get(p.team, "???")
            players_out.append(d)

        return jsonify({"players": players_out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fixtures")
def api_fixtures():
    """Returns fixtures grouped by gameweek, with scores for finished matches."""
    try:
        _, teams, gameweeks, fixtures = _get_cached_data()
        teams_dict = {tid: t.to_dict() for tid, t in teams.items()}

        # Group fixtures by gameweek
        by_gw: dict[int, list] = {}
        for f in fixtures:
            gw = f.gameweek
            if gw is None:
                continue
            if gw not in by_gw:
                by_gw[gw] = []
            fd = f.to_dict()
            fd["home_team_name"] = teams_dict.get(f.home_team, {}).get("short_name", "???")
            fd["away_team_name"] = teams_dict.get(f.away_team, {}).get("short_name", "???")
            by_gw[gw].append(fd)

        # Build ordered list
        gw_list = []
        for gw in sorted(by_gw.keys()):
            gw_info = next((g for g in gameweeks if g.id == gw), None)
            gw_list.append({
                "gameweek": gw,
                "name": gw_info.name if gw_info else f"Gameweek {gw}",
                "finished": gw_info.finished if gw_info else False,
                "is_current": gw_info.is_current if gw_info else False,
                "is_next": gw_info.is_next if gw_info else False,
                "fixtures": by_gw[gw],
            })

        return jsonify({"gameweeks": gw_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/<int:user_id>")
def api_user(user_id):
    """Fetch FPL user entry: manager name, team name, and leagues."""
    try:
        with httpx.Client(timeout=30) as client:
            entry = fetch_user_entry(client, user_id)
        manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
        team_name = entry.get("name", "")
        leagues = []
        for league_type in ("classic", "h2h"):
            for lg in entry.get("leagues", {}).get(league_type, []):
                leagues.append({"id": lg["id"], "name": lg["name"]})
        return jsonify({
            "manager_name": manager_name,
            "team_name": team_name,
            "leagues": leagues,
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "FPL user not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/league/<int:league_id>")
def api_league(league_id):
    """Fetch classic league standings."""
    try:
        with httpx.Client(timeout=30) as client:
            data = fetch_league_standings(client, league_id)
        league_name = data.get("league", {}).get("name", "")
        standings = []
        for s in data.get("standings", {}).get("results", []):
            standings.append({
                "rank": s.get("rank"),
                "entry": s.get("entry"),
                "manager_name": s.get("player_name", ""),
                "team_name": s.get("entry_name", ""),
                "gw_points": s.get("event_total", 0),
                "total_points": s.get("total", 0),
            })
        return jsonify({"league_name": league_name, "standings": standings})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "League not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/<int:user_id>")
def api_live(user_id):
    """Live GW points for a user's team."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            return jsonify({"error": "No active gameweek"}), 400

        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        # Build upcoming fixtures (next 3 GWs) per team
        gw_start = current_gw.id + 1
        team_upcoming: dict[int, list] = {}
        for f in fixtures:
            if f.gameweek is None or f.gameweek < gw_start or f.gameweek >= gw_start + 3:
                continue
            for team_id, opp_id, is_home, diff in [
                (f.home_team, f.away_team, True, f.home_difficulty),
                (f.away_team, f.home_team, False, f.away_difficulty),
            ]:
                team_upcoming.setdefault(team_id, []).append({
                    "gw": f.gameweek,
                    "opponent": teams_dict.get(opp_id, "???"),
                    "is_home": is_home,
                    "difficulty": diff,
                })
        for tid in team_upcoming:
            team_upcoming[tid].sort(key=lambda x: x["gw"])

        with httpx.Client(timeout=30) as client:
            picks_data = fetch_user_picks_full(client, user_id, current_gw.id)
            live_data = fetch_live_gameweek(client, current_gw.id)

        live_map = {e["id"]: e["stats"] for e in live_data.get("elements", [])}
        picks = picks_data.get("picks", [])
        entry_history = picks_data.get("entry_history", {})

        pick_list = []
        total_live = 0
        for pick in picks:
            pid = pick["element"]
            stats = live_map.get(pid, {})
            pts = stats.get("total_points", 0)
            multiplier = pick.get("multiplier", 1)
            live_pts = pts * multiplier
            is_captain = pick.get("is_captain", False)
            is_vice = pick.get("vice_captain", False)
            p = player_map.get(pid)
            pick_list.append({
                "id": pid,
                "name": p.name if p else f"ID {pid}",
                "team_name": teams_dict.get(p.team, "???") if p else "???",
                "position": p.position if p else 0,
                "position_name": p.position_name if p else "??",
                "photo": p.photo if p else "",
                "multiplier": multiplier,
                "is_captain": is_captain,
                "is_vice_captain": is_vice,
                "live_points": live_pts,
                "raw_points": pts,
                "minutes": stats.get("minutes", 0),
                "goals": stats.get("goals_scored", 0),
                "assists": stats.get("assists", 0),
                "bonus": stats.get("bonus", 0),
                "bps": stats.get("bps", 0),
                "saves": stats.get("saves", 0),
                "yellow_cards": stats.get("yellow_cards", 0),
                "red_cards": stats.get("red_cards", 0),
                "ep_next": p.ep_next if p else 0,
                "news": p.news if p else "",
                "chance_of_playing": p.chance_of_playing if p else None,
                "upcoming": team_upcoming.get(p.team, [])[:3] if p else [],
            })
            if multiplier > 0:
                total_live += live_pts

        # Fetch total season points and team value from entry
        try:
            with httpx.Client(timeout=30) as client2:
                entry_data = fetch_user_entry(client2, user_id)
            total_points = entry_data.get("summary_overall_points", 0)
            team_value = entry_data.get("last_deadline_value", 0)
        except Exception:
            total_points = entry_history.get("total_points", 0)
            team_value = 0

        return jsonify({
            "gameweek": current_gw.id,
            "gameweek_name": current_gw.name,
            "picks": pick_list,
            "total_live_points": total_live,
            "total_points": total_points,
            "team_value": team_value,
            "overall_rank": entry_history.get("overall_rank"),
            "points_on_bench": sum(
                p["live_points"] for p in pick_list if p["multiplier"] == 0
            ),
            "hits": entry_history.get("event_transfers_cost", 0),
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "Data not available yet for this gameweek"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fdr")
def api_fdr():
    """Fixture difficulty rating heatmap data."""
    try:
        _, teams, gameweeks, fixtures = _get_cached_data()
        lookahead = int(request.args.get("lookahead", 8))

        current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            return jsonify({"error": "No upcoming gameweeks"}), 400

        gw_start = current_gw.id
        gw_end = gw_start + lookahead
        gw_ids = list(range(gw_start, gw_end))

        teams_dict = {tid: t for tid, t in teams.items()}

        # Build team -> gw -> list of fixtures
        grid = {}
        for tid, t in teams_dict.items():
            grid[tid] = {"team_name": t.short_name, "team_full": t.name, "fixtures": {}}

        for f in fixtures:
            if f.gameweek is None or f.gameweek < gw_start or f.gameweek >= gw_end:
                continue
            # Home team entry
            grid.setdefault(f.home_team, {"team_name": "???", "team_full": "???", "fixtures": {}})
            grid[f.home_team]["fixtures"].setdefault(f.gameweek, []).append({
                "opponent": teams_dict[f.away_team].short_name if f.away_team in teams_dict else "???",
                "is_home": True,
                "difficulty": f.home_difficulty,
            })
            # Away team entry
            grid.setdefault(f.away_team, {"team_name": "???", "team_full": "???", "fixtures": {}})
            grid[f.away_team]["fixtures"].setdefault(f.gameweek, []).append({
                "opponent": teams_dict[f.home_team].short_name if f.home_team in teams_dict else "???",
                "is_home": False,
                "difficulty": f.away_difficulty,
            })

        rows = []
        for tid in sorted(grid.keys(), key=lambda t: grid[t]["team_name"]):
            row = {"team_id": tid, "team_name": grid[tid]["team_name"], "team_full": grid[tid]["team_full"], "gws": []}
            avg_diff = []
            for gw in gw_ids:
                fx_list = grid[tid]["fixtures"].get(gw, [])
                row["gws"].append(fx_list)
                for fx in fx_list:
                    avg_diff.append(fx["difficulty"])
            row["avg_difficulty"] = round(sum(avg_diff) / len(avg_diff), 2) if avg_diff else 3.0
            rows.append(row)

        # Sort by average difficulty (easiest first)
        rows.sort(key=lambda r: r["avg_difficulty"])

        return jsonify({"gw_ids": gw_ids, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/captain-picks")
def api_captain_picks():
    """Top captain picks with expected value for chart display."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        # Count fixtures per team for the next GW (DGW detection)
        current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        team_fixture_count: dict[int, int] = {}
        if current_gw:
            for f in fixtures:
                if f.gameweek == current_gw.id:
                    team_fixture_count[f.home_team] = team_fixture_count.get(f.home_team, 0) + 1
                    team_fixture_count[f.away_team] = team_fixture_count.get(f.away_team, 0) + 1

        # Outfield players sorted by composite score (1-GW lookahead)
        candidates = sorted(
            [p for p in players if p.position in (2, 3, 4)],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:15]

        picks = []
        for p in candidates:
            num_fixtures = team_fixture_count.get(p.team, 1)
            # Captain EV = ep_next * 2 (captain multiplier) * fixture multiplier
            # For DGW players, ep_next already accounts for double fixtures
            captain_ev = round(p.ep_next * 2, 2)
            # Breakdown components for chart
            form_component = round(p.form * 0.3, 2)
            fixture_component = round(p.fixture_difficulty * 3, 2)
            xg_component = round((p.xG + p.xA) * 0.4, 2)

            picks.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": p.xG,
                "xA": p.xA,
                "fixture_difficulty": round(p.fixture_difficulty, 2),
                "composite_score": round(p.composite_score, 3),
                "selected_by_percent": p.selected_by_percent,
                "ep_next": round(p.ep_next, 2),
                "captain_ev": captain_ev,
                "num_fixtures": num_fixtures,
                "is_dgw": num_fixtures >= 2,
                "ev_breakdown": {
                    "form": form_component,
                    "fixture": fixture_component,
                    "xgi": xg_component,
                },
            })

        # Re-score with normal lookahead to not affect cache
        score_players(players, fixtures, gameweeks, teams, lookahead=5)

        return jsonify({"picks": picks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/price-changes")
def api_price_changes():
    """Players likely to rise/fall in price based on transfer activity."""
    try:
        players, teams, _, _ = _get_cached_data()
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        def player_price_data(p):
            net_transfers = p.transfers_in_event - p.transfers_out_event
            return {
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "selected_by_percent": p.selected_by_percent,
                "transfers_in": p.transfers_in_event,
                "transfers_out": p.transfers_out_event,
                "net_transfers": net_transfers,
                "cost_change_event": p.cost_change_event,
                "form": p.form,
            }

        # Top risers: highest net transfers in
        risers = sorted(players, key=lambda p: p.transfers_in_event - p.transfers_out_event, reverse=True)[:15]
        # Top fallers: highest net transfers out
        fallers = sorted(players, key=lambda p: p.transfers_in_event - p.transfers_out_event)[:15]

        return jsonify({
            "risers": [player_price_data(p) for p in risers],
            "fallers": [player_price_data(p) for p in fallers],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/differentials")
def api_differentials():
    """Differential Finder 2.0: low-ownership players with high upside."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        max_ownership = float(request.args.get("max_ownership", 10.0))
        pos_filter = request.args.get("position", "").upper()
        pos_id = POS_MAP.get(pos_filter)

        candidates = [p for p in players
                      if p.selected_by_percent <= max_ownership
                      and p.composite_score > 0 and p.minutes >= 90]
        if pos_id:
            candidates = [p for p in candidates if p.position == pos_id]

        # Upside score: composite + fixture ease + form momentum, penalize rotation
        for p in candidates:
            form_momentum = max(0, p.form - p.points_per_game) * 0.2
            xgi_upside = (p.xG + p.xA) * 0.15
            fixture_boost = p.fixture_difficulty * 0.2
            rotation_penalty = p.rotation_risk * 0.15
            p._upside = p.composite_score + form_momentum + xgi_upside + fixture_boost - rotation_penalty

        candidates.sort(key=lambda p: p._upside, reverse=True)

        return jsonify({
            "differentials": [{
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": round(p.xG, 2),
                "xA": round(p.xA, 2),
                "xGI_per90": round(p.xGI_per90, 2),
                "ep_next": round(p.ep_next, 2),
                "fixture_difficulty": round(p.fixture_difficulty, 2),
                "projected_minutes": p.projected_minutes,
                "rotation_risk": round(p.rotation_risk, 2),
                "selected_by_percent": p.selected_by_percent,
                "composite_score": round(p.composite_score, 3),
                "upside_score": round(p._upside, 3),
            } for p in candidates[:25]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rotation-risk")
def api_rotation_risk():
    """Players flagged as rotation risks."""
    try:
        players, teams, _, _ = _get_cached_data()
        compute_rotation_risk(players)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        risky = sorted(
            [p for p in players if p.rotation_risk >= 0.2],
            key=lambda p: p.rotation_risk,
            reverse=True,
        )[:40]

        return jsonify({
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "team_name": teams_dict.get(p.team, "???"),
                    "position_name": p.position_name,
                    "rotation_risk": round(p.rotation_risk, 2),
                    "starts": p.starts,
                    "minutes": p.minutes,
                    "chance_of_playing": p.chance_of_playing,
                    "news": p.news,
                    "selected_by_percent": p.selected_by_percent,
                    "cost": p.cost,
                    "form": p.form,
                    "total_points": p.total_points,
                }
                for p in risky
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/minutes-projection")
def api_minutes_projection():
    """Projected minutes per GW for all players."""
    try:
        players, teams, gameweeks, _ = _get_cached_data()
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        result = list(players)

        # Filter by position
        pos_filter = request.args.get("position", "").upper()
        if pos_filter and pos_filter in POS_MAP:
            result = [p for p in result if p.position == POS_MAP[pos_filter]]

        # Only players with meaningful minutes
        min_minutes = int(request.args.get("min_minutes", 200))
        result = [p for p in result if p.minutes >= min_minutes]

        result.sort(key=lambda p: p.projected_minutes, reverse=True)
        result = result[:50]

        return jsonify({
            "players": [{
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "minutes": p.minutes,
                "starts": p.starts,
                "projected_minutes": p.projected_minutes,
                "rotation_risk": round(p.rotation_risk, 2),
                "chance_of_playing": p.chance_of_playing,
                "form": p.form,
                "total_points": p.total_points,
                "news": p.news,
            } for p in result]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare 2-4 players side by side."""
    try:
        data = request.get_json(force=True)
        player_ids = data.get("player_ids", [])
        if len(player_ids) < 2 or len(player_ids) > 4:
            return jsonify({"error": "Provide 2-4 player IDs"}), 400

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        result = []
        for pid in player_ids:
            p = player_map.get(pid)
            if not p:
                continue
            result.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": p.xG,
                "xA": p.xA,
                "xG_per90": round(p.xG_per90, 2),
                "xA_per90": round(p.xA_per90, 2),
                "expected_goal_involvements": round(p.expected_goal_involvements, 2),
                "goals": p.goals,
                "assists": p.assists,
                "clean_sheets": p.clean_sheets,
                "goals_conceded": p.goals_conceded,
                "bonus": p.bonus,
                "bps": p.bps,
                "minutes": p.minutes,
                "starts": p.starts,
                "ict_index": p.ict_index,
                "influence": round(p.influence, 1),
                "creativity": round(p.creativity, 1),
                "threat": round(p.threat, 1),
                "saves": p.saves,
                "selected_by_percent": p.selected_by_percent,
                "ep_next": round(p.ep_next, 1),
                "composite_score": round(p.composite_score, 3),
                "fixture_difficulty": round(p.fixture_difficulty, 2),
            })

        return jsonify({"players": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rival/<int:rival_id>")
def api_rival(rival_id):
    """Compare loaded user's squad with a rival's squad."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id parameter required"}), 400
        user_id = int(user_id)

        players, teams, gameweeks, _ = _get_cached_data()
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            return jsonify({"error": "Cannot determine current gameweek"}), 400

        with httpx.Client(timeout=30) as client:
            my_entry = fetch_user_entry(client, user_id)
            rival_entry = fetch_user_entry(client, rival_id)
            my_picks = fetch_user_picks_full(client, user_id, current_gw.id)
            rival_picks = fetch_user_picks_full(client, rival_id, current_gw.id)

        def pick_ids(picks_data):
            return {p["element"] for p in picks_data.get("picks", [])}

        def player_info(pid):
            p = player_map.get(pid)
            if not p:
                return {"id": pid, "name": f"ID {pid}", "team_name": "???", "position_name": "??", "cost": 0, "total_points": 0}
            return {"id": p.id, "name": p.name, "team_name": teams_dict.get(p.team, "???"), "position_name": p.position_name, "cost": p.cost, "total_points": p.total_points}

        my_ids = pick_ids(my_picks)
        rival_ids = pick_ids(rival_picks)
        shared = my_ids & rival_ids
        only_mine = my_ids - rival_ids
        only_rival = rival_ids - my_ids

        return jsonify({
            "my_manager": f"{my_entry.get('player_first_name', '')} {my_entry.get('player_last_name', '')}".strip(),
            "my_team": my_entry.get("name", ""),
            "my_total_points": my_entry.get("summary_overall_points", 0),
            "rival_manager": f"{rival_entry.get('player_first_name', '')} {rival_entry.get('player_last_name', '')}".strip(),
            "rival_team": rival_entry.get("name", ""),
            "rival_total_points": rival_entry.get("summary_overall_points", 0),
            "shared": [player_info(pid) for pid in sorted(shared)],
            "only_mine": [player_info(pid) for pid in sorted(only_mine)],
            "only_rival": [player_info(pid) for pid in sorted(only_rival)],
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/schedule")
def api_schedule():
    """Full fixture schedule with double/blank GW detection and CS probability."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        teams_dict = {tid: t for tid, t in teams.items()}

        # Compute per-team clean sheet rate using the first-choice GK
        from collections import defaultdict
        team_cs_rate = {}
        # Find the GK with most starts per team
        team_gks = defaultdict(list)
        for p in players:
            if p.position == 1 and p.starts > 0:
                team_gks[p.team].append(p)
        for tid, gks in team_gks.items():
            gk = max(gks, key=lambda g: g.starts)
            team_cs_rate[tid] = round(gk.clean_sheets / gk.starts, 2) if gk.starts > 0 else 0.0
        # Fill missing teams
        for tid in teams:
            if tid not in team_cs_rate:
                team_cs_rate[tid] = 0.0

        # Count fixtures per team per GW for double/blank detection
        team_gw_fixtures = defaultdict(lambda: defaultdict(list))
        all_gw_ids = set()

        for f in fixtures:
            gw = f.gameweek
            if gw is None:
                continue
            all_gw_ids.add(gw)
            fx_data = {
                "id": f.id,
                "kickoff_time": f.kickoff_time,
                "finished": f.finished,
                "started": f.started,
                "home_score": f.home_score,
                "away_score": f.away_score,
            }
            # Home team entry
            team_gw_fixtures[f.home_team][gw].append({
                **fx_data,
                "opponent": f.away_team,
                "opponent_name": teams_dict[f.away_team].short_name if f.away_team in teams_dict else "???",
                "opponent_full": teams_dict[f.away_team].name if f.away_team in teams_dict else "???",
                "is_home": True,
                "difficulty": f.home_difficulty,
                "opponent_cs_rate": team_cs_rate.get(f.away_team, 0),
            })
            # Away team entry
            team_gw_fixtures[f.away_team][gw].append({
                **fx_data,
                "opponent": f.home_team,
                "opponent_name": teams_dict[f.home_team].short_name if f.home_team in teams_dict else "???",
                "opponent_full": teams_dict[f.home_team].name if f.home_team in teams_dict else "???",
                "is_home": False,
                "difficulty": f.away_difficulty,
                "opponent_cs_rate": team_cs_rate.get(f.home_team, 0),
            })

        gw_ids = sorted(all_gw_ids)

        # Detect double/blank for each GW
        gw_info = []
        for gw_id in gw_ids:
            gw_obj = next((g for g in gameweeks if g.id == gw_id), None)
            total_fixtures = sum(
                len(team_gw_fixtures[tid].get(gw_id, []))
                for tid in teams_dict
            ) // 2  # each fixture counted twice
            has_blank = any(
                gw_id not in team_gw_fixtures[tid] for tid in teams_dict
            )
            has_double = any(
                len(team_gw_fixtures[tid].get(gw_id, [])) >= 2
                for tid in teams_dict
            )
            gw_info.append({
                "id": gw_id,
                "name": gw_obj.name if gw_obj else f"Gameweek {gw_id}",
                "finished": gw_obj.finished if gw_obj else False,
                "is_current": gw_obj.is_current if gw_obj else False,
                "is_next": gw_obj.is_next if gw_obj else False,
                "fixture_count": total_fixtures,
                "has_blank": has_blank,
                "has_double": has_double,
            })

        # Build per-team schedule
        schedule = []
        for tid in sorted(teams_dict.keys(), key=lambda t: teams_dict[t].short_name):
            t = teams_dict[tid]
            team_data = {
                "team_id": tid,
                "team_name": t.short_name,
                "team_full": t.name,
                "cs_rate": team_cs_rate.get(tid, 0),
                "gameweeks": {},
            }
            for gw_id in gw_ids:
                fxs = team_gw_fixtures[tid].get(gw_id, [])
                team_data["gameweeks"][gw_id] = {
                    "fixtures": fxs,
                    "is_blank": len(fxs) == 0,
                    "is_double": len(fxs) >= 2,
                }
            schedule.append(team_data)

        return jsonify({
            "gw_info": gw_info,
            "schedule": schedule,
            "team_cs_rates": {tid: rate for tid, rate in team_cs_rate.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


OPTA_SORT_FIELDS = {
    "bps", "influence", "creativity", "threat", "ict_index",
    "expected_goal_involvements", "expected_goals_conceded",
    "saves", "penalties_saved", "yellow_cards", "red_cards",
    "own_goals", "starts", "clean_sheets", "bonus", "total_points",
    "form", "xG", "xA", "minutes", "goals", "assists", "cost",
    "ep_next", "ep_this", "xG_per90", "xA_per90", "xGI_per90", "xGC_per90",
}


@app.route("/api/opta-stats")
def api_opta_stats():
    """Returns players with full Opta/BPS statistics."""
    try:
        players, teams, _, _ = _get_cached_data()
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        result = list(players)

        # Filter by position
        pos_filter = request.args.get("position", "").upper()
        if pos_filter and pos_filter in POS_MAP:
            result = [p for p in result if p.position == POS_MAP[pos_filter]]

        # Filter by team
        team_filter = request.args.get("team", "")
        if team_filter:
            try:
                team_id = int(team_filter)
                result = [p for p in result if p.team == team_id]
            except ValueError:
                pass

        # Search
        search = request.args.get("search", "").strip().lower()
        if search:
            result = [p for p in result if search in p.name.lower()]

        # Sort
        sort_by = request.args.get("sort", "bps")
        if sort_by not in OPTA_SORT_FIELDS:
            sort_by = "bps"
        sort_dir = request.args.get("dir", "desc")
        reverse = sort_dir != "asc"
        result.sort(key=lambda p: getattr(p, sort_by, 0), reverse=reverse)

        # Limit
        limit = min(int(request.args.get("limit", 100)), 200)
        result = result[:limit]

        players_out = []
        for p in result:
            players_out.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "total_points": p.total_points,
                "form": p.form,
                "minutes": p.minutes,
                "starts": p.starts,
                "goals": p.goals,
                "assists": p.assists,
                "clean_sheets": p.clean_sheets,
                "goals_conceded": p.goals_conceded,
                "bonus": p.bonus,
                "bps": p.bps,
                "influence": p.influence,
                "creativity": p.creativity,
                "threat": p.threat,
                "ict_index": p.ict_index,
                "xG": p.xG,
                "xA": p.xA,
                "expected_goal_involvements": p.expected_goal_involvements,
                "expected_goals_conceded": p.expected_goals_conceded,
                "saves": p.saves,
                "penalties_saved": p.penalties_saved,
                "penalties_missed": p.penalties_missed,
                "yellow_cards": p.yellow_cards,
                "red_cards": p.red_cards,
                "own_goals": p.own_goals,
                "selected_by_percent": p.selected_by_percent,
                "ep_next": p.ep_next,
                "ep_this": p.ep_this,
                "xG_per90": round(p.xG_per90, 2),
                "xA_per90": round(p.xA_per90, 2),
                "xGI_per90": round(p.xGI_per90, 2),
                "xGC_per90": round(p.xGC_per90, 2),
            })

        return jsonify({"players": players_out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transfer-planner/<int:user_id>")
def api_transfer_planner(user_id):
    """Returns user's current squad with full player data for the transfer planner."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            return jsonify({"error": "Cannot determine current gameweek"}), 400

        with httpx.Client(timeout=30) as client:
            entry = fetch_user_entry(client, user_id)
            picks_data = fetch_user_picks_full(client, user_id, current_gw.id)
            history = fetch_entry_history(client, user_id)

        bank = entry.get("last_deadline_bank", 0) / 10.0
        team_value = entry.get("last_deadline_value", 0) / 10.0

        # Determine free transfers from history
        current_history = None
        for h in history.get("current", []):
            if h.get("event") == current_gw.id:
                current_history = h
                break
        # FPL doesn't expose free transfers directly; estimate from entry
        free_transfers = 1  # default

        picks = picks_data.get("picks", [])
        squad = []
        for pick in picks:
            pid = pick["element"]
            p = player_map.get(pid)
            if not p:
                continue
            selling_price = pick.get("selling_price", p.cost * 10) / 10.0
            squad.append({
                "id": p.id,
                "name": p.name,
                "team": p.team,
                "team_name": teams_dict.get(p.team, "???"),
                "position": p.position,
                "position_name": p.position_name,
                "cost": p.cost,
                "selling_price": round(selling_price, 1),
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": p.xG,
                "xA": p.xA,
                "selected_by_percent": p.selected_by_percent,
                "composite_score": round(p.composite_score, 3),
                "photo": p.photo,
                "news": p.news,
                "chance_of_playing": p.chance_of_playing,
                "cost_change_event": p.cost_change_event,
                "transfers_in_event": p.transfers_in_event,
                "transfers_out_event": p.transfers_out_event,
                "multiplier": pick.get("multiplier", 1),
            })

        # Build replacement candidates grouped by position
        # For each position, top 30 players by composite score not in squad
        squad_ids = {pick["element"] for pick in picks}
        replacements = {}
        for pos in [1, 2, 3, 4]:
            pos_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[pos]
            candidates = sorted(
                [p for p in players if p.position == pos and p.id not in squad_ids],
                key=lambda p: p.composite_score,
                reverse=True,
            )[:30]
            replacements[pos_name] = [{
                "id": p.id,
                "name": p.name,
                "team": p.team,
                "team_name": teams_dict.get(p.team, "???"),
                "position": p.position,
                "position_name": p.position_name,
                "cost": p.cost,
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": p.xG,
                "xA": p.xA,
                "selected_by_percent": p.selected_by_percent,
                "transfers_in_event": p.transfers_in_event,
                "transfers_out_event": p.transfers_out_event,
                "cost_change_event": p.cost_change_event,
                "composite_score": round(p.composite_score, 3),
                "photo": p.photo,
            } for p in candidates]

        # Most transferred in/out (top 10 each)
        most_in = sorted(players, key=lambda p: p.transfers_in_event, reverse=True)[:10]
        most_out = sorted(players, key=lambda p: p.transfers_out_event, reverse=True)[:10]

        def transfer_info(p):
            return {
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "form": p.form,
                "selected_by_percent": p.selected_by_percent,
                "transfers_in_event": p.transfers_in_event,
                "transfers_out_event": p.transfers_out_event,
                "cost_change_event": p.cost_change_event,
            }

        return jsonify({
            "squad": squad,
            "bank": round(bank, 1),
            "team_value": round(team_value, 1),
            "free_transfers": free_transfers,
            "gameweek": current_gw.id,
            "replacements": replacements,
            "most_transferred_in": [transfer_info(p) for p in most_in],
            "most_transferred_out": [transfer_info(p) for p in most_out],
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found or data not available"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True)
    budget = float(data.get("budget", 100.0))
    lookahead = int(data.get("lookahead", 5))
    user_id = data.get("user_id", "").strip() if data.get("user_id") else ""
    risk_mode = data.get("risk_mode", "balanced")
    if risk_mode not in ("safe", "balanced", "aggressive"):
        risk_mode = "balanced"

    try:
        players, teams, gameweeks, fixtures = load_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=lookahead)
        # Compute rotation risk (needed for safe mode scoring)
        compute_rotation_risk(players)

        # If user ID provided, load their team and suggest transfers
        user_team_info = None
        if user_id:
            try:
                uid = int(user_id)
                squad_players, bank = load_user_team(uid, players, gameweeks)
                # Build a Squad from the user's current team
                user_squad = Squad(players=squad_players)
                user_squad.budget_remaining = bank
                # Select starting XI from user's squad
                from .optimizer import _select_starting
                _select_starting(user_squad)
                # Suggest transfers for their team
                user_transfers = suggest_transfers(
                    user_squad, players, budget=sum(p.cost for p in squad_players) + bank
                )
                user_team_info = {
                    "squad": user_squad.to_dict(),
                    "transfers": [t.to_dict() for t in user_transfers],
                    "bank": round(bank, 1),
                }
            except ValueError:
                return jsonify({"error": "Invalid FPL user ID"}), 400
            except Exception as e:
                user_team_info = {"error": str(e)}

        # Always run the full optimizer for the "optimal squad" section
        squad = select_squad(players, budget=budget, risk_mode=risk_mode)
        transfer_list = suggest_transfers(squad, players, budget=budget)

        # Top 10 per position
        pos_names = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}
        top_by_position = {}
        for pos, name in pos_names.items():
            ranked = sorted(
                [p for p in players if p.position == pos],
                key=lambda p: p.composite_score,
                reverse=True,
            )[:10]
            top_by_position[name] = [p.to_dict() for p in ranked]

        teams_dict = {str(tid): {"name": t.name, "short_name": t.short_name} for tid, t in teams.items()}

        result = {
            "squad": squad.to_dict(),
            "transfers": [t.to_dict() for t in transfer_list],
            "top_by_position": top_by_position,
            "teams": teams_dict,
            "risk_mode": risk_mode,
        }

        if user_team_info:
            result["user_team"] = user_team_info

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ownership")
def api_ownership():
    """Effective ownership modelling with captaincy estimates."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        # Estimate captaincy rate: top owned players get captained more
        # Simple model: captain rate ~ ownership * form ranking
        outfield = [p for p in players if p.position in (2, 3, 4) and p.selected_by_percent > 0]
        outfield.sort(key=lambda p: p.ep_next, reverse=True)

        # Top ~10 players by ep_next get most captaincy
        captain_pool_total = sum(p.selected_by_percent for p in outfield[:10])

        result = []
        for rank, p in enumerate(outfield[:50]):
            ownership = p.selected_by_percent
            # Estimated captaincy rate: proportional to ep_next share among top picks
            if rank < 10 and captain_pool_total > 0:
                captain_rate = round((p.selected_by_percent / captain_pool_total) * 100, 1)
            else:
                captain_rate = round(max(0, ownership * 0.02), 1)

            # Effective ownership = ownership + captain_rate (captain doubles points)
            eo = round(ownership + captain_rate, 1)

            # Net point swing: if you own and captain, positive. If you don't, negative.
            result.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "cost": p.cost,
                "ownership": ownership,
                "captain_rate": captain_rate,
                "effective_ownership": eo,
                "ep_next": round(p.ep_next, 2),
                "form": p.form,
                "total_points": p.total_points,
                # Expected point swing if player hauls (scores 10+)
                "haul_swing_own": round(10 * (1 - eo / 100), 2),
                "haul_swing_miss": round(-10 * (eo / 100), 2),
            })

        # Re-score with normal lookahead
        score_players(players, fixtures, gameweeks, teams, lookahead=5)

        return jsonify({"players": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transfer-gain/<int:user_id>")
def api_transfer_gain(user_id):
    """Transfer gain vs hit cost chart data for a user's squad."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        squad_players, bank = load_user_team(user_id, players, gameweeks)
        squad_ids = {p.id for p in squad_players}

        team_counts: dict[int, int] = {}
        for p in squad_players:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1

        # Evaluate every possible single transfer
        transfers = []
        for out_p in squad_players:
            remaining_budget = bank + out_p.cost
            for candidate in players:
                if candidate.id in squad_ids:
                    continue
                if candidate.position != out_p.position:
                    continue
                if candidate.cost > remaining_budget:
                    continue
                if candidate.team != out_p.team and team_counts.get(candidate.team, 0) >= 3:
                    continue

                pts_gain = candidate.ep_next - out_p.ep_next
                score_gain = candidate.composite_score - out_p.composite_score
                if score_gain <= 0:
                    continue

                # Breakeven: how many GWs until hit pays off
                breakeven_gws = round(4 / pts_gain, 1) if pts_gain > 0 else 99

                transfers.append({
                    "out_name": out_p.name,
                    "out_team": teams_dict.get(out_p.team, "???"),
                    "out_cost": out_p.cost,
                    "out_ep": round(out_p.ep_next, 2),
                    "in_name": candidate.name,
                    "in_team": teams_dict.get(candidate.team, "???"),
                    "in_cost": candidate.cost,
                    "in_ep": round(candidate.ep_next, 2),
                    "position_name": out_p.position_name,
                    "pts_gain": round(pts_gain, 2),
                    "score_gain": round(score_gain, 3),
                    "cost_change": round(candidate.cost - out_p.cost, 1),
                    "net_gain_free": round(pts_gain, 2),
                    "net_gain_hit": round(pts_gain - 4, 2),
                    "worth_hit": pts_gain > 4,
                    "breakeven_gws": breakeven_gws,
                })

        transfers.sort(key=lambda t: t["pts_gain"], reverse=True)
        return jsonify({"transfers": transfers[:30], "hit_cost": 4, "bank": round(bank, 1)})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/multi-gw/<int:user_id>")
def api_multi_gw(user_id):
    """Multi-gameweek transfer plan."""
    try:
        horizon = int(request.args.get("horizon", 6))
        horizon = max(3, min(6, horizon))

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)

        squad_players, bank = load_user_team(user_id, players, gameweeks)

        from .multi_gw import plan_transfers
        plan = plan_transfers(squad_players, players, fixtures, gameweeks,
                              teams, bank, horizon)

        return jsonify({"plan": plan, "horizon": horizon, "bank": round(bank, 1)})
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chip-strategy/<int:user_id>")
def api_chip_strategy(user_id):
    """Chip strategy recommendations."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        squad_players, bank = load_user_team(user_id, players, gameweeks)

        from .multi_gw import recommend_chips
        recs = recommend_chips(squad_players, players, fixtures, gameweeks, teams)

        return jsonify(recs)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-simulation/<int:user_id>")
def api_rank_simulation(user_id):
    """Monte Carlo rank simulation over upcoming GWs."""
    try:
        num_sims = min(int(request.args.get("sims", 1000)), 5000)
        horizon = min(int(request.args.get("horizon", 5)), 10)

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)

        squad_players, bank = load_user_team(user_id, players, gameweeks)

        # Get current rank
        with httpx.Client(timeout=30) as client:
            entry = fetch_user_entry(client, user_id)
        current_rank = entry.get("summary_overall_rank", 500000)
        current_points = entry.get("summary_overall_points", 0)

        # Squad expected points per GW (sum of ep_next for starting XI)
        squad_ep = sorted(squad_players, key=lambda p: p.ep_next, reverse=True)
        # Approximate starting XI: best 11 by ep_next respecting 1 GK
        gks = [p for p in squad_ep if p.position == 1]
        outfield = [p for p in squad_ep if p.position != 1]
        starting_ep = sum(p.ep_next for p in gks[:1]) + sum(p.ep_next for p in outfield[:10])

        # Average GW score across all managers (~50 pts)
        avg_gw_score = 50.0

        # Run simulations
        final_points = []
        for _ in range(num_sims):
            sim_total = current_points
            for gw in range(horizon):
                # Your score: normal distribution around ep_next with variance
                my_score = max(0, random.gauss(starting_ep, starting_ep * 0.35))
                sim_total += my_score
            final_points.append(sim_total)

        final_points.sort(reverse=True)

        # Estimate rank change based on points gained vs average
        avg_total_gain = sum(final_points) / len(final_points) - current_points
        avg_others_gain = avg_gw_score * horizon

        # Rough rank model: ~200 points per 100k ranks around top 500k
        pts_per_100k = 200
        rank_delta = -(avg_total_gain - avg_others_gain) / pts_per_100k * 100000

        # Percentiles
        p10 = final_points[int(num_sims * 0.9)]
        p25 = final_points[int(num_sims * 0.75)]
        p50 = final_points[int(num_sims * 0.5)]
        p75 = final_points[int(num_sims * 0.25)]
        p90 = final_points[int(num_sims * 0.1)]

        # Build histogram buckets for chart
        min_pts = min(final_points)
        max_pts = max(final_points)
        bucket_size = max(1, (max_pts - min_pts) / 20)
        buckets = []
        for i in range(20):
            lo = min_pts + i * bucket_size
            hi = lo + bucket_size
            count = sum(1 for p in final_points if lo <= p < hi)
            buckets.append({"min": round(lo, 1), "max": round(hi, 1), "count": count})

        return jsonify({
            "current_rank": current_rank,
            "current_points": current_points,
            "squad_ep_per_gw": round(starting_ep, 1),
            "horizon": horizon,
            "simulations": num_sims,
            "projected_points": {
                "p10": round(p10, 1),
                "p25": round(p25, 1),
                "median": round(p50, 1),
                "p75": round(p75, 1),
                "p90": round(p90, 1),
            },
            "projected_rank_change": round(rank_delta),
            "projected_rank": max(1, round(current_rank + rank_delta)),
            "histogram": buckets,
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/fixture-swing")
def api_fixture_swing():
    """Fixture swing data: per-team difficulty over upcoming GWs for rotation planning."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        teams_dict = {tid: t for tid, t in teams.items()}
        lookahead = min(int(request.args.get("lookahead", 8)), 12)

        current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            return jsonify({"error": "No upcoming gameweeks"}), 400

        gw_ids = list(range(current_gw.id, current_gw.id + lookahead))

        # Build team->gw->difficulty mapping
        team_gw_diff: dict[int, dict[int, list]] = {}
        for tid in teams_dict:
            team_gw_diff[tid] = {}

        for f in fixtures:
            if f.gameweek is None or f.gameweek < current_gw.id or f.gameweek >= current_gw.id + lookahead:
                continue
            team_gw_diff.setdefault(f.home_team, {}).setdefault(f.gameweek, []).append({
                "opponent": teams_dict[f.away_team].short_name if f.away_team in teams_dict else "???",
                "is_home": True,
                "difficulty": f.home_difficulty,
            })
            team_gw_diff.setdefault(f.away_team, {}).setdefault(f.gameweek, []).append({
                "opponent": teams_dict[f.home_team].short_name if f.home_team in teams_dict else "???",
                "is_home": False,
                "difficulty": f.away_difficulty,
            })

        # Compute swing: difference between best and worst stretch
        rows = []
        for tid, gws in team_gw_diff.items():
            diffs_per_gw = []
            fixtures_per_gw = []
            for gw_id in gw_ids:
                fx = gws.get(gw_id, [])
                fixtures_per_gw.append(fx)
                avg_d = sum(f["difficulty"] for f in fx) / len(fx) if fx else 3.0
                diffs_per_gw.append(round(avg_d, 2))

            # Swing = max diff - min diff over window
            swing = max(diffs_per_gw) - min(diffs_per_gw) if diffs_per_gw else 0

            rows.append({
                "team_id": tid,
                "team_name": teams_dict[tid].short_name if tid in teams_dict else "???",
                "gw_difficulties": diffs_per_gw,
                "gw_fixtures": fixtures_per_gw,
                "avg_difficulty": round(sum(diffs_per_gw) / len(diffs_per_gw), 2) if diffs_per_gw else 3.0,
                "swing": round(swing, 2),
            })

        rows.sort(key=lambda r: r["avg_difficulty"])

        # Rotation pairs: teams whose fixtures complement each other
        pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                # Combined min difficulty per GW
                combined = [min(ad, bd) for ad, bd in zip(a["gw_difficulties"], b["gw_difficulties"])]
                avg_combined = sum(combined) / len(combined) if combined else 3.0
                if avg_combined < 2.8:
                    pairs.append({
                        "team_a": a["team_name"],
                        "team_b": b["team_name"],
                        "combined_avg": round(avg_combined, 2),
                        "combined_per_gw": [round(c, 2) for c in combined],
                    })
        pairs.sort(key=lambda p: p["combined_avg"])

        return jsonify({"gw_ids": gw_ids, "teams": rows, "rotation_pairs": pairs[:10]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== DRAFT / WAIVERS ====================


def _draft_score(p: Player, fixture_ease: float) -> float:
    """Compute a draft ranking score ignoring cost (unlike FPL classic)."""
    pos = p.position
    if pos == 1:  # GK
        w = (0.15, 0.25, 0.0, 0.0, 0.30, 0.10, 0.05, 0.15)
    elif pos == 2:  # DEF
        w = (0.15, 0.20, 0.05, 0.05, 0.25, 0.10, 0.05, 0.15)
    elif pos == 3:  # MID
        w = (0.20, 0.15, 0.20, 0.15, 0.0, 0.10, 0.10, 0.10)
    else:  # FWD
        w = (0.20, 0.15, 0.25, 0.10, 0.0, 0.10, 0.10, 0.10)

    vals = [
        p.form, p.points_per_game, p.xG, p.xA,
        float(p.clean_sheets), float(p.bonus), p.ict_index, fixture_ease,
    ]
    return sum(a * b for a, b in zip(w, vals))


@app.route("/api/draft/rankings")
def api_draft_rankings():
    """Draft rankings: best available players ignoring price."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        # Rostered IDs (sent as comma-separated query param)
        rostered_raw = request.args.get("rostered", "")
        rostered_ids = set()
        if rostered_raw:
            rostered_ids = {int(x) for x in rostered_raw.split(",") if x.strip().isdigit()}

        # Position filter
        pos_filter = request.args.get("position", "").upper()
        pos_id = POS_MAP.get(pos_filter)

        # Compute draft scores
        for p in players:
            p._draft_score = _draft_score(p, p.fixture_difficulty)

        available = [p for p in players if p.id not in rostered_ids]
        if pos_id:
            available = [p for p in available if p.position == pos_id]

        # Min minutes filter (skip players with near-zero playing time)
        available = [p for p in available if p.minutes >= 90]

        available.sort(key=lambda p: p._draft_score, reverse=True)

        result = []
        for rank, p in enumerate(available[:80], 1):
            result.append({
                "rank": rank,
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "xG": round(p.xG, 2),
                "xA": round(p.xA, 2),
                "ict_index": round(p.ict_index, 1),
                "fixture_difficulty": round(p.fixture_difficulty, 2),
                "projected_minutes": p.projected_minutes,
                "rotation_risk": round(p.rotation_risk, 2),
                "selected_by_percent": p.selected_by_percent,
                "draft_score": round(p._draft_score, 3),
            })

        return jsonify({"players": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/draft/waiver-wire")
def api_draft_waiver_wire():
    """Waiver wire picks: best unrostered players by recent form and upside."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=3)
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        rostered_raw = request.args.get("rostered", "")
        rostered_ids = set()
        if rostered_raw:
            rostered_ids = {int(x) for x in rostered_raw.split(",") if x.strip().isdigit()}

        available = [p for p in players if p.id not in rostered_ids and p.minutes >= 90]

        # Waiver score: heavy weight on recent form + fixture ease (short-term value)
        def waiver_score(p):
            form_w = p.form * 0.35
            fixture_w = p.fixture_difficulty * 0.25
            ep_w = p.ep_next * 0.20
            xgi_w = (p.xG + p.xA) * 0.10
            mins_w = (p.projected_minutes / 90.0) * 0.10
            return form_w + fixture_w + ep_w + xgi_w + mins_w

        for p in available:
            p._waiver_score = waiver_score(p)

        available.sort(key=lambda p: p._waiver_score, reverse=True)

        # Group by position for structured recommendations
        picks_by_pos = {}
        for pos_name, pos_id in POS_MAP.items():
            pos_list = [p for p in available if p.position == pos_id][:10]
            picks_by_pos[pos_name] = [{
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "form": p.form,
                "ep_next": round(p.ep_next, 2),
                "points_per_game": p.points_per_game,
                "fixture_difficulty": round(p.fixture_difficulty, 2),
                "projected_minutes": p.projected_minutes,
                "rotation_risk": round(p.rotation_risk, 2),
                "xG": round(p.xG, 2),
                "xA": round(p.xA, 2),
                "waiver_score": round(p._waiver_score, 3),
                "selected_by_percent": p.selected_by_percent,
            } for p in pos_list]

        # Overall top 20 waiver picks
        top_picks = [{
            "id": p.id,
            "name": p.name,
            "team_name": teams_dict.get(p.team, "???"),
            "position_name": p.position_name,
            "form": p.form,
            "ep_next": round(p.ep_next, 2),
            "points_per_game": p.points_per_game,
            "fixture_difficulty": round(p.fixture_difficulty, 2),
            "projected_minutes": p.projected_minutes,
            "rotation_risk": round(p.rotation_risk, 2),
            "waiver_score": round(p._waiver_score, 3),
        } for p in available[:20]]

        return jsonify({"top_picks": top_picks, "by_position": picks_by_pos})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/draft/my-roster", methods=["POST"])
def api_draft_my_roster():
    """Analyse a draft roster: starters, bench ranking, weaknesses."""
    try:
        data = request.get_json(force=True)
        roster_ids = data.get("roster_ids", [])
        if not roster_ids or len(roster_ids) > 15:
            return jsonify({"error": "Provide 1-15 player IDs"}), 400

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        roster = [player_map[pid] for pid in roster_ids if pid in player_map]
        if not roster:
            return jsonify({"error": "No valid players found"}), 400

        # Rank roster players by draft score
        for p in roster:
            p._draft_score = _draft_score(p, p.fixture_difficulty)
        roster.sort(key=lambda p: p._draft_score, reverse=True)

        # Identify weakest position
        pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        pos_scores = {1: [], 2: [], 3: [], 4: []}
        for p in roster:
            pos_counts[p.position] += 1
            pos_scores[p.position].append(p._draft_score)

        pos_avg = {}
        for pos in (1, 2, 3, 4):
            scores = pos_scores[pos]
            pos_avg[pos] = sum(scores) / len(scores) if scores else 0

        weakest_pos = min(pos_avg, key=pos_avg.get) if pos_avg else 3
        pos_name_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        result_players = []
        for p in roster:
            result_players.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "total_points": p.total_points,
                "form": p.form,
                "points_per_game": p.points_per_game,
                "fixture_difficulty": round(p.fixture_difficulty, 2),
                "projected_minutes": p.projected_minutes,
                "rotation_risk": round(p.rotation_risk, 2),
                "draft_score": round(p._draft_score, 3),
            })

        return jsonify({
            "roster": result_players,
            "roster_size": len(roster),
            "position_breakdown": {pos_name_map[k]: v for k, v in pos_counts.items()},
            "weakest_position": pos_name_map[weakest_pos],
            "total_draft_score": round(sum(p._draft_score for p in roster), 2),
            "avg_form": round(sum(p.form for p in roster) / len(roster), 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
