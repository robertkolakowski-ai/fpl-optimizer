from __future__ import annotations

import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import httpx

from .analyzer import score_players
from .api import (
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
    """Top captain picks for the upcoming gameweek."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, lookahead=1)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        # Outfield players sorted by composite score (1-GW lookahead)
        candidates = sorted(
            [p for p in players if p.position in (2, 3, 4)],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:10]

        picks = []
        for p in candidates:
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
            })

        # Re-score with normal lookahead to not affect cache
        score_players(players, fixtures, gameweeks, lookahead=5)

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
    """High-performing players with low ownership."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, lookahead=5)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        max_ownership = float(request.args.get("max_ownership", 10.0))

        diffs = sorted(
            [p for p in players if p.selected_by_percent <= max_ownership and p.composite_score > 0],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:20]

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
                "xG": p.xG,
                "xA": p.xA,
                "selected_by_percent": p.selected_by_percent,
                "composite_score": round(p.composite_score, 3),
            } for p in diffs],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare 2-3 players side by side."""
    try:
        data = request.get_json(force=True)
        player_ids = data.get("player_ids", [])
        if len(player_ids) < 2 or len(player_ids) > 3:
            return jsonify({"error": "Provide 2-3 player IDs"}), 400

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, lookahead=5)
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
                "goals": p.goals,
                "assists": p.assists,
                "clean_sheets": p.clean_sheets,
                "bonus": p.bonus,
                "minutes": p.minutes,
                "ict_index": p.ict_index,
                "selected_by_percent": p.selected_by_percent,
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


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.get_json(force=True)
    budget = float(data.get("budget", 100.0))
    lookahead = int(data.get("lookahead", 5))
    user_id = data.get("user_id", "").strip() if data.get("user_id") else ""

    try:
        players, teams, gameweeks, fixtures = load_data()
        score_players(players, fixtures, gameweeks, lookahead=lookahead)

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
        squad = select_squad(players, budget=budget)
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
        }

        if user_team_info:
            result["user_team"] = user_team_info

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
