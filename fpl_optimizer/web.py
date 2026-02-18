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
from .football_data import get_fd_team_id, get_match_referee, get_referee_data, get_team_stats, fetch_season_matches
from .predictions import generate_predictions, predict_cards, predict_corners, predict_shots, get_player_card_risks, predict_other_markets
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

    players, teams, gameweeks, fixtures, chip_windows = load_data()
    _cache["data"] = (players, teams, gameweeks, fixtures)
    _cache["chip_windows"] = chip_windows
    _cache["ts"] = now
    return players, teams, gameweeks, fixtures


# Predictions cache (separate, longer TTL)
_pred_cache: dict = {}
_PRED_CACHE_TTL = 900  # 15 minutes


def _get_cached_predictions(gw: int) -> list:
    """Return cached predictions for a gameweek, regenerating if stale."""
    now = time.time()
    cache_key = f"gw_{gw}"
    if _pred_cache.get(cache_key) and now - _pred_cache.get(f"{cache_key}_ts", 0) < _PRED_CACHE_TTL:
        return _pred_cache[cache_key]

    players, teams, gameweeks, fixtures = _get_cached_data()
    predictions = generate_predictions(fixtures, players, teams, gameweeks, target_gw=gw)
    _pred_cache[cache_key] = predictions
    _pred_cache[f"{cache_key}_ts"] = now
    _store_predictions(gw, predictions)
    return predictions


def _get_cached_team_stats():
    """Return team season stats derived from FPL data."""
    players, teams, gameweeks, fixtures = _get_cached_data()
    return get_team_stats(players, teams, fixtures, gameweeks)


def _get_cached_referee_data():
    """Return (raw_matches, referee_stats) from Football-Data.org."""
    return get_referee_data()


def _resolve_fd_team_id(fpl_team_id: int) -> int | None:
    """Map FPL team ID → Football-Data.org team ID via team code."""
    _, teams, _, _ = _get_cached_data()
    team = teams.get(fpl_team_id)
    if not team:
        return None
    return get_fd_team_id(team.code)


# Prediction Tracker (in-memory)
_prediction_history: list[dict] = []


def _store_predictions(gw: int, predictions: list) -> None:
    """Store predictions for accuracy tracking."""
    # Avoid duplicates
    existing_gws = {p["gameweek"] for p in _prediction_history}
    if gw in existing_gws:
        return
    for pred in predictions:
        _prediction_history.append({
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


def _update_actuals() -> None:
    """Check finished fixtures and fill in actual results."""
    _, _, _, fixtures = _get_cached_data()
    fixture_map = {f.id: f for f in fixtures}
    for entry in _prediction_history:
        if entry["actual_1x2"] is not None:
            continue
        f = fixture_map.get(entry["fixture_id"])
        if not f or not f.finished or f.home_score is None:
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


def _compute_tracker_accuracy() -> dict:
    """Calculate prediction accuracy from history."""
    _update_actuals()
    resolved = [e for e in _prediction_history if e["actual_1x2"] is not None]
    if not resolved:
        return {"total": 0, "result_accuracy": 0, "over25_accuracy": 0, "btts_accuracy": 0,
                "per_gw": [], "recent": []}

    correct_1x2 = sum(1 for e in resolved if e["predicted_1x2"] == e["actual_1x2"])
    correct_ou = sum(1 for e in resolved if e["predicted_over_25"] == e["actual_over_25"])
    correct_btts = sum(1 for e in resolved if e["predicted_btts"] == e["actual_btts"])
    n = len(resolved)

    # Per-GW breakdown
    gw_data: dict[int, dict] = {}
    for e in resolved:
        gw = e["gameweek"]
        if gw not in gw_data:
            gw_data[gw] = {"total": 0, "correct_1x2": 0, "correct_ou": 0, "correct_btts": 0}
        gw_data[gw]["total"] += 1
        if e["predicted_1x2"] == e["actual_1x2"]:
            gw_data[gw]["correct_1x2"] += 1
        if e["predicted_over_25"] == e["actual_over_25"]:
            gw_data[gw]["correct_ou"] += 1
        if e["predicted_btts"] == e["actual_btts"]:
            gw_data[gw]["correct_btts"] += 1

    per_gw = []
    for gw in sorted(gw_data):
        d = gw_data[gw]
        per_gw.append({
            "gameweek": gw,
            "total": d["total"],
            "result_pct": round(d["correct_1x2"] / d["total"] * 100, 1),
            "over25_pct": round(d["correct_ou"] / d["total"] * 100, 1),
            "btts_pct": round(d["correct_btts"] / d["total"] * 100, 1),
        })

    # Recent 10 predictions (newest first)
    recent = sorted(resolved, key=lambda e: (e["gameweek"], e["fixture_id"]), reverse=True)[:10]
    recent_out = []
    for e in recent:
        recent_out.append({
            "gameweek": e["gameweek"],
            "match": f"{e['home_team']} vs {e['away_team']}",
            "predicted_1x2": e["predicted_1x2"],
            "actual_1x2": e["actual_1x2"],
            "correct_1x2": e["predicted_1x2"] == e["actual_1x2"],
            "predicted_over_25": e["predicted_over_25"],
            "actual_over_25": e["actual_over_25"],
            "correct_over_25": e["predicted_over_25"] == e["actual_over_25"],
            "actual_score": f"{e['actual_home_score']}-{e['actual_away_score']}",
        })

    return {
        "total": n,
        "result_accuracy": round(correct_1x2 / n * 100, 1),
        "over25_accuracy": round(correct_ou / n * 100, 1),
        "btts_accuracy": round(correct_btts / n * 100, 1),
        "per_gw": per_gw,
        "recent": recent_out,
    }


def _get_chip_windows() -> list[dict]:
    """Return chip window definitions from cached bootstrap data."""
    if not _cache.get("chip_windows"):
        _get_cached_data()  # populates chip_windows as side-effect
    return _cache.get("chip_windows", [])


def _compute_chips_status(chip_windows: list[dict], user_chips_history: list[dict]) -> tuple[list[dict], list[str]]:
    """Compute chips_used and chips_available from bootstrap windows and user history.

    Each chip window (e.g. wildcard GW2-19, wildcard GW20-38) is a separate slot.
    A chip is available if the user hasn't used it in that window.
    """
    chips_used = []
    chips_available = []
    for window in chip_windows:
        name = window["name"]
        start = window["start_event"]
        stop = window["stop_event"]
        # Check if user used this chip in this window
        used_in_window = None
        for usage in user_chips_history:
            if usage.get("name") == name and start <= usage.get("event", 0) <= stop:
                used_in_window = usage
                break
        if used_in_window:
            chips_used.append({
                "name": name,
                "event": used_in_window["event"],
                "window": f"GW{start}-{stop}",
            })
        else:
            chips_available.append(name)
    return chips_used, chips_available


DEMO_USER_ID = 0


def _get_demo_squad():
    """Build a realistic demo squad from current player data using the optimizer.

    Caches the result alongside general data so it refreshes every TTL cycle.
    """
    now = time.time()
    if _cache.get("demo_squad") and now - _cache.get("demo_ts", 0) < _CACHE_TTL:
        return _cache["demo_squad"]

    players, teams, gameweeks, fixtures = _get_cached_data()
    score_players(players, fixtures, gameweeks, teams, lookahead=5)
    compute_rotation_risk(players)

    squad = select_squad(players, budget=99.0, risk_mode="balanced")

    # Determine captain (highest ep_next outfield starter) and vice
    outfield_starting = [p for p in squad.starting if p.position != 1]
    ranked = sorted(outfield_starting, key=lambda p: p.ep_next, reverse=True)
    captain = ranked[0] if ranked else squad.starting[0]
    vice = ranked[1] if len(ranked) > 1 else captain

    picks = []
    for i, p in enumerate(squad.starting):
        picks.append({
            "element": p.id,
            "position": i + 1,
            "multiplier": 2 if p.id == captain.id else 1,
            "is_captain": p.id == captain.id,
            "vice_captain": p.id == vice.id,
            "selling_price": int(p.cost * 10),
        })
    for i, p in enumerate(squad.bench):
        picks.append({
            "element": p.id,
            "position": len(squad.starting) + i + 1,
            "multiplier": 0,
            "is_captain": False,
            "vice_captain": False,
            "selling_price": int(p.cost * 10),
        })

    result = {
        "squad": squad,
        "bank": round(100.0 - squad.total_cost, 1),
        "picks": picks,
        "captain_id": captain.id,
    }
    _cache["demo_squad"] = result
    _cache["demo_ts"] = now
    return result


def _demo_user_team():
    """Return (squad_players, bank) for demo mode, matching load_user_team() signature."""
    demo = _get_demo_squad()
    return demo["squad"].players, demo["bank"]


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


@app.route("/api/cache-status")
def api_cache_status():
    """Return cache age info for the frontend freshness indicator."""
    ts = _cache.get("ts", 0)
    age = int(time.time() - ts) if ts else -1
    return jsonify({"age": age, "ttl": _CACHE_TTL})


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


@app.route("/api/player/<int:player_id>")
def api_player_detail(player_id):
    """Detailed player view with fixture ticker and composite breakdown."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        compute_projected_minutes(players, gameweeks)

        player_map = {p.id: p for p in players}
        p = player_map.get(player_id)
        if not p:
            return jsonify({"error": "Player not found"}), 404

        teams_dict = {tid: t.short_name for tid, t in teams.items()}
        teams_code_dict = {tid: t.code for tid, t in teams.items()}

        # Build fixture ticker (next 6 GWs)
        current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_current), None)

        fixture_ticker = []
        if current_gw:
            for f in fixtures:
                if f.gameweek is None or f.gameweek < current_gw.id or f.gameweek >= current_gw.id + 6:
                    continue
                if f.home_team == p.team:
                    fixture_ticker.append({
                        "gw": f.gameweek, "opponent": teams_dict.get(f.away_team, "???"),
                        "is_home": True, "difficulty": f.home_difficulty,
                    })
                elif f.away_team == p.team:
                    fixture_ticker.append({
                        "gw": f.gameweek, "opponent": teams_dict.get(f.home_team, "???"),
                        "is_home": False, "difficulty": f.away_difficulty,
                    })

        return jsonify({
            "player": p.to_dict(),
            "team_name": teams_dict.get(p.team, "???"),
            "team_code": teams_code_dict.get(p.team, 0),
            "fixture_ticker": sorted(fixture_ticker, key=lambda x: x["gw"]),
            "composite_breakdown": {
                "form": round(p.form, 2),
                "ppg": round(p.points_per_game, 2),
                "xG": round(p.xG, 2),
                "xA": round(p.xA, 2),
                "fixture_ease": round(p.fixture_difficulty, 2),
                "rotation_risk": round(p.rotation_risk, 2),
                "projected_minutes": p.projected_minutes,
            },
        })
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
    """Fetch FPL user entry with full profile: chips, bank, transfers, history."""
    if user_id == DEMO_USER_ID:
        try:
            demo = _get_demo_squad()
            tv = round(sum(p.cost for p in demo["squad"].players) + demo["bank"], 1)
            return jsonify({
                "manager_name": "Demo Manager",
                "team_name": "FPL Optimizer XI",
                "leagues": [{"id": 0, "name": "Demo League"}],
                "overall_rank": 247832,
                "overall_points": 1453,
                "team_value": tv,
                "bank": demo["bank"],
                "gw_points": 62,
                "gw_rank": 198432,
                "gw_transfers": 1,
                "gw_hits": 0,
                "free_transfers": 2,
                "chips_used": [],
                "chips_available": ["wildcard", "freehit", "bboost", "3xc", "wildcard", "freehit", "bboost", "3xc"],
                "season_history": [
                    {"gw": i, "points": 40 + (i * 3) + (i % 3) * 8, "rank": 300000 - i * 5000}
                    for i in range(1, 11)
                ],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    try:
        with httpx.Client(timeout=30) as client:
            entry = fetch_user_entry(client, user_id)
            history = fetch_entry_history(client, user_id)

        manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip()
        team_name = entry.get("name", "")
        leagues = []
        for league_type in ("classic", "h2h"):
            for lg in entry.get("leagues", {}).get(league_type, []):
                leagues.append({"id": lg["id"], "name": lg["name"]})

        # Profile data
        overall_rank = entry.get("summary_overall_rank")
        overall_points = entry.get("summary_overall_points", 0)
        team_value = entry.get("last_deadline_value", 0) / 10.0
        bank = entry.get("last_deadline_bank", 0) / 10.0

        # Chips — use bootstrap window definitions to handle 2-per-season chips
        chip_windows = _get_chip_windows()
        chips_used, chips_available = _compute_chips_status(
            chip_windows, history.get("chips", [])
        )

        # Current GW from history
        gw_history = history.get("current", [])
        latest_gw = gw_history[-1] if gw_history else {}
        gw_points = latest_gw.get("points", 0)
        gw_rank = latest_gw.get("rank")
        gw_transfers = latest_gw.get("event_transfers", 0)
        gw_hits = latest_gw.get("event_transfers_cost", 0)

        # Estimate free transfers
        free_transfers = 1
        if len(gw_history) >= 2:
            prev_transfers = gw_history[-2].get("event_transfers", 0)
            if prev_transfers == 0:
                free_transfers = min(free_transfers + 1, 5)

        # Season history (last 10 GWs for sparkline)
        season_history = [
            {"gw": h.get("event", 0), "points": h.get("points", 0), "rank": h.get("overall_rank", 0)}
            for h in gw_history[-10:]
        ]

        return jsonify({
            "manager_name": manager_name,
            "team_name": team_name,
            "leagues": leagues,
            "overall_rank": overall_rank,
            "overall_points": overall_points,
            "team_value": round(team_value, 1),
            "bank": round(bank, 1),
            "gw_points": gw_points,
            "gw_rank": gw_rank,
            "gw_transfers": gw_transfers,
            "gw_hits": gw_hits,
            "free_transfers": free_transfers,
            "chips_used": chips_used,
            "chips_available": chips_available,
            "season_history": season_history,
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
    if league_id == DEMO_USER_ID:
        return jsonify({
            "league_name": "Demo League",
            "standings": [
                {"rank": 1, "entry": 0, "manager_name": "Demo Manager", "team_name": "FPL Optimizer XI", "gw_points": 62, "total_points": 1453},
                {"rank": 2, "entry": 1, "manager_name": "Alice FPL", "team_name": "Alice's Aces", "gw_points": 58, "total_points": 1441},
                {"rank": 3, "entry": 2, "manager_name": "Bob Fantasy", "team_name": "Bob's Best XI", "gw_points": 71, "total_points": 1428},
                {"rank": 4, "entry": 3, "manager_name": "Carol Chips", "team_name": "Chip & Run", "gw_points": 45, "total_points": 1412},
                {"rank": 5, "entry": 4, "manager_name": "Dave Draft", "team_name": "Draft Day", "gw_points": 53, "total_points": 1398},
            ],
        })
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

        if user_id == DEMO_USER_ID:
            demo = _get_demo_squad()
            with httpx.Client(timeout=30) as client:
                live_data = fetch_live_gameweek(client, current_gw.id)
            picks = demo["picks"]
            entry_history = {}
        else:
            with httpx.Client(timeout=30) as client:
                picks_data = fetch_user_picks_full(client, user_id, current_gw.id)
                live_data = fetch_live_gameweek(client, current_gw.id)

        live_map = {e["id"]: e["stats"] for e in live_data.get("elements", [])}
        if user_id != DEMO_USER_ID:
            picks = picks_data.get("picks", [])
            entry_history = picks_data.get("entry_history", {})

        # Extract automatic subs
        auto_subs_raw = [] if user_id == DEMO_USER_ID else picks_data.get("automatic_subs", [])
        auto_sub_in_ids = {s["element_in"] for s in auto_subs_raw}
        auto_sub_out_ids = {s["element_out"] for s in auto_subs_raw}

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
            # Determine autosub status
            subbed_in = pid in auto_sub_in_ids
            subbed_out = pid in auto_sub_out_ids

            pick_list.append({
                "id": pid,
                "name": p.name if p else f"ID {pid}",
                "team": p.team if p else 0,
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
                "auto_sub_in": subbed_in,
                "auto_sub_out": subbed_out,
            })
            if multiplier > 0:
                total_live += live_pts

        # Fetch total season points and team value from entry
        if user_id == DEMO_USER_ID:
            total_points = 1453
            team_value = round(sum(p.cost for p in _get_demo_squad()["squad"].players) * 10)
        else:
            try:
                with httpx.Client(timeout=30) as client2:
                    entry_data = fetch_user_entry(client2, user_id)
                total_points = entry_data.get("summary_overall_points", 0)
                team_value = entry_data.get("last_deadline_value", 0)
            except Exception:
                total_points = entry_history.get("total_points", 0)
                team_value = 0

        # Determine if any matches are in progress for this GW
        gw_fixtures = [f for f in fixtures if f.gameweek == current_gw.id]
        is_live = any(f.started and not f.finished for f in gw_fixtures)

        return jsonify({
            "gameweek": current_gw.id,
            "gameweek_name": current_gw.name,
            "is_live": is_live,
            "picks": pick_list,
            "total_live_points": total_live,
            "total_points": total_points,
            "team_value": team_value,
            "overall_rank": entry_history.get("overall_rank"),
            "points_on_bench": sum(
                p["live_points"] for p in pick_list if p["multiplier"] == 0
            ),
            "hits": entry_history.get("event_transfers_cost", 0),
            "automatic_subs": [
                {
                    "element_in": s["element_in"],
                    "element_out": s["element_out"],
                    "in_name": player_map[s["element_in"]].name if s["element_in"] in player_map else f"ID {s['element_in']}",
                    "out_name": player_map[s["element_out"]].name if s["element_out"] in player_map else f"ID {s['element_out']}",
                }
                for s in auto_subs_raw
            ],
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

        if user_id == DEMO_USER_ID:
            demo = _get_demo_squad()
            bank = demo["bank"]
            team_value = round(sum(p.cost for p in demo["squad"].players) + bank, 1)
            free_transfers = 2
            picks = demo["picks"]
        else:
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
    bank_adjust = float(data.get("bank_adjust", 0))
    if risk_mode not in ("safe", "balanced", "aggressive"):
        risk_mode = "balanced"

    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=lookahead)
        # Compute rotation risk (needed for safe mode scoring)
        compute_rotation_risk(players)

        # If user ID provided, load their team and suggest transfers
        user_team_info = None
        if user_id:
            try:
                uid = int(user_id)
                if uid == DEMO_USER_ID:
                    squad_players, bank = _demo_user_team()
                else:
                    squad_players, bank = load_user_team(uid, players, gameweeks)
                # Apply bank adjustment (user correction for selling price differences)
                bank += bank_adjust
                # Build a Squad from the user's current team
                user_squad = Squad(players=squad_players)
                user_squad.budget_remaining = bank
                # Select starting XI from user's squad
                from .optimizer import _select_starting
                _select_starting(user_squad)
                # Suggest transfers for their team
                # Use sell_value (FPL selling prices) for accurate budget
                user_transfers = suggest_transfers(
                    user_squad, players, budget=sum(p.sell_value for p in squad_players) + bank
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
    """Enhanced EO modelling with personal risk, league comparison, and captain chart."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        outfield = [p for p in players if p.position in (2, 3, 4) and p.selected_by_percent > 0]
        outfield.sort(key=lambda p: p.ep_next, reverse=True)
        captain_pool_total = sum(p.selected_by_percent for p in outfield[:10])

        result = []
        for rank, p in enumerate(outfield[:50]):
            ownership = p.selected_by_percent
            if rank < 10 and captain_pool_total > 0:
                captain_rate = round((p.selected_by_percent / captain_pool_total) * 100, 1)
            else:
                captain_rate = round(max(0, ownership * 0.02), 1)
            eo = round(ownership + captain_rate, 1)
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
                "haul_swing_own": round(10 * (1 - eo / 100), 2),
                "haul_swing_miss": round(-10 * (eo / 100), 2),
            })

        # Personal risk exposure
        personal_risk = None
        user_id = request.args.get("user_id")
        if user_id:
            try:
                uid = int(user_id)
                if uid == DEMO_USER_ID:
                    squad_players, _ = _demo_user_team()
                else:
                    squad_players, _ = load_user_team(uid, players, gameweeks)
                squad_ids = {p.id for p in squad_players}

                current_gw = next((gw for gw in gameweeks if gw.is_current), None)
                if current_gw is None:
                    current_gw = next((gw for gw in gameweeks if gw.is_next), None)
                user_captain_id = None
                if current_gw:
                    if uid == DEMO_USER_ID:
                        demo = _get_demo_squad()
                        user_captain_id = demo["captain_id"]
                    else:
                        try:
                            with httpx.Client(timeout=30) as client:
                                picks_data = fetch_user_picks_full(client, uid, current_gw.id)
                            for pick in picks_data.get("picks", []):
                                if pick.get("is_captain"):
                                    user_captain_id = pick["element"]
                        except Exception:
                            pass

                for item in result:
                    item["user_owns"] = item["id"] in squad_ids
                    item["user_captains"] = item["id"] == user_captain_id

                owned_eo = sum(r["effective_ownership"] for r in result if r["user_owns"])
                top15_eo = sum(r["effective_ownership"] for r in result[:15]) or 1
                shield_score = round(owned_eo / top15_eo * 100, 1)
                attack_score = round(100 - shield_score, 1)

                personal_risk = {
                    "squad_ids": list(squad_ids),
                    "captain_id": user_captain_id,
                    "owned_eo_total": round(owned_eo, 1),
                    "missed_eo_total": round(sum(r["effective_ownership"] for r in result if not r.get("user_owns")), 1),
                    "shield_score": shield_score,
                    "attack_score": attack_score,
                    "recommendation": "SHIELD" if shield_score >= 60 else "ATTACK",
                    "recommendation_text": (
                        "Your team is template-heavy. Low variance, safe play."
                        if shield_score >= 60
                        else "Your team is differential. High variance, potential for big rank swings."
                    ),
                }
            except Exception:
                personal_risk = None

        # League EO comparison
        league_comparison = None
        league_id = request.args.get("league_id")
        if league_id and user_id:
            try:
                lid = int(league_id)
                uid = int(user_id)
                with httpx.Client(timeout=30) as client:
                    league_data = fetch_league_standings(client, lid)

                standings = league_data.get("standings", {}).get("results", [])[:10]
                current_gw = next((gw for gw in gameweeks if gw.is_current), None)
                if current_gw is None:
                    current_gw = next((gw for gw in gameweeks if gw.is_next), None)

                rival_ownership = []
                if current_gw:
                    with httpx.Client(timeout=30) as client:
                        for entry in standings:
                            eid = entry.get("entry")
                            if eid == uid:
                                continue
                            try:
                                rival_picks = fetch_user_picks_full(client, eid, current_gw.id)
                                rival_ids = {p["element"] for p in rival_picks.get("picks", [])}
                                rival_ownership.append({
                                    "manager": entry.get("player_name", ""),
                                    "entry_id": eid,
                                    "player_ids": list(rival_ids),
                                })
                            except Exception:
                                continue

                top_ids = [r["id"] for r in result[:15]]
                league_eo_grid = []
                for pid in top_ids:
                    p_info = next((r for r in result if r["id"] == pid), None)
                    if not p_info:
                        continue
                    rival_count = sum(1 for ro in rival_ownership if pid in ro["player_ids"])
                    league_eo_grid.append({
                        "id": pid,
                        "name": p_info["name"],
                        "rivals_owning": rival_count,
                        "total_rivals": len(rival_ownership),
                        "league_ownership_pct": round(rival_count / max(len(rival_ownership), 1) * 100, 1),
                    })

                league_comparison = {
                    "league_name": league_data.get("league", {}).get("name", ""),
                    "rivals_checked": len(rival_ownership),
                    "player_grid": league_eo_grid,
                }
            except Exception:
                league_comparison = None

        # Rank band EO adjustment
        rank_band = request.args.get("rank_band", "overall")
        if rank_band in ("10k", "100k"):
            for item in result:
                own = item["ownership"]
                ep = item["ep_next"]
                if rank_band == "10k":
                    # Meta players concentrated more in top 10k
                    if own > 30 and ep > 5:
                        mult = 1.3
                    elif own > 15:
                        mult = 1.15
                    elif own < 5:
                        mult = 0.5
                    else:
                        mult = 1.0
                else:  # 100k
                    if own > 30 and ep > 5:
                        mult = 1.15
                    elif own < 5:
                        mult = 0.7
                    else:
                        mult = 1.0
                item["ownership_adjusted"] = round(own * mult, 1)
                # Recalculate captain rate for this band
                adjusted_pool = sum(r.get("ownership_adjusted", r["ownership"]) for r in result[:10])
                if adjusted_pool > 0:
                    item["captain_rate"] = round((item.get("ownership_adjusted", own) / adjusted_pool) * 100, 1)
                item["effective_ownership"] = round(item.get("ownership_adjusted", own) + item["captain_rate"], 1)

        # Captain scenarios
        captain_scenarios = []
        for r in result[:6]:
            ep = r["ep_next"]
            eo = r["effective_ownership"]
            # Scenario: player scores a brace (14 pts for MID/FWD, 12 for DEF)
            haul_pts = 14
            blank_pts = 2
            # If you captain and they haul: you gain (haul * 2 - eo% * haul)
            haul_gain = round(haul_pts * 2 * (1 - eo / 100), 1)
            # If you don't captain and they haul: you lose
            haul_loss = round(-haul_pts * 2 * (eo / 100), 1)
            # If they blank and you captained:
            blank_cost = round(blank_pts * 2 * (1 - eo / 100), 1) if eo < 50 else round(-blank_pts * (eo / 100), 1)
            strategy = "PROTECT" if eo > 30 else "ATTACK"
            captain_scenarios.append({
                "id": r["id"], "name": r["name"], "eo": eo,
                "haul_gain_if_captain": haul_gain,
                "haul_loss_if_not_captain": haul_loss,
                "blank_impact": blank_cost,
                "strategy": strategy,
                "strategy_text": f"Captaining {r['name']} = {'protecting rank' if strategy == 'PROTECT' else 'attacking rank'}",
            })

        # Captain chart data
        captain_chart = [
            {"name": r["name"], "eo": r["effective_ownership"], "captain_rate": r["captain_rate"], "ep_next": r["ep_next"]}
            for r in result[:8]
        ]

        score_players(players, fixtures, gameweeks, teams, lookahead=5)

        response = {"players": result, "captain_chart": captain_chart, "captain_scenarios": captain_scenarios, "rank_band": rank_band}
        if personal_risk:
            response["personal_risk"] = personal_risk
        if league_comparison:
            response["league_comparison"] = league_comparison
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_reasoning(out_p, in_p):
    """Build short reasoning text for a transfer suggestion."""
    reasons = []
    fd_diff = in_p.fixture_difficulty - out_p.fixture_difficulty
    if fd_diff > 0.15:
        reasons.append(f"Better fixtures ahead (+{fd_diff:.0%} easier)")
    xgi_diff = (in_p.xG + in_p.xA) - (out_p.xG + out_p.xA)
    if xgi_diff > 0.5:
        reasons.append(f"Higher xGI ({in_p.xG + in_p.xA:.1f} vs {out_p.xG + out_p.xA:.1f})")
    if in_p.form > out_p.form + 1.0:
        reasons.append(f"Better form ({in_p.form} vs {out_p.form})")
    if out_p.rotation_risk > 0.3:
        reasons.append(f"{out_p.name} rotation risk ({out_p.rotation_risk:.0%})")
    if in_p.minutes > out_p.minutes * 1.2 and out_p.minutes > 0:
        reasons.append("More minutes played")
    return "; ".join(reasons[:3]) if reasons else "Higher composite score"


@app.route("/api/squad-analysis/<int:user_id>")
def api_squad_analysis(user_id):
    """Squad health summary: score, fixture outlook, strengths and weaknesses."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        player_map = {p.id: p for p in players}

        if user_id == DEMO_USER_ID:
            squad_players, bank = _demo_user_team()
        else:
            squad_players, bank = load_user_team(user_id, players, gameweeks)

        if not squad_players:
            return jsonify({"error": "No squad data available"}), 404

        # Squad score: avg composite_score as percentile (0-100)
        all_scores = sorted(p.composite_score for p in players if p.minutes > 0)
        squad_avg = sum(p.composite_score for p in squad_players) / len(squad_players)
        # Percentile: what fraction of all players score below squad_avg
        below = sum(1 for s in all_scores if s < squad_avg)
        squad_score = round(below / max(len(all_scores), 1) * 100)

        # Fixture outlook
        avg_fixture = sum(p.fixture_difficulty for p in squad_players) / len(squad_players)
        if avg_fixture > 0.6:
            fixture_outlook = "favorable"
        elif avg_fixture > 0.4:
            fixture_outlook = "neutral"
        else:
            fixture_outlook = "tough"

        # Weaknesses and strengths
        weaknesses = []
        strengths = []
        pos_names = {1: "Goalkeepers", 2: "Defense", 3: "Midfield", 4: "Attack"}
        for pos in (1, 2, 3, 4):
            group = [p for p in squad_players if p.position == pos]
            if not group:
                continue
            avg_fd = sum(p.fixture_difficulty for p in group) / len(group)
            avg_rot = sum(p.rotation_risk for p in group) / len(group)
            avg_cs = sum(p.composite_score for p in group) / len(group)
            if avg_fd < 0.35:
                weaknesses.append({
                    "area": pos_names[pos].lower(),
                    "description": f"{pos_names[pos]} facing tough fixtures (avg ease {avg_fd:.0%})",
                    "severity": "high",
                })
            elif avg_fd > 0.65:
                strengths.append({
                    "area": pos_names[pos].lower(),
                    "description": f"{pos_names[pos]} have favorable fixtures (avg ease {avg_fd:.0%})",
                })
            if avg_rot > 0.3:
                weaknesses.append({
                    "area": pos_names[pos].lower(),
                    "description": f"{pos_names[pos]} have high rotation risk (avg {avg_rot:.0%})",
                    "severity": "medium",
                })

        # Check for injured/flagged players
        flagged = [p for p in squad_players if p.chance_of_playing is not None and p.chance_of_playing < 100]
        if flagged:
            weaknesses.append({
                "area": "availability",
                "description": f"{len(flagged)} player{'s' if len(flagged) > 1 else ''} flagged with injury/doubt",
                "severity": "high" if any(p.chance_of_playing is not None and p.chance_of_playing < 50 for p in flagged) else "medium",
            })

        return jsonify({
            "squad_score": squad_score,
            "fixture_outlook": fixture_outlook,
            "fixture_outlook_score": round(avg_fixture, 2),
            "weaknesses": weaknesses,
            "strengths": strengths,
        })
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transfer-gain/<int:user_id>")
def api_transfer_gain(user_id):
    """Transfer gain vs hit cost chart data for a user's squad."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=5)
        compute_rotation_risk(players)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        if user_id == DEMO_USER_ID:
            squad_players, bank = _demo_user_team()
        else:
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
                    "out_id": out_p.id,
                    "out_name": out_p.name,
                    "out_team": teams_dict.get(out_p.team, "???"),
                    "out_cost": out_p.cost,
                    "out_ep": round(out_p.ep_next, 2),
                    "in_id": candidate.id,
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
                    "reasoning": _build_reasoning(out_p, candidate),
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

        if user_id == DEMO_USER_ID:
            squad_players, bank = _demo_user_team()
        else:
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
        if user_id == DEMO_USER_ID:
            squad_players, bank = _demo_user_team()
        else:
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
    """Enhanced Monte Carlo rank sim with per-player modelling, captain comparison, and scenarios."""
    try:
        num_sims = min(int(request.args.get("sims", 1000)), 5000)
        horizon = min(int(request.args.get("horizon", 5)), 10)
        captain_id_param = request.args.get("captain_id")
        alt_captain_id_param = request.args.get("alt_captain_id")
        hits = int(request.args.get("hits", 0))

        players, teams, gameweeks, fixtures = _get_cached_data()
        score_players(players, fixtures, gameweeks, teams, lookahead=1)
        player_map = {p.id: p for p in players}
        teams_dict = {tid: t.short_name for tid, t in teams.items()}

        if user_id == DEMO_USER_ID:
            squad_players, bank = _demo_user_team()
            current_rank = 247832
            current_points = 1453
        else:
            squad_players, bank = load_user_team(user_id, players, gameweeks)
            with httpx.Client(timeout=30) as client:
                entry = fetch_user_entry(client, user_id)
            current_rank = entry.get("summary_overall_rank", 500000)
            current_points = entry.get("summary_overall_points", 0)

        # Build starting XI
        gks = [p for p in squad_players if p.position == 1]
        outfield = sorted([p for p in squad_players if p.position != 1], key=lambda p: p.ep_next, reverse=True)
        starting_xi = gks[:1] + outfield[:10]

        # Determine captain
        captain_id = int(captain_id_param) if captain_id_param else None
        if captain_id is None and starting_xi:
            captain_id = max(starting_xi, key=lambda p: p.ep_next).id

        # Per-player variance model (position-dependent)
        pos_variance = {1: 0.25, 2: 0.30, 3: 0.35, 4: 0.40}

        def simulate_gw(xi, cap_id):
            total = 0
            for p in xi:
                variance = p.ep_next * pos_variance.get(p.position, 0.35)
                pts = max(0, random.gauss(p.ep_next, variance))
                total += pts * (2 if p.id == cap_id else 1)
            return total

        # EO-adjusted average manager score
        eo_players = sorted(
            [p for p in players if p.position in (2, 3, 4) and p.selected_by_percent > 0],
            key=lambda p: p.selected_by_percent, reverse=True
        )
        avg_gw_score = max(sum(p.ep_next * (p.selected_by_percent / 100) for p in eo_players[:50]), 40.0)

        # Rank sensitivity by rank band
        def sensitivity(rank):
            if rank < 100000:
                return 80
            if rank < 500000:
                return 150
            if rank < 1000000:
                return 200
            return 300

        sens = sensitivity(current_rank)

        def run_sim(cap_id, n):
            results = []
            for _ in range(n):
                sim_total = current_points - (hits * 4)
                for _ in range(horizon):
                    sim_total += simulate_gw(starting_xi, cap_id)
                results.append(sim_total)
            results.sort(reverse=True)
            return results

        final_points = run_sim(captain_id, num_sims)

        p10 = final_points[int(num_sims * 0.9)]
        p25 = final_points[int(num_sims * 0.75)]
        p50 = final_points[int(num_sims * 0.5)]
        p75 = final_points[int(num_sims * 0.25)]
        p90 = final_points[int(num_sims * 0.1)]

        avg_total_gain = sum(final_points) / len(final_points) - current_points
        avg_others_gain = avg_gw_score * horizon
        rank_delta = -(avg_total_gain - avg_others_gain) / sens * 100000

        def rank_from_pts(pts_total):
            diff = pts_total - current_points - avg_others_gain
            return max(1, round(current_rank - diff / sens * 100000))

        scenarios = {
            "pessimistic": {"points": round(p10, 1), "rank": rank_from_pts(p10)},
            "expected": {"points": round(p50, 1), "rank": rank_from_pts(p50)},
            "optimistic": {"points": round(p90, 1), "rank": rank_from_pts(p90)},
        }

        # Histogram
        min_pts, max_pts = min(final_points), max(final_points)
        bucket_size = max(1, (max_pts - min_pts) / 20)
        buckets = []
        for i in range(20):
            lo = min_pts + i * bucket_size
            hi = lo + bucket_size
            count = sum(1 for pt in final_points if lo <= pt < hi)
            buckets.append({"min": round(lo, 1), "max": round(hi, 1), "count": count})

        # Player contributions
        player_contributions = []
        for p in starting_xi:
            multiplier = 2 if p.id == captain_id else 1
            player_contributions.append({
                "id": p.id,
                "name": p.name,
                "team_name": teams_dict.get(p.team, "???"),
                "position_name": p.position_name,
                "ep_next": round(p.ep_next, 2),
                "ep_contribution": round(p.ep_next * multiplier, 2),
                "is_captain": p.id == captain_id,
                "eo": round(p.selected_by_percent, 1),
            })
        player_contributions.sort(key=lambda x: x["ep_contribution"], reverse=True)

        # Decision comparison
        comparison = None
        if alt_captain_id_param:
            alt_id = int(alt_captain_id_param)
            alt_points = run_sim(alt_id, num_sims)
            alt_p50 = alt_points[int(num_sims * 0.5)]
            alt_p10 = alt_points[int(num_sims * 0.9)]
            alt_p90 = alt_points[int(num_sims * 0.1)]
            prim_name = player_map.get(captain_id)
            alt_name = player_map.get(alt_id)
            comparison = {
                "option_a": {
                    "captain_id": captain_id,
                    "captain_name": prim_name.name if prim_name else f"ID {captain_id}",
                    "median_points": round(p50, 1),
                    "p10": round(p10, 1),
                    "p90": round(p90, 1),
                    "projected_rank": rank_from_pts(p50),
                },
                "option_b": {
                    "captain_id": alt_id,
                    "captain_name": alt_name.name if alt_name else f"ID {alt_id}",
                    "median_points": round(alt_p50, 1),
                    "p10": round(alt_p10, 1),
                    "p90": round(alt_p90, 1),
                    "projected_rank": rank_from_pts(alt_p50),
                },
                "median_diff": round(p50 - alt_p50, 1),
                "recommendation": "A" if p50 >= alt_p50 else "B",
            }

        response = {
            "current_rank": current_rank,
            "current_points": current_points,
            "squad_ep_per_gw": round(sum(pc["ep_contribution"] for pc in player_contributions), 1),
            "horizon": horizon,
            "simulations": num_sims,
            "hits": hits,
            "captain_id": captain_id,
            "captain_name": player_map[captain_id].name if captain_id and captain_id in player_map else None,
            "projected_points": {
                "p10": round(p10, 1), "p25": round(p25, 1),
                "median": round(p50, 1), "p75": round(p75, 1), "p90": round(p90, 1),
            },
            "scenarios": scenarios,
            "projected_rank_change": round(rank_delta),
            "projected_rank": max(1, round(current_rank + rank_delta)),
            "histogram": buckets,
            "player_contributions": player_contributions,
        }
        if comparison:
            response["comparison"] = comparison

        return jsonify(response)
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


# ── Match Predictions ─────────────────────────────────────────────────────


@app.route("/api/predictions/<int:gw>")
def api_predictions(gw):
    """Match predictions for a gameweek (list view, no scoreline matrix)."""
    try:
        predictions = _get_cached_predictions(gw)
        if not predictions:
            return jsonify({"error": f"No fixtures found for GW {gw}"}), 404

        matches = []
        for pred in predictions:
            d = pred.to_dict()
            d.pop("scoreline_matrix", None)
            matches.append(d)

        return jsonify({"gameweek": gw, "matches": matches})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/<int:gw>/<int:match_id>")
def api_prediction_detail(gw, match_id):
    """Detailed prediction for a single match (includes scoreline matrix)."""
    try:
        predictions = _get_cached_predictions(gw)
        pred = next((p for p in predictions if p.fixture_id == match_id), None)
        if not pred:
            return jsonify({"error": "Match not found"}), 404

        return jsonify({"gameweek": gw, "match": pred.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Specialty Markets (Phase 2) ──────────────────────────────────────────


@app.route("/api/match/<int:gw>/<int:match_id>/cards")
def api_match_cards(gw, match_id):
    """Card prediction + referee analysis + player card risk."""
    try:
        predictions = _get_cached_predictions(gw)
        pred = next((p for p in predictions if p.fixture_id == match_id), None)
        if not pred:
            return jsonify({"error": "Match not found"}), 404

        players, teams, gameweeks, fixtures = _get_cached_data()
        team_stats = _get_cached_team_stats()

        # Team stats are keyed by FPL team ID
        home_ts = team_stats.get(pred.home_team_id)
        away_ts = team_stats.get(pred.away_team_id)
        home_ts_d = home_ts.to_dict() if home_ts else None
        away_ts_d = away_ts.to_dict() if away_ts else None

        # Referee from Football-Data.org
        fd_matches, ref_stats = _get_cached_referee_data()
        ref_name = None
        home_fd_id = _resolve_fd_team_id(pred.home_team_id)
        away_fd_id = _resolve_fd_team_id(pred.away_team_id)
        if home_fd_id and away_fd_id and fd_matches:
            ref_name = get_match_referee(fd_matches, home_fd_id, away_fd_id)
        ref_s = ref_stats.get(ref_name).to_dict() if ref_name and ref_name in ref_stats else None

        card_pred = predict_cards(home_ts_d, away_ts_d, ref_s)

        # Player card risks
        finished_gws = sum(1 for g in gameweeks if g.finished)
        home_risks = get_player_card_risks(players, pred.home_team_id, finished_gws)
        away_risks = get_player_card_risks(players, pred.away_team_id, finished_gws)

        return jsonify({
            "has_external_data": True,
            "prediction": card_pred,
            "home_team": pred.home_team_short,
            "away_team": pred.away_team_short,
            "home_player_risks": home_risks,
            "away_player_risks": away_risks,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/match/<int:gw>/<int:match_id>/corners")
def api_match_corners(gw, match_id):
    """Corner prediction + team comparison."""
    try:
        predictions = _get_cached_predictions(gw)
        pred = next((p for p in predictions if p.fixture_id == match_id), None)
        if not pred:
            return jsonify({"error": "Match not found"}), 404

        team_stats = _get_cached_team_stats()
        home_ts = team_stats.get(pred.home_team_id)
        away_ts = team_stats.get(pred.away_team_id)

        corner_pred = predict_corners(
            home_ts.to_dict() if home_ts else None,
            away_ts.to_dict() if away_ts else None,
        )

        return jsonify({
            "has_external_data": True,
            "prediction": corner_pred,
            "home_team": pred.home_team_short,
            "away_team": pred.away_team_short,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/match/<int:gw>/<int:match_id>/shots")
def api_match_shots(gw, match_id):
    """Shot prediction + goalscorer probabilities."""
    try:
        predictions = _get_cached_predictions(gw)
        pred = next((p for p in predictions if p.fixture_id == match_id), None)
        if not pred:
            return jsonify({"error": "Match not found"}), 404

        players, _, _, _ = _get_cached_data()
        team_stats = _get_cached_team_stats()
        home_ts = team_stats.get(pred.home_team_id)
        away_ts = team_stats.get(pred.away_team_id)

        home_players = [p for p in players if p.team == pred.home_team_id]
        away_players = [p for p in players if p.team == pred.away_team_id]

        shot_pred = predict_shots(
            home_ts.to_dict() if home_ts else None,
            away_ts.to_dict() if away_ts else None,
            home_players, away_players,
            pred.home_xg, pred.away_xg,
        )

        return jsonify({
            "has_external_data": True,
            "prediction": shot_pred,
            "home_team": pred.home_team_short,
            "away_team": pred.away_team_short,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/match/<int:gw>/<int:match_id>/other")
def api_match_other(gw, match_id):
    """Other markets — penalty, offsides, free kicks, throw-ins,
    first goal timing, first corner/card, HT/FT, per-half stats."""
    try:
        predictions = _get_cached_predictions(gw)
        pred = next((p for p in predictions if p.fixture_id == match_id), None)
        if not pred:
            return jsonify({"error": "Match not found"}), 404

        team_stats = _get_cached_team_stats()
        home_ts = team_stats.get(pred.home_team_id)
        away_ts = team_stats.get(pred.away_team_id)
        home_ts_d = home_ts.to_dict() if home_ts else None
        away_ts_d = away_ts.to_dict() if away_ts else None

        # Referee data (for penalty prediction)
        fd_matches, ref_stats = _get_cached_referee_data()
        ref_name = None
        home_fd_id = _resolve_fd_team_id(pred.home_team_id)
        away_fd_id = _resolve_fd_team_id(pred.away_team_id)
        if home_fd_id and away_fd_id and fd_matches:
            ref_name = get_match_referee(fd_matches, home_fd_id, away_fd_id)
        ref_s = ref_stats.get(ref_name).to_dict() if ref_name and ref_name in ref_stats else None

        other_pred = predict_other_markets(
            home_ts_d, away_ts_d, ref_s,
            pred.home_xg, pred.away_xg,
        )

        return jsonify({
            "has_external_data": True,
            "prediction": other_pred,
            "home_team": pred.home_team_short,
            "away_team": pred.away_team_short,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/tracker")
def api_prediction_tracker():
    """Historical prediction accuracy data."""
    try:
        accuracy = _compute_tracker_accuracy()
        return jsonify(accuracy)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# Set Pieces
# ------------------------------------------------------------------ #
@app.route("/api/set-pieces")
def api_set_pieces():
    """Return players with set-piece duties."""
    try:
        players, teams, _, _ = _get_cached_data()
        team_filter = request.args.get("team", type=int)
        result = []
        for p in players:
            if p.penalties_order == 0 and p.direct_freekicks_order == 0 and p.corners_order == 0:
                continue
            if team_filter and p.team != team_filter:
                continue
            result.append({
                "id": p.id,
                "name": p.name,
                "team": p.team,
                "team_name": teams[p.team].short_name if p.team in teams else "???",
                "position": p.position_name,
                "penalties_order": p.penalties_order,
                "freekicks_order": p.direct_freekicks_order,
                "corners_order": p.corners_order,
                "total_points": p.total_points,
                "goals": p.goals,
                "assists": p.assists,
                "xG": round(p.xG, 2),
                "cost": p.cost,
            })
        result.sort(key=lambda x: (x["penalties_order"] or 99, x["freekicks_order"] or 99, x["corners_order"] or 99))
        return jsonify({"players": result, "teams": {tid: t.to_dict() for tid, t in teams.items()}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# Player History (per-GW form data)
# ------------------------------------------------------------------ #
@app.route("/api/player/<int:player_id>/history")
def api_player_history(player_id):
    """Return per-gameweek history for a single player."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"https://fantasy.premierleague.com/api/element-summary/{player_id}/")
            resp.raise_for_status()
            data = resp.json()
        history = []
        for h in data.get("history", []):
            history.append({
                "gw": h.get("round"),
                "points": h.get("total_points", 0),
                "minutes": h.get("minutes", 0),
                "goals": h.get("goals_scored", 0),
                "assists": h.get("assists", 0),
                "xG": float(h.get("expected_goals", 0)),
                "xA": float(h.get("expected_assists", 0)),
                "xGI": float(h.get("expected_goal_involvements", 0)),
                "bps": h.get("bps", 0),
                "bonus": h.get("bonus", 0),
                "cost": h.get("value", 0) / 10.0,
            })
        return jsonify({"player_id": player_id, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# Lineup Predictor
# ------------------------------------------------------------------ #
@app.route("/api/lineup-predictor")
def api_lineup_predictor():
    """Return starting probability for all players, optionally filtered by team."""
    try:
        players, teams, gameweeks, fixtures = _get_cached_data()
        compute_rotation_risk(players)
        team_filter = request.args.get("team", type=int)

        finished_gws = max(sum(1 for gw in gameweeks if gw.finished), 1)

        result = []
        for p in players:
            if team_filter and p.team != team_filter:
                continue

            # Base: starts / finished_gws
            starts_ratio = min(p.starts / finished_gws, 1.0) if finished_gws > 0 else 0.5
            prob = starts_ratio

            # Minutes factor
            if p.starts > 0:
                avg_mins = p.minutes / p.starts
                if avg_mins > 85:
                    prob += 0.10
                elif avg_mins > 80:
                    prob += 0.05

            # Availability
            if p.chance_of_playing is not None:
                if p.chance_of_playing == 0:
                    prob -= 0.90
                elif p.chance_of_playing <= 25:
                    prob -= 0.50
                elif p.chance_of_playing <= 50:
                    prob -= 0.30
                elif p.chance_of_playing <= 75:
                    prob -= 0.10

            # Rotation risk
            prob -= p.rotation_risk * 0.15

            # Injury news
            if p.news:
                nl = p.news.lower()
                injury_kw = ("injured", "injury", "hamstring", "knee", "ankle", "groin",
                             "muscle", "surgery", "broken", "fracture", "illness",
                             "suspended", "ban")
                if any(kw in nl for kw in injury_kw):
                    prob -= 0.20

            prob = max(0.0, min(0.99, prob))

            result.append({
                "id": p.id,
                "name": p.name,
                "team": p.team,
                "team_name": teams[p.team].short_name if p.team in teams else "???",
                "position": p.position,
                "position_name": p.position_name,
                "cost": p.cost,
                "start_prob": round(prob * 100, 1),
                "starts": p.starts,
                "minutes": p.minutes,
                "chance_of_playing": p.chance_of_playing,
                "news": p.news or "",
                "rotation_risk": round(p.rotation_risk, 2),
            })

        result.sort(key=lambda x: -x["start_prob"])

        # Build predicted XI per team
        predicted_xi = {}
        for tid, team in teams.items():
            team_players = sorted(
                [r for r in result if r["team"] == tid],
                key=lambda x: -x["start_prob"],
            )
            # Pick best XI: 1 GK, then top outfield
            gks = [p for p in team_players if p["position"] == 1]
            outfield = [p for p in team_players if p["position"] != 1]
            xi = gks[:1] + outfield[:10]
            predicted_xi[tid] = xi

        return jsonify({
            "players": result,
            "predicted_xi": predicted_xi,
            "teams": {tid: t.to_dict() for tid, t in teams.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# User Season History
# ------------------------------------------------------------------ #
@app.route("/api/history/<int:user_id>")
def api_user_history(user_id):
    """Return full season GW-by-GW history for a user."""
    try:
        with httpx.Client(timeout=30) as client:
            data = fetch_entry_history(client, user_id)
        current = data.get("current", [])
        chips = data.get("chips", [])
        chip_map = {c["event"]: c["name"] for c in chips}

        history = []
        for gw in current:
            history.append({
                "gw": gw["event"],
                "points": gw["points"],
                "total_points": gw["total_points"],
                "overall_rank": gw["overall_rank"],
                "rank": gw.get("rank"),
                "transfers": gw.get("event_transfers", 0),
                "hits": gw.get("event_transfers_cost", 0),
                "bench_points": gw.get("points_on_bench", 0),
                "chip": chip_map.get(gw["event"]),
            })

        points_list = [h["points"] for h in history]
        avg_pts = round(sum(points_list) / len(points_list), 1) if points_list else 0
        best_gw = max(history, key=lambda h: h["points"]) if history else None
        worst_gw = min(history, key=lambda h: h["points"]) if history else None
        total_hits = sum(h["hits"] for h in history)

        return jsonify({
            "history": history,
            "summary": {
                "avg_points": avg_pts,
                "best_gw": best_gw["gw"] if best_gw else None,
                "best_pts": best_gw["points"] if best_gw else 0,
                "worst_gw": worst_gw["gw"] if worst_gw else None,
                "worst_pts": worst_gw["points"] if worst_gw else 0,
                "best_rank": min((h["overall_rank"] for h in history), default=0),
                "current_rank": history[-1]["overall_rank"] if history else 0,
                "total_hits": total_hits,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
# Mini-League Differentials
# ------------------------------------------------------------------ #
@app.route("/api/league-differentials/<int:league_id>")
def api_league_differentials(league_id):
    """Show differential/template/threat players vs mini-league rivals."""
    try:
        user_id = request.args.get("user_id", type=int)
        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        players, teams, gameweeks, _ = _get_cached_data()
        player_map = {p.id: p for p in players}

        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        if current_gw is None:
            current_gw = next((gw for gw in gameweeks if gw.is_next), None)
        if current_gw is None:
            return jsonify({"error": "Cannot determine gameweek"}), 500

        with httpx.Client(timeout=30) as client:
            standings = fetch_league_standings(client, league_id)
            entries = standings.get("standings", {}).get("results", [])

            # Get user's squad
            user_picks_data = fetch_user_picks_full(client, user_id, current_gw.id)
            user_pids = {pick["element"] for pick in user_picks_data.get("picks", [])}

            # Get rival squads (top 10, excluding user)
            rival_ownership: dict[int, int] = {}
            rival_count = 0
            for entry in entries[:11]:
                rid = entry["entry"]
                if rid == user_id:
                    continue
                rival_count += 1
                try:
                    rival_data = fetch_user_picks_full(client, rid, current_gw.id)
                    for pick in rival_data.get("picks", []):
                        pid = pick["element"]
                        rival_ownership[pid] = rival_ownership.get(pid, 0) + 1
                except Exception:
                    continue
                if rival_count >= 10:
                    break

        if rival_count == 0:
            return jsonify({"error": "No rivals found"}), 404

        # Categorize user's players
        differentials = []
        template = []
        for pid in user_pids:
            p = player_map.get(pid)
            if not p:
                continue
            rival_own = rival_ownership.get(pid, 0)
            info = {
                "id": p.id, "name": p.name, "team": teams[p.team].short_name if p.team in teams else "???",
                "position": p.position_name, "cost": p.cost,
                "rival_ownership": rival_own, "rival_pct": round(rival_own / rival_count * 100, 1),
                "total_points": p.total_points, "ep_next": p.ep_next,
            }
            if rival_own <= 2:
                differentials.append(info)
            if rival_own >= 5:
                template.append(info)

        # Threats: players owned by 3+ rivals but not by user
        threats = []
        for pid, count in sorted(rival_ownership.items(), key=lambda x: -x[1]):
            if pid in user_pids or count < 3:
                continue
            p = player_map.get(pid)
            if not p:
                continue
            threats.append({
                "id": p.id, "name": p.name, "team": teams[p.team].short_name if p.team in teams else "???",
                "position": p.position_name, "cost": p.cost,
                "rival_ownership": count, "rival_pct": round(count / rival_count * 100, 1),
                "total_points": p.total_points, "ep_next": p.ep_next,
            })
            if len(threats) >= 15:
                break

        differentials.sort(key=lambda x: x["rival_ownership"])
        template.sort(key=lambda x: -x["rival_ownership"])

        return jsonify({
            "differentials": differentials,
            "template": template,
            "threats": threats,
            "rival_count": rival_count,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
