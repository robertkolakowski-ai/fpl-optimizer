from __future__ import annotations

import httpx

from .models import Fixture, Gameweek, Player, Team

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap(client: httpx.Client) -> dict:
    resp = client.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures(client: httpx.Client) -> list[dict]:
    resp = client.get(f"{BASE_URL}/fixtures/")
    resp.raise_for_status()
    return resp.json()


def parse_teams(data: dict) -> dict[int, Team]:
    teams = {}
    for t in data["teams"]:
        teams[t["id"]] = Team(
            id=t["id"],
            name=t["name"],
            short_name=t["short_name"],
            code=t.get("code", 0),
        )
    return teams


def parse_gameweeks(data: dict) -> list[Gameweek]:
    return [
        Gameweek(
            id=gw["id"],
            name=gw["name"],
            finished=gw["finished"],
            is_current=gw["is_current"],
            is_next=gw["is_next"],
        )
        for gw in data["events"]
    ]


def parse_players(data: dict) -> list[Player]:
    players = []
    for e in data["elements"]:
        # Skip unavailable players (status 'u' = unavailable, 'i' = injured, 's' = suspended)
        if e.get("status") in ("u",):
            continue
        # Skip players with 0 minutes
        if e.get("minutes", 0) == 0:
            continue

        p = Player(
            id=e["id"],
            name=e.get("web_name", "Unknown"),
            team=e["team"],
            position=e["element_type"],
            cost=e["now_cost"] / 10.0,
            total_points=e.get("total_points", 0),
            minutes=e.get("minutes", 0),
            goals=e.get("goals_scored", 0),
            assists=e.get("assists", 0),
            clean_sheets=e.get("clean_sheets", 0),
            goals_conceded=e.get("goals_conceded", 0),
            bonus=e.get("bonus", 0),
            form=float(e.get("form", 0)),
            points_per_game=float(e.get("points_per_game", 0)),
            xG=float(e.get("expected_goals", 0)),
            xA=float(e.get("expected_assists", 0)),
            ict_index=float(e.get("ict_index", 0)),
            photo=e.get("photo", "").replace(".jpg", ""),
            selected_by_percent=float(e.get("selected_by_percent", 0)),
            transfers_in_event=e.get("transfers_in_event", 0),
            transfers_out_event=e.get("transfers_out_event", 0),
            cost_change_event=e.get("cost_change_event", 0),
            news=e.get("news", ""),
            chance_of_playing=e.get("chance_of_playing_next_round"),
            # Expected points
            ep_next=float(e.get("ep_next") or 0),
            ep_this=float(e.get("ep_this") or 0),
            xG_per90=float(e.get("expected_goals_per_90", 0)),
            xA_per90=float(e.get("expected_assists_per_90", 0)),
            xGI_per90=float(e.get("expected_goal_involvements_per_90", 0)),
            xGC_per90=float(e.get("expected_goals_conceded_per_90", 0)),
            # Opta / BPS fields
            influence=float(e.get("influence", 0)),
            creativity=float(e.get("creativity", 0)),
            threat=float(e.get("threat", 0)),
            expected_goal_involvements=float(e.get("expected_goal_involvements", 0)),
            expected_goals_conceded=float(e.get("expected_goals_conceded", 0)),
            bps=e.get("bps", 0),
            saves=e.get("saves", 0),
            penalties_saved=e.get("penalties_saved", 0),
            penalties_missed=e.get("penalties_missed", 0),
            yellow_cards=e.get("yellow_cards", 0),
            red_cards=e.get("red_cards", 0),
            own_goals=e.get("own_goals", 0),
            starts=e.get("starts", 0),
        )
        players.append(p)
    return players


def parse_fixtures(raw: list[dict]) -> list[Fixture]:
    return [
        Fixture(
            id=f["id"],
            gameweek=f.get("event"),
            home_team=f["team_h"],
            away_team=f["team_a"],
            home_difficulty=f.get("team_h_difficulty", 3),
            away_difficulty=f.get("team_a_difficulty", 3),
            finished=f.get("finished", False),
            home_score=f.get("team_h_score"),
            away_score=f.get("team_a_score"),
            kickoff_time=f.get("kickoff_time"),
            started=f.get("started"),
        )
        for f in raw
    ]


def fetch_user_picks(client: httpx.Client, user_id: int, gameweek: int) -> list[dict]:
    resp = client.get(f"{BASE_URL}/entry/{user_id}/event/{gameweek}/picks/")
    resp.raise_for_status()
    return resp.json()["picks"]


def fetch_user_entry(client: httpx.Client, user_id: int) -> dict:
    resp = client.get(f"{BASE_URL}/entry/{user_id}/")
    resp.raise_for_status()
    return resp.json()


def fetch_league_standings(client: httpx.Client, league_id: int) -> dict:
    resp = client.get(f"{BASE_URL}/leagues-classic/{league_id}/standings/?page_standings=1")
    resp.raise_for_status()
    return resp.json()


def fetch_live_gameweek(client: httpx.Client, gameweek: int) -> dict:
    resp = client.get(f"{BASE_URL}/event/{gameweek}/live/")
    resp.raise_for_status()
    return resp.json()


def fetch_user_picks_full(client: httpx.Client, user_id: int, gameweek: int) -> dict:
    """Returns the full picks response including automatic_subs, entry_history, etc."""
    resp = client.get(f"{BASE_URL}/entry/{user_id}/event/{gameweek}/picks/")
    resp.raise_for_status()
    return resp.json()


def fetch_entry_history(client: httpx.Client, user_id: int) -> dict:
    resp = client.get(f"{BASE_URL}/entry/{user_id}/history/")
    resp.raise_for_status()
    return resp.json()


def load_user_team(
    user_id: int,
    all_players: list[Player],
    gameweeks: list[Gameweek],
) -> tuple[list[Player], float]:
    """Load a user's current FPL team. Returns (squad_players, selling_prices_total)."""
    current_gw = next((gw for gw in gameweeks if gw.is_current), None)
    if current_gw is None:
        current_gw = next((gw for gw in gameweeks if gw.is_next), None)
    if current_gw is None:
        raise ValueError("Cannot determine current gameweek")

    player_map = {p.id: p for p in all_players}

    with httpx.Client(timeout=30) as client:
        entry = fetch_user_entry(client, user_id)
        picks = fetch_user_picks(client, user_id, current_gw.id)

    bank = entry.get("last_deadline_bank", 0) / 10.0  # Convert to millions
    team_value = entry.get("last_deadline_value", 0) / 10.0

    squad_players = []
    for pick in picks:
        pid = pick["element"]
        if pid in player_map:
            squad_players.append(player_map[pid])

    return squad_players, bank


def load_data() -> tuple[list[Player], dict[int, Team], list[Gameweek], list[Fixture], list[dict]]:
    with httpx.Client(timeout=30) as client:
        bootstrap = fetch_bootstrap(client)
        raw_fixtures = fetch_fixtures(client)

    teams = parse_teams(bootstrap)
    gameweeks = parse_gameweeks(bootstrap)
    players = parse_players(bootstrap)
    fixtures = parse_fixtures(raw_fixtures)
    chip_windows = bootstrap.get("chips", [])

    return players, teams, gameweeks, fixtures, chip_windows
