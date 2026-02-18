from __future__ import annotations

import pytest

from fpl_optimizer.models import Fixture, Gameweek, Player, Squad, Team


def _make_player(id: int, name: str, team: int, position: int, cost: float, **kw) -> Player:
    defaults = dict(
        total_points=50, minutes=900, goals=3, assists=2,
        clean_sheets=4, bonus=5, form=4.0, points_per_game=4.0,
        xG=3.0, xA=2.0, ict_index=80.0, selected_by_percent=10.0,
        ep_next=4.0, starts=10, bps=100,
    )
    defaults.update(kw)
    return Player(id=id, name=name, team=team, position=position, cost=cost, **defaults)


@pytest.fixture
def mock_teams() -> dict[int, Team]:
    return {
        1: Team(id=1, name="Arsenal", short_name="ARS", code=3),
        2: Team(id=2, name="Chelsea", short_name="CHE", code=8),
        3: Team(id=3, name="Liverpool", short_name="LIV", code=14),
        4: Team(id=4, name="Man City", short_name="MCI", code=43),
        5: Team(id=5, name="Tottenham", short_name="TOT", code=6),
        6: Team(id=6, name="Aston Villa", short_name="AVL", code=7),
        7: Team(id=7, name="Newcastle", short_name="NEW", code=4),
    }


@pytest.fixture
def mock_gameweeks() -> list[Gameweek]:
    return [
        Gameweek(id=1, name="Gameweek 1", finished=True, is_current=False, is_next=False),
        Gameweek(id=2, name="Gameweek 2", finished=False, is_current=True, is_next=False),
        Gameweek(id=3, name="Gameweek 3", finished=False, is_current=False, is_next=True),
    ]


@pytest.fixture
def mock_fixtures() -> list[Fixture]:
    return [
        # GW 1 (finished)
        Fixture(id=1, gameweek=1, home_team=1, away_team=2, home_difficulty=3, away_difficulty=3,
                finished=True, home_score=2, away_score=1, started=True),
        Fixture(id=2, gameweek=1, home_team=3, away_team=4, home_difficulty=2, away_difficulty=4,
                finished=True, home_score=1, away_score=1, started=True),
        Fixture(id=9, gameweek=1, home_team=5, away_team=6, home_difficulty=3, away_difficulty=3,
                finished=True, home_score=3, away_score=0, started=True),
        # GW 2 (current)
        Fixture(id=3, gameweek=2, home_team=2, away_team=3, home_difficulty=3, away_difficulty=3,
                finished=False, started=False),
        Fixture(id=4, gameweek=2, home_team=4, away_team=1, home_difficulty=4, away_difficulty=2,
                finished=False, started=False),
        Fixture(id=10, gameweek=2, home_team=6, away_team=7, home_difficulty=3, away_difficulty=3,
                finished=False, started=False),
        # GW 3 (next)
        Fixture(id=5, gameweek=3, home_team=1, away_team=3, home_difficulty=3, away_difficulty=3,
                finished=False, started=False),
        Fixture(id=6, gameweek=3, home_team=2, away_team=4, home_difficulty=2, away_difficulty=4,
                finished=False, started=False),
        Fixture(id=11, gameweek=3, home_team=5, away_team=7, home_difficulty=2, away_difficulty=3,
                finished=False, started=False),
        # GW 4-5 (future)
        Fixture(id=7, gameweek=4, home_team=3, away_team=2, home_difficulty=3, away_difficulty=3,
                finished=False, started=False),
        Fixture(id=8, gameweek=5, home_team=1, away_team=4, home_difficulty=2, away_difficulty=4,
                finished=False, started=False),
    ]


@pytest.fixture
def mock_players() -> list[Player]:
    """22 players spread across 7 teams — enough for the LP to solve."""
    return [
        # Goalkeepers (position=1) — need 2
        _make_player(1, "GK1", 1, 1, 5.0, clean_sheets=8, saves=40),
        _make_player(2, "GK2", 2, 1, 4.5, clean_sheets=5, saves=35),
        _make_player(3, "GK3", 3, 1, 5.5, clean_sheets=10, saves=50),
        # Defenders (position=2) — need 5
        _make_player(6, "DEF1", 1, 2, 6.0, clean_sheets=8, goals=2, assists=3),
        _make_player(7, "DEF2", 2, 2, 5.5, clean_sheets=6, goals=1, assists=2),
        _make_player(8, "DEF3", 3, 2, 6.5, clean_sheets=10, goals=3, assists=4),
        _make_player(9, "DEF4", 4, 2, 5.0, clean_sheets=4, goals=0, assists=1),
        _make_player(10, "DEF5", 5, 2, 4.5, clean_sheets=3, goals=0, assists=0),
        _make_player(21, "DEF6", 6, 2, 4.5, clean_sheets=5, goals=1, assists=1),
        _make_player(22, "DEF7", 7, 2, 4.0, clean_sheets=4, goals=0, assists=1),
        # Midfielders (position=3) — need 5
        _make_player(11, "MID1", 1, 3, 8.0, xG=8.0, xA=6.0, goals=8, assists=6),
        _make_player(12, "MID2", 2, 3, 7.5, xG=6.0, xA=5.0, goals=6, assists=5),
        _make_player(13, "MID3", 3, 3, 9.0, xG=10.0, xA=8.0, goals=10, assists=8),
        _make_player(14, "MID4", 4, 3, 6.5, xG=4.0, xA=3.0, goals=4, assists=3),
        _make_player(15, "MID5", 5, 3, 5.5, xG=2.0, xA=2.0, goals=2, assists=2),
        _make_player(23, "MID6", 6, 3, 6.0, xG=5.0, xA=4.0, goals=5, assists=4),
        # Forwards (position=4) — need 3
        _make_player(16, "FWD1", 4, 4, 9.0, xG=12.0, xA=4.0, goals=12, assists=4),
        _make_player(17, "FWD2", 5, 4, 8.5, xG=10.0, xA=3.0, goals=10, assists=3),
        _make_player(18, "FWD3", 6, 4, 7.0, xG=6.0, xA=2.0, goals=6, assists=2),
        _make_player(19, "FWD4", 7, 4, 6.0, xG=4.0, xA=1.0, goals=4, assists=1),
        _make_player(20, "FWD5", 3, 4, 5.5, xG=2.0, xA=1.0, goals=2, assists=1),
    ]
