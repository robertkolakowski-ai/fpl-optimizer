from __future__ import annotations

from fpl_optimizer.analyzer import score_players
from fpl_optimizer.models import Player, Squad
from fpl_optimizer.transfers import suggest_transfers


def _build_squad(players, fixtures, gameweeks):
    """Score players and build a squad from the first 15."""
    score_players(players, fixtures, gameweeks)
    # Pick a valid 15: 2 GK + 5 DEF + 5 MID + 3 FWD
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for p in players:
        by_pos[p.position].append(p)
    squad_players = (
        sorted(by_pos[1], key=lambda p: -p.composite_score)[:2]
        + sorted(by_pos[2], key=lambda p: -p.composite_score)[:5]
        + sorted(by_pos[3], key=lambda p: -p.composite_score)[:5]
        + sorted(by_pos[4], key=lambda p: -p.composite_score)[:3]
    )
    return Squad(players=squad_players, budget_remaining=100.0 - sum(p.cost for p in squad_players))


def test_suggest_transfers_same_position(mock_players, mock_fixtures, mock_gameweeks):
    squad = _build_squad(mock_players, mock_fixtures, mock_gameweeks)
    suggestions = suggest_transfers(squad, mock_players, budget=100.0)
    for s in suggestions:
        assert s.player_out.position == s.player_in.position


def test_suggest_transfers_budget_respected(mock_players, mock_fixtures, mock_gameweeks):
    squad = _build_squad(mock_players, mock_fixtures, mock_gameweeks)
    budget = squad.total_cost + squad.budget_remaining
    suggestions = suggest_transfers(squad, mock_players, budget=budget)
    for s in suggestions:
        new_cost = squad.total_cost - s.player_out.cost + s.player_in.cost
        assert new_cost <= budget + 0.1  # small float tolerance


def test_suggest_transfers_team_limit(mock_players, mock_fixtures, mock_gameweeks):
    squad = _build_squad(mock_players, mock_fixtures, mock_gameweeks)
    team_counts = {}
    for p in squad.players:
        team_counts[p.team] = team_counts.get(p.team, 0) + 1

    suggestions = suggest_transfers(squad, mock_players, budget=100.0)
    for s in suggestions:
        # Simulate swap
        new_counts = dict(team_counts)
        new_counts[s.player_out.team] = new_counts.get(s.player_out.team, 0) - 1
        new_counts[s.player_in.team] = new_counts.get(s.player_in.team, 0) + 1
        assert new_counts[s.player_in.team] <= 3, (
            f"Transfer {s.player_out.name}->{s.player_in.name} breaks team limit"
        )


def test_suggestions_have_positive_gain(mock_players, mock_fixtures, mock_gameweeks):
    squad = _build_squad(mock_players, mock_fixtures, mock_gameweeks)
    suggestions = suggest_transfers(squad, mock_players, budget=100.0)
    for s in suggestions:
        assert s.score_gain > 0
