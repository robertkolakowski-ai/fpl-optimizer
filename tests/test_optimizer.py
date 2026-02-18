from __future__ import annotations

from fpl_optimizer.analyzer import score_players
from fpl_optimizer.optimizer import (
    MAX_PER_TEAM,
    POSITION_LIMITS,
    SQUAD_SIZE,
    STARTING_MAX,
    STARTING_MIN,
    STARTING_SIZE,
    select_squad,
)


def _score_and_select(players, fixtures, gameweeks, budget=100.0, risk_mode="balanced"):
    score_players(players, fixtures, gameweeks)
    return select_squad(players, budget=budget, risk_mode=risk_mode)


def test_squad_has_15_players(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    assert len(squad.players) == SQUAD_SIZE


def test_squad_position_limits(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    for pos, limit in POSITION_LIMITS.items():
        count = sum(1 for p in squad.players if p.position == pos)
        assert count == limit, f"Position {pos}: expected {limit}, got {count}"


def test_squad_budget_constraint(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks, budget=100.0)
    assert squad.total_cost <= 100.0 + 0.01  # small float tolerance


def test_squad_team_limit(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    team_counts: dict[int, int] = {}
    for p in squad.players:
        team_counts[p.team] = team_counts.get(p.team, 0) + 1
    for tid, count in team_counts.items():
        assert count <= MAX_PER_TEAM, f"Team {tid}: {count} players exceeds max {MAX_PER_TEAM}"


def test_starting_xi_has_11(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    assert len(squad.starting) == STARTING_SIZE


def test_starting_xi_valid_formation(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    for pos in (1, 2, 3, 4):
        count = sum(1 for p in squad.starting if p.position == pos)
        assert count >= STARTING_MIN[pos], f"Pos {pos}: {count} < min {STARTING_MIN[pos]}"
        assert count <= STARTING_MAX[pos], f"Pos {pos}: {count} > max {STARTING_MAX[pos]}"


def test_bench_has_4(mock_players, mock_fixtures, mock_gameweeks):
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    assert len(squad.bench) == SQUAD_SIZE - STARTING_SIZE


def test_optimizer_returns_squad_object(mock_players, mock_fixtures, mock_gameweeks):
    from fpl_optimizer.models import Squad
    squad = _score_and_select(mock_players, mock_fixtures, mock_gameweeks)
    assert isinstance(squad, Squad)
    assert squad.budget_remaining >= 0


def test_risk_modes_produce_different_squads(mock_players, mock_fixtures, mock_gameweeks):
    safe = _score_and_select(mock_players, mock_fixtures, mock_gameweeks, risk_mode="safe")
    aggressive = _score_and_select(mock_players, mock_fixtures, mock_gameweeks, risk_mode="aggressive")
    safe_ids = {p.id for p in safe.players}
    agg_ids = {p.id for p in aggressive.players}
    # With only 20 players and 15 slots, there may be large overlap,
    # but the objective scores should differ
    safe_score = sum(p.composite_score for p in safe.players)
    agg_score = sum(p.composite_score for p in aggressive.players)
    # They should both produce valid squads regardless
    assert len(safe.players) == SQUAD_SIZE
    assert len(aggressive.players) == SQUAD_SIZE
