from __future__ import annotations

from fpl_optimizer.analyzer import (
    POSITION_WEIGHTS,
    CL_TEAMS,
    compute_fixture_difficulty,
    compute_rotation_risk,
    score_players,
)
from fpl_optimizer.models import Player


def test_score_players_produces_scores(mock_players, mock_fixtures, mock_gameweeks):
    score_players(mock_players, mock_fixtures, mock_gameweeks)
    for p in mock_players:
        assert p.composite_score > 0, f"{p.name} has composite_score={p.composite_score}"


def test_position_weights_sum_to_one():
    for pos, weights in POSITION_WEIGHTS.items():
        total = sum(weights)
        assert abs(total - 1.0) < 1e-6, f"Position {pos} weights sum to {total}"


def test_fixture_difficulty_range(mock_players, mock_fixtures, mock_gameweeks):
    compute_fixture_difficulty(mock_players, mock_fixtures, mock_gameweeks)
    for p in mock_players:
        assert 0.0 <= p.fixture_difficulty <= 1.0, (
            f"{p.name} fixture_difficulty={p.fixture_difficulty} out of range"
        )


def test_rotation_risk_cl_teams_higher(mock_players):
    # Set all players to same stats, but vary team
    non_cl = Player(id=100, name="NonCL", team=99, position=3, cost=6.0,
                    minutes=900, starts=10, chance_of_playing=100)
    cl = Player(id=101, name="CL", team=list(CL_TEAMS)[0], position=3, cost=6.0,
                minutes=900, starts=10, chance_of_playing=100)
    compute_rotation_risk([non_cl, cl])
    assert cl.rotation_risk > non_cl.rotation_risk


def test_gk_weights_prioritize_clean_sheets():
    gk_weights = POSITION_WEIGHTS[1]
    # Index 4 = clean_sheets weight for GK
    cs_weight = gk_weights[4]
    # Clean sheets should be the highest weight for GKs
    assert cs_weight == max(gk_weights), (
        f"GK clean_sheet weight {cs_weight} is not the max among {gk_weights}"
    )


def test_fwd_weights_prioritize_xg():
    fwd_weights = POSITION_WEIGHTS[4]
    # Index 2 = xG weight
    xg_weight = fwd_weights[2]
    assert xg_weight == max(fwd_weights), (
        f"FWD xG weight {xg_weight} is not the max among {fwd_weights}"
    )
