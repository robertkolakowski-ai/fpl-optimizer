from __future__ import annotations

import math

from fpl_optimizer.predictions import (
    _build_scoreline_matrix,
    _derive_probabilities,
    _poisson_pmf,
)


def test_poisson_pmf_basic():
    # P(X=0 | lam=1) = e^{-1} ≈ 0.3679
    assert abs(_poisson_pmf(0, 1.0) - math.exp(-1)) < 1e-10


def test_poisson_pmf_zero_lambda():
    assert _poisson_pmf(0, 0.0) == 1.0
    assert _poisson_pmf(1, 0.0) == 0.0
    assert _poisson_pmf(5, 0.0) == 0.0


def test_poisson_pmf_sums_approximately_to_one():
    lam = 2.5
    total = sum(_poisson_pmf(k, lam) for k in range(20))
    assert abs(total - 1.0) < 1e-6


def test_scoreline_matrix_sums_to_approximately_one():
    matrix = _build_scoreline_matrix(1.5, 1.2)
    total = sum(cell for row in matrix for cell in row)
    # 6x6 truncation loses some mass, but should be >0.95
    assert total > 0.95
    assert total <= 1.0 + 1e-6


def test_derive_probabilities_sum():
    matrix = _build_scoreline_matrix(1.5, 1.0)
    probs = _derive_probabilities(matrix)
    total = probs["home_win"] + probs["draw"] + probs["away_win"]
    assert abs(total - 1.0) < 1e-4


def test_over_under_monotonic():
    matrix = _build_scoreline_matrix(1.5, 1.2)
    probs = _derive_probabilities(matrix)
    assert probs["over_05"] >= probs["over_15"]
    assert probs["over_15"] >= probs["over_25"]
    assert probs["over_25"] >= probs["over_35"]
    assert probs["over_35"] >= probs["over_45"]


def test_btts_range():
    matrix = _build_scoreline_matrix(1.5, 1.2)
    probs = _derive_probabilities(matrix)
    assert 0 < probs["btts_yes"] < 1


def test_high_xg_favors_home_win():
    matrix = _build_scoreline_matrix(3.0, 0.5)
    probs = _derive_probabilities(matrix)
    assert probs["home_win"] > probs["draw"]
    assert probs["home_win"] > probs["away_win"]


def test_symmetric_xg_favors_draw():
    matrix = _build_scoreline_matrix(1.0, 1.0)
    probs = _derive_probabilities(matrix)
    # With equal xG, draw should be significant (but not necessarily largest)
    assert probs["draw"] > 0.2
    # Home/away should be approximately equal
    assert abs(probs["home_win"] - probs["away_win"]) < 0.05
