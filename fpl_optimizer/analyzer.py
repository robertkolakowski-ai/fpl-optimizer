from __future__ import annotations

from .models import Fixture, Gameweek, Player

# Position-specific weights: (form, ppg, xG, xA, clean_sheets, bonus, ict, fixture_ease)
POSITION_WEIGHTS = {
    1: (0.10, 0.20, 0.00, 0.00, 0.35, 0.10, 0.05, 0.20),  # GK
    2: (0.10, 0.20, 0.05, 0.05, 0.25, 0.10, 0.05, 0.20),  # DEF
    3: (0.15, 0.15, 0.20, 0.15, 0.00, 0.10, 0.10, 0.15),  # MID
    4: (0.15, 0.15, 0.30, 0.10, 0.00, 0.10, 0.10, 0.10),  # FWD
}


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def compute_fixture_difficulty(
    players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    lookahead: int = 5,
) -> None:
    current_gw = next((gw for gw in gameweeks if gw.is_next), None)
    if current_gw is None:
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
    if current_gw is None:
        for p in players:
            p.fixture_difficulty = 0.5
        return

    gw_start = current_gw.id
    gw_end = gw_start + lookahead

    # Build team → average difficulty over the window
    # Lower difficulty = easier fixtures = higher ease score
    team_difficulty: dict[int, list[int]] = {}
    for f in fixtures:
        if f.gameweek is None or f.gameweek < gw_start or f.gameweek >= gw_end:
            continue
        team_difficulty.setdefault(f.home_team, []).append(f.home_difficulty)
        team_difficulty.setdefault(f.away_team, []).append(f.away_difficulty)

    team_avg: dict[int, float] = {}
    for team_id, diffs in team_difficulty.items():
        team_avg[team_id] = sum(diffs) / len(diffs) if diffs else 3.0

    # Convert difficulty (1-5 scale, lower=easier) to ease (0-1, higher=easier)
    all_avgs = list(team_avg.values()) or [3.0]
    lo, hi = min(all_avgs), max(all_avgs)
    for p in players:
        avg = team_avg.get(p.team, 3.0)
        if hi == lo:
            p.fixture_difficulty = 0.5
        else:
            p.fixture_difficulty = 1.0 - (avg - lo) / (hi - lo)


def score_players(
    players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    lookahead: int = 5,
) -> None:
    compute_fixture_difficulty(players, fixtures, gameweeks, lookahead)

    for pos in (1, 2, 3, 4):
        group = [p for p in players if p.position == pos]
        if not group:
            continue

        weights = POSITION_WEIGHTS[pos]

        raw_stats = [
            [p.form for p in group],
            [p.points_per_game for p in group],
            [p.xG for p in group],
            [p.xA for p in group],
            [float(p.clean_sheets) for p in group],
            [float(p.bonus) for p in group],
            [p.ict_index for p in group],
            [p.fixture_difficulty for p in group],
        ]

        normalized = [_normalize(stat) for stat in raw_stats]

        for i, p in enumerate(group):
            p.composite_score = sum(
                w * normalized[j][i] for j, w in enumerate(weights)
            )
