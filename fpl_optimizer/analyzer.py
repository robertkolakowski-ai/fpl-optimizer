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
    team_form: dict[int, float] | None = None,
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
    team_difficulty: dict[int, list[float]] = {}
    for f in fixtures:
        if f.gameweek is None or f.gameweek < gw_start or f.gameweek >= gw_end:
            continue
        # Home/away adjustment: home is easier (-0.4), away is harder (+0.3)
        home_diff = max(1.0, f.home_difficulty - 0.4)
        away_diff = min(5.0, f.away_difficulty + 0.3)

        # Opponent form adjustment (if available)
        if team_form:
            opp_home_form = team_form.get(f.away_team, 0.5)
            opp_away_form = team_form.get(f.home_team, 0.5)
            home_diff -= (0.5 - opp_home_form) * 0.4  # weak opponent = easier
            away_diff -= (0.5 - opp_away_form) * 0.4

        team_difficulty.setdefault(f.home_team, []).append(max(1.0, min(5.0, home_diff)))
        team_difficulty.setdefault(f.away_team, []).append(max(1.0, min(5.0, away_diff)))

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


# Champions League / European competition teams (2024-25 season team IDs)
CL_TEAMS = {1, 6, 11, 12, 14, 20}  # Arsenal, Chelsea, Liverpool, Man City, Man Utd, Spurs (approximate)


def compute_team_form(players: list[Player]) -> dict[int, float]:
    """Compute team-level form (0-1) based on aggregate player form."""
    from collections import defaultdict
    team_stats = defaultdict(lambda: {"form_sum": 0.0, "count": 0})
    for p in players:
        team_stats[p.team]["form_sum"] += p.form
        team_stats[p.team]["count"] += 1

    forms = {}
    for tid, s in team_stats.items():
        forms[tid] = s["form_sum"] / s["count"] if s["count"] > 0 else 0.0

    vals = list(forms.values())
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {tid: 0.5 for tid in forms}
    return {tid: (v - lo) / (hi - lo) for tid, v in forms.items()}


def compute_rotation_risk(players: list[Player]) -> None:
    """Flag each player with a rotation_risk score (0.0 - 1.0)."""
    for p in players:
        risk = 0.0
        # Factor 1: Low starts-to-appearances ratio
        approx_apps = p.minutes / 90.0 if p.minutes > 0 else 0
        if approx_apps >= 3:
            start_ratio = p.starts / approx_apps if approx_apps > 0 else 0
            if start_ratio < 0.6:
                risk += 0.4
            elif start_ratio < 0.75:
                risk += 0.2

        # Factor 2: European competition team
        if p.team in CL_TEAMS:
            risk += 0.15

        # Factor 3: Chance of playing
        if p.chance_of_playing is not None and p.chance_of_playing < 75:
            risk += 0.3
        elif p.chance_of_playing is not None and p.chance_of_playing < 100:
            risk += 0.1

        # Factor 4: News suggesting rotation/doubt
        if p.news:
            nl = p.news.lower()
            if any(w in nl for w in ('doubt', 'rotation', 'rested', 'managed')):
                risk += 0.1

        p.rotation_risk = min(risk, 1.0)


def compute_projected_minutes(players: list[Player], gameweeks: list[Gameweek]) -> None:
    """Project expected minutes per GW based on recent history and availability."""
    finished_gws = sum(1 for gw in gameweeks if gw.finished)
    if finished_gws == 0:
        finished_gws = 1

    for p in players:
        # Base: average minutes per finished GW
        avg_mins = p.minutes / finished_gws

        # Adjust for chance of playing
        availability = 1.0
        if p.chance_of_playing is not None:
            availability = p.chance_of_playing / 100.0

        # Adjust for start ratio (players who start get ~90, subs get ~20)
        approx_apps = p.minutes / 90.0 if p.minutes > 0 else 0
        if approx_apps >= 3 and p.starts > 0:
            start_ratio = min(p.starts / approx_apps, 1.0)
            # Weighted: starters expected ~85 min, subs ~20 min
            mins_if_start = 85.0
            mins_if_sub = 20.0
            projected = start_ratio * mins_if_start + (1 - start_ratio) * mins_if_sub
        elif avg_mins > 0:
            projected = avg_mins
        else:
            projected = 0.0

        # Scale by availability and rotation risk
        projected *= availability
        projected *= (1.0 - p.rotation_risk * 0.3)

        p.projected_minutes = round(min(projected, 90.0), 1)


def score_players(
    players: list[Player],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
    teams: dict | None = None,
    lookahead: int = 5,
) -> None:
    team_form = compute_team_form(players) if teams else None
    compute_fixture_difficulty(players, fixtures, gameweeks, team_form=team_form, lookahead=lookahead)

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
