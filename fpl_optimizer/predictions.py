from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from .models import Fixture, Gameweek, Player, Team

# ---------------------------------------------------------------------------
# Default league averages (Premier League historical)
# ---------------------------------------------------------------------------
DEFAULT_HOME_GOALS = 1.35
DEFAULT_AWAY_GOALS = 1.10

INJURY_KEYWORDS = (
    "injured", "injury", "hamstring", "knee", "ankle", "groin", "muscle",
    "ligament", "surgery", "broken", "fracture", "illness", "doubt",
    "suspended", "ban", "hip", "thigh", "calf", "shoulder", "concussion",
)

DECAY_FACTOR = 0.9  # exponential decay per game


# ---------------------------------------------------------------------------
# MatchPrediction dataclass
# ---------------------------------------------------------------------------
@dataclass
class MatchPrediction:
    fixture_id: int
    gameweek: int
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    home_team_short: str
    away_team_short: str
    kickoff_time: str | None

    home_xg: float
    away_xg: float

    # 1X2
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Over/Under
    over_05: float
    over_15: float
    over_25: float
    over_35: float
    over_45: float

    # BTTS
    btts_yes: float

    # Scoreline matrix 6x6 (home rows, away cols)
    scoreline_matrix: list[list[float]] = field(default_factory=list)

    # Top 5 most likely results
    top_scorelines: list[dict] = field(default_factory=list)

    # Confidence
    confidence: str = "MEDIUM"

    # Form (last 5: W/D/L)
    home_form: list[str] = field(default_factory=list)
    away_form: list[str] = field(default_factory=list)

    # H2H this season
    h2h: list[dict] = field(default_factory=list)

    # Injuries
    home_injuries: list[dict] = field(default_factory=list)
    away_injuries: list[dict] = field(default_factory=list)

    # FDR
    home_fdr: int = 3
    away_fdr: int = 3

    # Strength metrics (for display)
    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """P(X=k) for Poisson distribution with rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def _build_scoreline_matrix(home_xg: float, away_xg: float, max_goals: int = 6) -> list[list[float]]:
    """6x6 matrix of P(home=i, away=j)."""
    return [
        [_poisson_pmf(i, home_xg) * _poisson_pmf(j, away_xg) for j in range(max_goals)]
        for i in range(max_goals)
    ]


def _derive_probabilities(matrix: list[list[float]]) -> dict:
    """From scoreline matrix, derive 1X2, O/U, BTTS."""
    home_win = draw = away_win = 0.0
    over = {0.5: 0.0, 1.5: 0.0, 2.5: 0.0, 3.5: 0.0, 4.5: 0.0}
    btts_yes = 0.0

    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            total = i + j
            for threshold in over:
                if total > threshold:
                    over[threshold] += p
            if i > 0 and j > 0:
                btts_yes += p

    # Normalise 1X2 to sum to 1
    s = home_win + draw + away_win
    if s > 0:
        home_win /= s
        draw /= s
        away_win /= s

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_05": over[0.5],
        "over_15": over[1.5],
        "over_25": over[2.5],
        "over_35": over[3.5],
        "over_45": over[4.5],
        "btts_yes": btts_yes,
    }


def _top_scorelines(matrix: list[list[float]], n: int = 5) -> list[dict]:
    """Return top N most likely scorelines."""
    scores = []
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            scores.append({"home": i, "away": j, "prob": round(p, 4)})
    scores.sort(key=lambda s: s["prob"], reverse=True)
    return scores[:n]


def _compute_team_strengths(
    fixtures: list[Fixture],
    players: list[Player],
) -> tuple[dict[int, dict], float, float]:
    """Compute per-team attack/defense strengths and league averages.

    Returns (team_strengths, league_avg_home_goals, league_avg_away_goals).
    team_strengths[team_id] = {home_attack, home_defense, away_attack, away_defense}.
    """
    # Gather finished fixtures sorted by gameweek (most recent last)
    finished = sorted(
        [f for f in fixtures if f.finished and f.home_score is not None and f.away_score is not None],
        key=lambda f: f.gameweek or 0,
    )

    if not finished:
        # Season start fallback — use player xG aggregation
        return _strengths_from_player_xg(players), DEFAULT_HOME_GOALS, DEFAULT_AWAY_GOALS

    # Per-team weighted goal records (exponential decay)
    team_data: dict[int, dict] = defaultdict(lambda: {
        "home_gf_w": 0.0, "home_ga_w": 0.0, "home_w": 0.0,
        "away_gf_w": 0.0, "away_ga_w": 0.0, "away_w": 0.0,
    })

    # Build per-team fixture list to compute decay
    home_fixtures: dict[int, list[Fixture]] = defaultdict(list)
    away_fixtures: dict[int, list[Fixture]] = defaultdict(list)
    for f in finished:
        home_fixtures[f.home_team].append(f)
        away_fixtures[f.away_team].append(f)

    total_home_goals = 0.0
    total_away_goals = 0.0
    total_matches = 0

    for f in finished:
        total_home_goals += f.home_score
        total_away_goals += f.away_score
        total_matches += 1

    league_avg_home = total_home_goals / total_matches if total_matches else DEFAULT_HOME_GOALS
    league_avg_away = total_away_goals / total_matches if total_matches else DEFAULT_AWAY_GOALS

    # Apply exponential decay per team (most recent game = weight 1.0)
    for team_id in set(f.home_team for f in finished) | set(f.away_team for f in finished):
        # Home games
        hf = home_fixtures.get(team_id, [])
        for idx, f in enumerate(reversed(hf)):
            w = DECAY_FACTOR ** idx
            team_data[team_id]["home_gf_w"] += f.home_score * w
            team_data[team_id]["home_ga_w"] += f.away_score * w
            team_data[team_id]["home_w"] += w

        # Away games
        af = away_fixtures.get(team_id, [])
        for idx, f in enumerate(reversed(af)):
            w = DECAY_FACTOR ** idx
            team_data[team_id]["away_gf_w"] += f.away_score * w
            team_data[team_id]["away_ga_w"] += f.home_score * w
            team_data[team_id]["away_w"] += w

    # Compute strengths
    strengths: dict[int, dict] = {}
    for team_id, d in team_data.items():
        home_gf_avg = d["home_gf_w"] / d["home_w"] if d["home_w"] > 0 else league_avg_home
        home_ga_avg = d["home_ga_w"] / d["home_w"] if d["home_w"] > 0 else league_avg_away
        away_gf_avg = d["away_gf_w"] / d["away_w"] if d["away_w"] > 0 else league_avg_away
        away_ga_avg = d["away_ga_w"] / d["away_w"] if d["away_w"] > 0 else league_avg_home

        strengths[team_id] = {
            "home_attack": home_gf_avg / league_avg_home if league_avg_home > 0 else 1.0,
            "home_defense": home_ga_avg / league_avg_away if league_avg_away > 0 else 1.0,
            "away_attack": away_gf_avg / league_avg_away if league_avg_away > 0 else 1.0,
            "away_defense": away_ga_avg / league_avg_home if league_avg_home > 0 else 1.0,
        }

    return strengths, league_avg_home, league_avg_away


def _strengths_from_player_xg(players: list[Player]) -> dict[int, dict]:
    """Fallback: derive team strengths from player xG when no fixtures finished."""
    team_xg: dict[int, float] = defaultdict(float)
    for p in players:
        team_xg[p.team] += p.xG

    avg_xg = sum(team_xg.values()) / len(team_xg) if team_xg else 1.0
    if avg_xg == 0:
        avg_xg = 1.0

    strengths: dict[int, dict] = {}
    for tid, xg in team_xg.items():
        ratio = xg / avg_xg
        strengths[tid] = {
            "home_attack": ratio,
            "home_defense": 1.0,
            "away_attack": ratio,
            "away_defense": 1.0,
        }
    return strengths


def _apply_injury_adjustment(
    strengths: dict[int, dict],
    players: list[Player],
    team_id: int,
    venue: str,
) -> float:
    """Reduce attack strength if top xG contributors are injured. Returns adjusted attack."""
    key = f"{venue}_attack"
    base = strengths.get(team_id, {}).get(key, 1.0)

    team_players = [p for p in players if p.team == team_id]
    if not team_players:
        return base

    total_xg = sum(p.xG for p in team_players) or 1.0
    injured_xg = 0.0
    for p in team_players:
        is_out = False
        if p.chance_of_playing is not None and p.chance_of_playing < 50:
            is_out = True
        elif p.news:
            nl = p.news.lower()
            if any(kw in nl for kw in INJURY_KEYWORDS):
                is_out = True
        if is_out:
            injured_xg += p.xG

    reduction = min(injured_xg / total_xg * 0.30, 0.15)  # cap at 15%
    return base * (1.0 - reduction)


def _team_form(fixtures: list[Fixture], team_id: int) -> list[str]:
    """Last 5 results for a team (most recent first)."""
    team_fixtures = sorted(
        [f for f in fixtures
         if f.finished and f.home_score is not None
         and (f.home_team == team_id or f.away_team == team_id)],
        key=lambda f: f.gameweek or 0,
        reverse=True,
    )
    results = []
    for f in team_fixtures[:5]:
        if f.home_team == team_id:
            diff = f.home_score - f.away_score
        else:
            diff = f.away_score - f.home_score
        results.append("W" if diff > 0 else "D" if diff == 0 else "L")
    return results


def _team_injuries(players: list[Player], team_id: int) -> list[dict]:
    """Injured/doubtful players for a team."""
    injured = []
    for p in players:
        if p.team != team_id:
            continue
        is_out = False
        if p.chance_of_playing is not None and p.chance_of_playing < 75:
            is_out = True
        if p.news:
            nl = p.news.lower()
            if any(kw in nl for kw in INJURY_KEYWORDS):
                is_out = True
        if is_out:
            injured.append({
                "id": p.id,
                "name": p.name,
                "position": p.position_name,
                "chance": p.chance_of_playing,
                "news": p.news or "",
            })
    return injured


def _h2h_this_season(
    fixtures: list[Fixture],
    home_id: int,
    away_id: int,
    teams: dict[int, Team],
) -> list[dict]:
    """Head-to-head results between two teams this season."""
    results = []
    for f in fixtures:
        if not f.finished or f.home_score is None:
            continue
        if (f.home_team == home_id and f.away_team == away_id) or \
           (f.home_team == away_id and f.away_team == home_id):
            results.append({
                "gameweek": f.gameweek,
                "home_team": teams[f.home_team].short_name if f.home_team in teams else "???",
                "away_team": teams[f.away_team].short_name if f.away_team in teams else "???",
                "home_score": f.home_score,
                "away_score": f.away_score,
            })
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_predictions(
    fixtures: list[Fixture],
    players: list[Player],
    teams: dict[int, Team],
    gameweeks: list[Gameweek],
    target_gw: int | None = None,
) -> list[MatchPrediction]:
    """Generate Poisson-based match predictions for a gameweek."""
    # Determine target GW
    if target_gw is None:
        gw = next((g for g in gameweeks if g.is_next), None)
        if gw is None:
            gw = next((g for g in gameweeks if g.is_current), None)
        if gw is None:
            return []
        target_gw = gw.id

    gw_fixtures = [f for f in fixtures if f.gameweek == target_gw]
    if not gw_fixtures:
        return []

    # Compute strengths once
    strengths, avg_home, avg_away = _compute_team_strengths(fixtures, players)

    predictions = []
    for f in gw_fixtures:
        ht = teams.get(f.home_team)
        at = teams.get(f.away_team)
        if not ht or not at:
            continue

        # Expected goals with injury adjustment
        home_att = _apply_injury_adjustment(strengths, players, f.home_team, "home")
        away_def = strengths.get(f.away_team, {}).get("away_defense", 1.0)
        away_att = _apply_injury_adjustment(strengths, players, f.away_team, "away")
        home_def = strengths.get(f.home_team, {}).get("home_defense", 1.0)

        home_xg = home_att * away_def * avg_home
        away_xg = away_att * home_def * avg_away

        # Clamp xG to reasonable range
        home_xg = max(0.2, min(home_xg, 4.5))
        away_xg = max(0.2, min(away_xg, 4.5))

        # Scoreline matrix & probabilities
        matrix = _build_scoreline_matrix(home_xg, away_xg)
        probs = _derive_probabilities(matrix)
        top5 = _top_scorelines(matrix)

        # Confidence
        max_prob = max(probs["home_win"], probs["draw"], probs["away_win"])
        if max_prob > 0.45:
            confidence = "HIGH"
        elif max_prob < 0.30:
            confidence = "LOW"
        else:
            confidence = "MEDIUM"

        # Round matrix values for JSON
        rounded_matrix = [
            [round(cell, 4) for cell in row]
            for row in matrix
        ]

        predictions.append(MatchPrediction(
            fixture_id=f.id,
            gameweek=target_gw,
            home_team_id=f.home_team,
            away_team_id=f.away_team,
            home_team_name=ht.name,
            away_team_name=at.name,
            home_team_short=ht.short_name,
            away_team_short=at.short_name,
            kickoff_time=f.kickoff_time,
            home_xg=round(home_xg, 2),
            away_xg=round(away_xg, 2),
            home_win_prob=round(probs["home_win"], 4),
            draw_prob=round(probs["draw"], 4),
            away_win_prob=round(probs["away_win"], 4),
            over_05=round(probs["over_05"], 4),
            over_15=round(probs["over_15"], 4),
            over_25=round(probs["over_25"], 4),
            over_35=round(probs["over_35"], 4),
            over_45=round(probs["over_45"], 4),
            btts_yes=round(probs["btts_yes"], 4),
            scoreline_matrix=rounded_matrix,
            top_scorelines=top5,
            confidence=confidence,
            home_form=_team_form(fixtures, f.home_team),
            away_form=_team_form(fixtures, f.away_team),
            h2h=_h2h_this_season(fixtures, f.home_team, f.away_team, teams),
            home_injuries=_team_injuries(players, f.home_team),
            away_injuries=_team_injuries(players, f.away_team),
            home_fdr=f.home_difficulty,
            away_fdr=f.away_difficulty,
            home_attack_strength=round(home_att, 3),
            home_defense_strength=round(home_def, 3),
            away_attack_strength=round(away_att, 3),
            away_defense_strength=round(away_def, 3),
        ))

    # Sort by kickoff time
    predictions.sort(key=lambda p: p.kickoff_time or "")
    return predictions
