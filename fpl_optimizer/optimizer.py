from __future__ import annotations

from pulp import PULP_CBC_CMD, LpMaximize, LpProblem, LpVariable, lpSum, value

from .models import Player, Squad

POSITION_LIMITS = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
MAX_PER_TEAM = 3
SQUAD_SIZE = 15

# Starting XI formation constraints
STARTING_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
STARTING_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
STARTING_SIZE = 11


def _compute_player_score(p: Player, risk_mode: str = "balanced") -> float:
    """Compute objective score for a player based on risk mode.

    Modes:
    - 'safe': favours high-floor players (consistent points, high ownership,
      low variance). Penalises rotation risk and low minutes.
    - 'aggressive': favours high-ceiling differentials (low ownership,
      high xG/xA upside, fixture ease). Rewards risk-taking.
    - 'balanced': plain composite_score (default).
    """
    base = p.composite_score

    if risk_mode == "safe":
        # Reward consistency: points_per_game and minutes reliability
        consistency = 0.0
        if p.minutes > 0:
            # High starts ratio = reliable starter
            approx_apps = p.minutes / 90.0
            if approx_apps > 0 and p.starts > 0:
                consistency = min(p.starts / approx_apps, 1.0)
        # Penalise rotation risk
        risk_penalty = p.rotation_risk * 0.15
        # Reward high ownership (template = safe)
        ownership_bonus = min(p.selected_by_percent / 100.0, 1.0) * 0.1
        return base + consistency * 0.1 + ownership_bonus - risk_penalty

    elif risk_mode == "aggressive":
        # Reward low ownership (differential upside)
        diff_bonus = max(0, (20 - p.selected_by_percent) / 20.0) * 0.15
        # Reward high xG+xA ceiling
        xgi_bonus = (p.xG + p.xA) * 0.05
        # Reward fixture ease more heavily
        fixture_bonus = p.fixture_difficulty * 0.1
        # Penalise template players
        template_penalty = min(p.selected_by_percent / 100.0, 1.0) * 0.05
        return base + diff_bonus + xgi_bonus + fixture_bonus - template_penalty

    return base


def select_squad(
    players: list[Player],
    budget: float = 100.0,
    risk_mode: str = "balanced",
) -> Squad:
    prob = LpProblem("FPL_Squad", LpMaximize)

    # Binary variable per player
    x = {p.id: LpVariable(f"x_{p.id}", cat="Binary") for p in players}

    # Objective: maximize risk-adjusted score
    prob += lpSum(_compute_player_score(p, risk_mode) * x[p.id] for p in players)

    # Squad size = 15
    prob += lpSum(x[p.id] for p in players) == SQUAD_SIZE

    # Position limits
    for pos, limit in POSITION_LIMITS.items():
        prob += lpSum(x[p.id] for p in players if p.position == pos) == limit

    # Max 3 per team
    team_ids = set(p.team for p in players)
    for tid in team_ids:
        prob += lpSum(x[p.id] for p in players if p.team == tid) <= MAX_PER_TEAM

    # Budget
    prob += lpSum(p.cost * x[p.id] for p in players) <= budget

    prob.solve(PULP_CBC_CMD(msg=False))

    squad_players = sorted(
        [p for p in players if value(x[p.id]) == 1],
        key=lambda p: (p.position, -p.composite_score),
    )

    squad = Squad(players=squad_players)
    squad.budget_remaining = budget - squad.total_cost

    # Pick starting XI
    _select_starting(squad)

    return squad


def _select_starting(squad: Squad) -> None:
    prob = LpProblem("FPL_Starting", LpMaximize)

    players = squad.players
    y = {p.id: LpVariable(f"y_{p.id}", cat="Binary") for p in players}

    prob += lpSum(p.composite_score * y[p.id] for p in players)

    # Exactly 11 starters
    prob += lpSum(y[p.id] for p in players) == STARTING_SIZE

    # Formation constraints
    for pos in (1, 2, 3, 4):
        pos_players = [p for p in players if p.position == pos]
        prob += lpSum(y[p.id] for p in pos_players) >= STARTING_MIN[pos]
        prob += lpSum(y[p.id] for p in pos_players) <= STARTING_MAX[pos]

    prob.solve(PULP_CBC_CMD(msg=False))

    squad.starting = sorted(
        [p for p in players if value(y[p.id]) == 1],
        key=lambda p: (p.position, -p.composite_score),
    )
    squad.bench = sorted(
        [p for p in players if value(y[p.id]) != 1],
        key=lambda p: (p.position, -p.composite_score),
    )
