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


def select_squad(players: list[Player], budget: float = 100.0) -> Squad:
    prob = LpProblem("FPL_Squad", LpMaximize)

    # Binary variable per player
    x = {p.id: LpVariable(f"x_{p.id}", cat="Binary") for p in players}

    # Objective: maximize composite score
    prob += lpSum(p.composite_score * x[p.id] for p in players)

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
