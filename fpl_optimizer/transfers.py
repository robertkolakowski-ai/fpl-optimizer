"""Transfer suggestions with proper EV math.

Best-practice version (raised from MVP):
  - Evaluates EP gain over a horizon (default 5 GWs), not single-GW score delta
  - Uses uncertainty.estimate_player_uncertainty to get fixture-weighted means
  - Subtracts -4 hit cost for each transfer beyond available free transfers
  - Honors FT rollover: at most 5 stored FT (FPL 2024+ rule; older was 2)
  - Recommends 0, 1, or 2 transfers based on which maximizes net EP

Why mean-not-composite-score: composite_score is a relative ranking
metric (normalized within position group). EP is in real points and can
be compared apples-to-apples to a -4 hit. Using the mean of the
fixture-weighted bootstrap gives a fair number that accounts for
upcoming opponent quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Fixture, Gameweek, Player, Squad

HIT_COST = 4  # Points cost per transfer beyond your free quota
MAX_FREE_TRANSFERS = 5  # FPL 2024+ cap


@dataclass
class TransferSuggestion:
    player_out: Player
    player_in: Player
    score_gain: float
    cost_change: float
    ep_gain_horizon: float = 0.0
    ep_gain_per_gw: float = 0.0
    horizon_gws: int = 1

    def to_dict(self) -> dict:
        pts_gain = round(self.player_in.ep_next - self.player_out.ep_next, 2)
        return {
            "player_out": self.player_out.to_dict(),
            "player_in": self.player_in.to_dict(),
            "score_gain": round(self.score_gain, 3),
            "cost_change": round(self.cost_change, 1),
            "pts_gain": pts_gain,
            "ep_gain_horizon": round(self.ep_gain_horizon, 2),
            "ep_gain_per_gw": round(self.ep_gain_per_gw, 2),
            "horizon_gws": self.horizon_gws,
            "breakeven_gws": round(HIT_COST / max(self.ep_gain_per_gw, 0.01), 1)
                              if self.ep_gain_per_gw > 0 else 99,
        }


@dataclass
class TransferPlan:
    """A 0/1/2-transfer plan with proper net-EP math."""
    transfers: list[TransferSuggestion] = field(default_factory=list)
    free_transfers_used: int = 0
    hit_cost: int = 0
    gross_ep_gain: float = 0.0
    net_ep_gain: float = 0.0
    horizon_gws: int = 1

    def to_dict(self) -> dict:
        return {
            "transfers": [t.to_dict() for t in self.transfers],
            "n_transfers": len(self.transfers),
            "free_transfers_used": self.free_transfers_used,
            "hit_cost": self.hit_cost,
            "gross_ep_gain": round(self.gross_ep_gain, 2),
            "net_ep_gain": round(self.net_ep_gain, 2),
            "horizon_gws": self.horizon_gws,
            "recommendation": (
                "Hold" if not self.transfers else
                "Take 1 transfer" if len(self.transfers) == 1 and self.hit_cost == 0 else
                f"Take {len(self.transfers)} transfers (-{self.hit_cost} hit)"
            ),
        }


def _player_ep_estimate(p: Player, uncertainty_map: dict[int, dict] | None) -> float:
    """Returns mean expected points per GW for a player.

    Prefers fixture-weighted bootstrap mean from #4 if available;
    falls back to FPL's ep_next, then to a composite-score-based estimate.
    """
    if uncertainty_map and p.id in uncertainty_map:
        return float(uncertainty_map[p.id].get("mean", 0))
    if p.ep_next:
        return float(p.ep_next)
    # Last resort: scale composite_score (which is in 0-1 range) to a
    # rough EP estimate using PPG as anchor
    return p.points_per_game or 3.0


def suggest_transfers(
    squad: Squad,
    all_players: list[Player],
    budget: float = 100.0,
    max_suggestions: int = 5,
    *,
    uncertainty_map: dict[int, dict] | None = None,
    horizon_gws: int = 5,
) -> list[TransferSuggestion]:
    """Single-transfer suggestions ranked by EP gain over horizon."""
    squad_ids = {p.id for p in squad.players}
    team_counts: dict[int, int] = {}
    for p in squad.players:
        team_counts[p.team] = team_counts.get(p.team, 0) + 1

    available = [p for p in all_players if p.id not in squad_ids]
    suggestions: list[TransferSuggestion] = []

    for out_player in squad.players:
        remaining_budget = budget - squad.total_cost + out_player.sell_value
        out_ep = _player_ep_estimate(out_player, uncertainty_map)

        for candidate in available:
            if candidate.position != out_player.position:
                continue
            if candidate.cost > remaining_budget:
                continue
            new_team_count = team_counts.get(candidate.team, 0)
            if candidate.team != out_player.team and new_team_count >= 3:
                continue

            in_ep = _player_ep_estimate(candidate, uncertainty_map)
            ep_gain_per_gw = in_ep - out_ep
            ep_gain_horizon = ep_gain_per_gw * horizon_gws
            score_gain = candidate.composite_score - out_player.composite_score

            if ep_gain_horizon > 0:
                suggestions.append(
                    TransferSuggestion(
                        player_out=out_player,
                        player_in=candidate,
                        score_gain=score_gain,
                        cost_change=candidate.cost - out_player.cost,
                        ep_gain_horizon=ep_gain_horizon,
                        ep_gain_per_gw=ep_gain_per_gw,
                        horizon_gws=horizon_gws,
                    )
                )

    suggestions.sort(key=lambda s: s.ep_gain_horizon, reverse=True)
    return suggestions[:max_suggestions]


def plan_transfers(
    squad: Squad,
    all_players: list[Player],
    *,
    free_transfers: int = 1,
    budget: float = 100.0,
    uncertainty_map: dict[int, dict] | None = None,
    horizon_gws: int = 5,
    consider_two_transfers: bool = True,
) -> TransferPlan:
    """Recommend the optimal 0/1/2-transfer plan that maximizes NET EP.

    Net EP = gross EP gain over horizon - 4*(transfers_used - free_transfers).
    Returns a TransferPlan with the highest net_ep_gain.
    """
    free_transfers = max(0, min(free_transfers, MAX_FREE_TRANSFERS))

    # Plan 0: do nothing
    plan_zero = TransferPlan(transfers=[], horizon_gws=horizon_gws)

    # Plan 1: best single transfer
    singles = suggest_transfers(
        squad, all_players, budget=budget, max_suggestions=1,
        uncertainty_map=uncertainty_map, horizon_gws=horizon_gws,
    )
    plan_one = TransferPlan(horizon_gws=horizon_gws)
    if singles:
        s = singles[0]
        plan_one.transfers = [s]
        plan_one.free_transfers_used = min(1, free_transfers)
        plan_one.hit_cost = HIT_COST * max(0, 1 - free_transfers)
        plan_one.gross_ep_gain = s.ep_gain_horizon
        plan_one.net_ep_gain = s.ep_gain_horizon - plan_one.hit_cost

    # Plan 2: best pair of disjoint transfers (greedy — pair top-2 from
    # different positions to keep solutions disjoint and fast)
    plan_two = TransferPlan(horizon_gws=horizon_gws)
    if consider_two_transfers:
        all_singles = suggest_transfers(
            squad, all_players, budget=budget, max_suggestions=20,
            uncertainty_map=uncertainty_map, horizon_gws=horizon_gws,
        )
        # Find top-2 disjoint by both out and in players
        top1 = all_singles[0] if all_singles else None
        top2 = None
        if top1:
            for s in all_singles[1:]:
                if (s.player_out.id != top1.player_out.id
                        and s.player_in.id != top1.player_in.id):
                    top2 = s
                    break
        if top1 and top2:
            plan_two.transfers = [top1, top2]
            plan_two.free_transfers_used = min(2, free_transfers)
            plan_two.hit_cost = HIT_COST * max(0, 2 - free_transfers)
            plan_two.gross_ep_gain = top1.ep_gain_horizon + top2.ep_gain_horizon
            plan_two.net_ep_gain = plan_two.gross_ep_gain - plan_two.hit_cost

    # Pick the plan with highest net gain
    plans = [plan_zero, plan_one, plan_two]
    return max(plans, key=lambda p: p.net_ep_gain)
