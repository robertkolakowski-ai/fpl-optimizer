from __future__ import annotations

from dataclasses import dataclass

from .models import Player, Squad


@dataclass
class TransferSuggestion:
    player_out: Player
    player_in: Player
    score_gain: float
    cost_change: float

    def to_dict(self) -> dict:
        pts_gain = round(self.player_in.ep_next - self.player_out.ep_next, 2)
        return {
            "player_out": self.player_out.to_dict(),
            "player_in": self.player_in.to_dict(),
            "score_gain": round(self.score_gain, 3),
            "cost_change": round(self.cost_change, 1),
            "pts_gain": pts_gain,
            "breakeven_gws": round(4 / max(pts_gain, 0.01), 1) if pts_gain > 0 else 99,
        }


def suggest_transfers(
    squad: Squad,
    all_players: list[Player],
    budget: float = 100.0,
    max_suggestions: int = 5,
) -> list[TransferSuggestion]:
    squad_ids = {p.id for p in squad.players}
    team_counts: dict[int, int] = {}
    for p in squad.players:
        team_counts[p.team] = team_counts.get(p.team, 0) + 1

    available = [p for p in all_players if p.id not in squad_ids]

    suggestions: list[TransferSuggestion] = []

    for out_player in squad.players:
        # Candidates: same position, affordable, won't break team limit
        # Use sell_value (FPL selling price) not market cost for budget calc
        remaining_budget = budget - squad.total_cost + out_player.sell_value

        for candidate in available:
            if candidate.position != out_player.position:
                continue
            if candidate.cost > remaining_budget:
                continue

            # Check team limit after swap
            new_team_count = team_counts.get(candidate.team, 0)
            if candidate.team != out_player.team:
                if new_team_count >= 3:
                    continue

            gain = candidate.composite_score - out_player.composite_score
            if gain > 0:
                suggestions.append(
                    TransferSuggestion(
                        player_out=out_player,
                        player_in=candidate,
                        score_gain=gain,
                        cost_change=candidate.cost - out_player.cost,
                    )
                )

    suggestions.sort(key=lambda s: s.score_gain, reverse=True)
    return suggestions[:max_suggestions]
