from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Team:
    id: int
    name: str
    short_name: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fixture:
    id: int
    gameweek: int | None
    home_team: int
    away_team: int
    home_difficulty: int
    away_difficulty: int
    finished: bool
    home_score: int | None = None
    away_score: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gameweek:
    id: int
    name: str
    finished: bool
    is_current: bool
    is_next: bool


@dataclass
class Player:
    id: int
    name: str
    team: int
    position: int  # 1=GK, 2=DEF, 3=MID, 4=FWD
    cost: float  # in millions (e.g. 6.5)
    total_points: int = 0
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    bonus: int = 0
    form: float = 0.0
    points_per_game: float = 0.0
    xG: float = 0.0
    xA: float = 0.0
    ict_index: float = 0.0
    composite_score: float = 0.0
    fixture_difficulty: float = 0.0
    photo: str = ""
    selected_by_percent: float = 0.0
    transfers_in_event: int = 0
    transfers_out_event: int = 0
    cost_change_event: int = 0
    news: str = ""
    chance_of_playing: int | None = None
    # Expected points (FPL official)
    ep_next: float = 0.0
    ep_this: float = 0.0
    # Per-90 expected stats
    xG_per90: float = 0.0
    xA_per90: float = 0.0
    xGI_per90: float = 0.0
    xGC_per90: float = 0.0
    # Opta / BPS fields
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    expected_goal_involvements: float = 0.0
    expected_goals_conceded: float = 0.0
    bps: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    starts: int = 0

    @property
    def position_name(self) -> str:
        return {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}.get(self.position, "??")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position_name"] = self.position_name
        return d


@dataclass
class Squad:
    players: list[Player] = field(default_factory=list)
    starting: list[Player] = field(default_factory=list)
    bench: list[Player] = field(default_factory=list)
    budget_remaining: float = 0.0

    @property
    def total_cost(self) -> float:
        return sum(p.cost for p in self.players)

    def to_dict(self) -> dict:
        return {
            "starting": [p.to_dict() for p in self.starting],
            "bench": [p.to_dict() for p in self.bench],
            "total_cost": round(self.total_cost, 1),
            "budget_remaining": round(self.budget_remaining, 1),
            "total_score": round(sum(p.composite_score for p in self.players), 3),
        }
