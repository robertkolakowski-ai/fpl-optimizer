from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

from .models import Fixture, Gameweek, Player, Team

logger = logging.getLogger(__name__)

FD_BASE_URL = "https://api.football-data.org/v4"
FD_COMPETITION = "PL"
FD_SEASON = 2024  # 2024-25 season

# Rate limiter: 10 req/min → min 6 sec between requests
_last_request_time: float = 0.0
_MIN_REQUEST_INTERVAL = 6.0

# In-memory cache
_fd_cache: dict = {}
_SEASON_CACHE_TTL = 1800  # 30 min


# ---------------------------------------------------------------------------
# FPL team code → Football-Data.org team ID mapping (2024-25 PL)
# ---------------------------------------------------------------------------
FPL_CODE_TO_FD_ID: dict[int, int] = {
    3: 57,    # Arsenal
    7: 58,    # Aston Villa
    91: 397,  # Brighton
    90: 1044, # Bournemouth
    8: 61,    # Chelsea
    31: 354,  # Crystal Palace
    11: 62,   # Everton
    54: 63,   # Fulham
    40: 349,  # Ipswich Town
    13: 338,  # Leicester City
    14: 64,   # Liverpool
    43: 65,   # Man City
    1: 66,    # Man United
    4: 67,    # Newcastle
    17: 68,   # Nottingham Forest
    20: 340,  # Southampton
    6: 73,    # Tottenham
    21: 563,  # West Ham
    39: 76,   # Wolverhampton
}

# Reverse mapping for convenience
FD_ID_TO_FPL_CODE: dict[int, int] = {v: k for k, v in FPL_CODE_TO_FD_ID.items()}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TeamSeasonStats:
    """Team stats derived from FPL player data + fixture results."""
    fpl_team_id: int = 0
    name: str = ""
    matches_played: int = 0
    home_matches: int = 0
    away_matches: int = 0
    # Card stats (from FPL player data)
    total_yellows: int = 0
    total_reds: int = 0
    avg_cards_per_match: float = 0.0
    avg_cards_home: float = 0.0
    avg_cards_away: float = 0.0
    # Shot/xG stats (from FPL player data)
    team_xg: float = 0.0
    team_xa: float = 0.0
    team_threat: float = 0.0
    avg_shots_est: float = 0.0       # estimated from threat
    avg_sot_est: float = 0.0         # estimated from xG
    # Corner estimate (from creativity + attacking play)
    avg_corners_est: float = 0.0
    avg_corners_home_est: float = 0.0
    avg_corners_away_est: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fpl_team_id": self.fpl_team_id,
            "name": self.name,
            "matches_played": self.matches_played,
            "total_yellows": self.total_yellows,
            "total_reds": self.total_reds,
            "avg_cards_per_match": round(self.avg_cards_per_match, 2),
            "avg_cards_home": round(self.avg_cards_home, 2),
            "avg_cards_away": round(self.avg_cards_away, 2),
            "team_xg": round(self.team_xg, 2),
            "team_xa": round(self.team_xa, 2),
            "avg_shots": round(self.avg_shots_est, 1),
            "avg_shots_on_target": round(self.avg_sot_est, 1),
            "avg_corners_home": round(self.avg_corners_home_est, 1),
            "avg_corners_away": round(self.avg_corners_away_est, 1),
            "avg_corners_total": round(self.avg_corners_est, 1),
        }


@dataclass
class RefereeStats:
    """Referee stats — matches officiated from Football-Data.org,
    card counts estimated from FPL aggregate if bookings unavailable."""
    name: str = ""
    matches_officiated: int = 0
    total_yellows: int = 0
    total_reds: int = 0
    avg_yellows_per_match: float = 0.0
    avg_reds_per_match: float = 0.0
    total_penalties: int = 0
    penalty_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "matches_officiated": self.matches_officiated,
            "avg_yellows_per_match": round(self.avg_yellows_per_match, 2),
            "avg_reds_per_match": round(self.avg_reds_per_match, 2),
            "penalty_rate": round(self.penalty_rate, 2),
        }


# ---------------------------------------------------------------------------
# HTTP helpers (Football-Data.org)
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return os.environ.get("FOOTBALL_DATA_API_KEY", "")


def _rate_limit() -> None:
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _fd_get(url: str) -> dict | list | None:
    api_key = _get_api_key()
    if not api_key:
        logger.warning("FOOTBALL_DATA_API_KEY not set — skipping external data")
        return None
    _rate_limit()
    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(url, headers={"X-Auth-Token": api_key})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Football-Data.org request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Football-Data.org — fetch season matches (for referee assignments)
# ---------------------------------------------------------------------------

def fetch_season_matches() -> list[dict]:
    """Fetch all PL matches for the season (cached 30 min)."""
    now = time.time()
    if _fd_cache.get("season_matches") and now - _fd_cache.get("season_ts", 0) < _SEASON_CACHE_TTL:
        return _fd_cache["season_matches"]

    url = f"{FD_BASE_URL}/competitions/{FD_COMPETITION}/matches?season={FD_SEASON}"
    data = _fd_get(url)
    if not data or "matches" not in data:
        return _fd_cache.get("season_matches", [])

    matches = data["matches"]
    _fd_cache["season_matches"] = matches
    _fd_cache["season_ts"] = now
    return matches


def get_match_referee(matches: list[dict], home_fd_id: int, away_fd_id: int) -> str | None:
    """Find referee assigned to an upcoming or recent match between two teams."""
    # Check scheduled first
    for m in matches:
        if m.get("status") in ("SCHEDULED", "TIMED"):
            h_id = m.get("homeTeam", {}).get("id", 0)
            a_id = m.get("awayTeam", {}).get("id", 0)
            if h_id == home_fd_id and a_id == away_fd_id:
                refs = m.get("referees", [])
                for r in refs:
                    if r.get("type") == "REFEREE":
                        return r.get("name")
                if refs:
                    return refs[0].get("name")
    return None


def get_fd_team_id(fpl_team_code: int) -> int | None:
    return FPL_CODE_TO_FD_ID.get(fpl_team_code)


# ---------------------------------------------------------------------------
# Build referee stats from Football-Data.org match list
# Free tier doesn't include bookings, so we track matches per referee
# and use league-average card rates for estimation
# ---------------------------------------------------------------------------

_PL_AVG_YELLOWS_PER_MATCH = 3.5  # Premier League historical average
_PL_AVG_REDS_PER_MATCH = 0.12
_PL_AVG_PENALTIES_PER_MATCH = 0.28


def build_referee_stats(matches: list[dict]) -> dict[str, RefereeStats]:
    """Build referee stats from season matches.

    Since the free tier doesn't include booking data, we track matches
    officiated and use league-average card rates.
    """
    ref_matches: dict[str, int] = defaultdict(int)

    finished = [m for m in matches if m.get("status") == "FINISHED"]
    for m in finished:
        refs = m.get("referees", [])
        for r in refs:
            if r.get("type") == "REFEREE":
                name = r.get("name", "")
                if name:
                    ref_matches[name] += 1
                break

    result: dict[str, RefereeStats] = {}
    for name, n in ref_matches.items():
        result[name] = RefereeStats(
            name=name,
            matches_officiated=n,
            avg_yellows_per_match=_PL_AVG_YELLOWS_PER_MATCH,
            avg_reds_per_match=_PL_AVG_REDS_PER_MATCH,
            penalty_rate=_PL_AVG_PENALTIES_PER_MATCH,
        )
    return result


# ---------------------------------------------------------------------------
# Build team stats from FPL player data + fixtures
# ---------------------------------------------------------------------------

def build_team_stats_from_fpl(
    players: list[Player],
    teams: dict[int, Team],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
) -> dict[int, TeamSeasonStats]:
    """Build team-level season stats from FPL API player data.

    This uses actual FPL data: player yellow/red cards, xG, xA,
    threat, creativity — aggregated per team.
    """
    finished_gws = sum(1 for gw in gameweeks if gw.finished) or 1

    # Count home/away matches per team
    team_home_matches: dict[int, int] = defaultdict(int)
    team_away_matches: dict[int, int] = defaultdict(int)
    for f in fixtures:
        if f.finished and f.home_score is not None:
            team_home_matches[f.home_team] += 1
            team_away_matches[f.away_team] += 1

    # Aggregate player stats per team
    team_agg: dict[int, dict] = defaultdict(lambda: {
        "yellows": 0, "reds": 0, "xg": 0.0, "xa": 0.0,
        "threat": 0.0, "creativity": 0.0, "players": 0,
    })
    for p in players:
        ta = team_agg[p.team]
        ta["yellows"] += p.yellow_cards
        ta["reds"] += p.red_cards
        ta["xg"] += p.xG
        ta["xa"] += p.xA
        ta["threat"] += p.threat
        ta["creativity"] += float(p.creativity)
        ta["players"] += 1

    stats: dict[int, TeamSeasonStats] = {}
    for tid, agg in team_agg.items():
        t = teams.get(tid)
        name = t.short_name if t else f"Team {tid}"
        home_m = team_home_matches.get(tid, 0)
        away_m = team_away_matches.get(tid, 0)
        total_m = home_m + away_m or finished_gws

        # Card averages
        total_cards = agg["yellows"] + agg["reds"]
        avg_cards = total_cards / total_m if total_m else 0
        # Estimate home/away split: home teams average ~5% fewer cards
        avg_home = avg_cards * 0.95 if home_m > 0 else avg_cards
        avg_away = avg_cards * 1.05 if away_m > 0 else avg_cards

        # Shot estimates from threat
        # PL: threat/match ranges ~70-190, avg ~130. PL avg ~13 shots/match.
        # Calibration: threat/10 maps 70→7, 130→13, 190→19
        threat_per_match = agg["threat"] / total_m if total_m else 130
        est_shots = max(7.0, min(20.0, threat_per_match / 10))
        # SOT from xG: PL avg ~4.5 SOT/match, avg xG/match ~1.35
        # Calibration: xG * 3.0 maps 0.6→1.8, 1.35→4.1, 2.2→6.6
        xg_per_match = agg["xg"] / total_m if total_m else 1.35
        est_sot = max(2.0, min(8.0, xg_per_match * 3.0))

        # Corner estimates from creativity
        # PL: creativity/match ranges ~90-200, avg ~140. PL avg ~5.2 corners/team.
        # Calibration: creativity/27 maps 90→3.3, 140→5.2, 200→7.4
        creativity_per_match = agg["creativity"] / total_m if total_m else 140
        corners_est = max(3.0, min(8.0, creativity_per_match / 27))
        # Home teams average ~0.5 more corners
        corners_home = corners_est + 0.25
        corners_away = corners_est - 0.25

        stats[tid] = TeamSeasonStats(
            fpl_team_id=tid,
            name=name,
            matches_played=total_m,
            home_matches=home_m,
            away_matches=away_m,
            total_yellows=agg["yellows"],
            total_reds=agg["reds"],
            avg_cards_per_match=round(avg_cards, 2),
            avg_cards_home=round(avg_home, 2),
            avg_cards_away=round(avg_away, 2),
            team_xg=round(agg["xg"], 2),
            team_xa=round(agg["xa"], 2),
            team_threat=round(agg["threat"], 1),
            avg_shots_est=round(est_shots, 1),
            avg_sot_est=round(est_sot, 1),
            avg_corners_est=round(corners_est, 1),
            avg_corners_home_est=round(corners_home, 1),
            avg_corners_away_est=round(corners_away, 1),
        )

    return stats


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def get_referee_data() -> tuple[list[dict], dict[str, RefereeStats]]:
    """Fetch referee data from Football-Data.org.

    Returns (raw_matches, referee_stats). Both empty if API unavailable.
    """
    now = time.time()
    if (_fd_cache.get("ref_stats")
            and now - _fd_cache.get("ref_ts", 0) < _SEASON_CACHE_TTL):
        return _fd_cache.get("season_matches", []), _fd_cache["ref_stats"]

    matches = fetch_season_matches()
    if not matches:
        return [], {}

    ref_stats = build_referee_stats(matches)
    _fd_cache["ref_stats"] = ref_stats
    _fd_cache["ref_ts"] = now
    return matches, ref_stats


def get_team_stats(
    players: list[Player],
    teams: dict[int, Team],
    fixtures: list[Fixture],
    gameweeks: list[Gameweek],
) -> dict[int, TeamSeasonStats]:
    """Build team stats from FPL data (cached 30 min)."""
    now = time.time()
    if _fd_cache.get("team_stats") and now - _fd_cache.get("team_stats_ts", 0) < _SEASON_CACHE_TTL:
        return _fd_cache["team_stats"]

    stats = build_team_stats_from_fpl(players, teams, fixtures, gameweeks)
    _fd_cache["team_stats"] = stats
    _fd_cache["team_stats_ts"] = now
    return stats
