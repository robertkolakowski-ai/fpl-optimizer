"""Loader for empiriske lag-priors fra data/team_priors.json.

Oppdateres 2× per uke via scripts/update_priors.ps1 (Windows Task Scheduler)
eller manuelt via scripts/build_team_priors.py.

Eksponeres via /api/team-priors og brukes i:
  - predictions.py (Dixon-Coles-priors basert på empirisk xG)
  - analyzer.py (clean-sheet-rater til defender/GK-scoring)
  - frontend-narrativer (Hjem, Liga, player-modal)

Filen reloades dynamisk hvis modtid har endret seg — slik at neste request
etter update_priors.ps1 har commitet får friske data uten Render-restart.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _data_dir() -> Path:
    base = os.environ.get("FPL_DATA_DIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data"


def _path() -> Path:
    return _data_dir() / "team_priors.json"


_state: dict[str, Any] = {"loaded_mtime": 0, "data": None}


def _load_if_changed() -> dict[str, Any] | None:
    p = _path()
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
        with _LOCK:
            if _state["loaded_mtime"] != mtime or _state["data"] is None:
                with p.open("r", encoding="utf-8") as f:
                    _state["data"] = json.load(f)
                _state["loaded_mtime"] = mtime
            return _state["data"]
    except Exception:
        return None


def get_priors() -> dict[str, Any]:
    data = _load_if_changed()
    if not data:
        return {"version": 0, "teams": {}, "teams_count": 0}
    return data


def get_team(team_name: str) -> dict[str, Any] | None:
    """Slå opp et lag på navn. Tolerant for små variasjoner."""
    data = _load_if_changed()
    if not data or "teams" not in data:
        return None
    teams = data["teams"]
    # Prøv eksakt match
    if team_name in teams:
        return teams[team_name]
    # Case-insensitive match
    lname = team_name.lower()
    for name, vals in teams.items():
        if name.lower() == lname:
            return vals
    return None


def fixture_difficulty_from_xg(home_team: str, away_team: str) -> dict[str, Any] | None:
    """Beregn empirisk fixture difficulty fra xG.

    Returnerer:
        {
          "home_xg_expected": float,    # forventet mål for hjemmelaget
          "away_xg_expected": float,    # forventet mål for bortelaget
          "home_difficulty": int (1-5), # vår skala, 1=lett 5=hard
          "away_difficulty": int (1-5),
        }
    Eller None hvis ikke begge lag har xG-data.
    """
    h = get_team(home_team)
    a = get_team(away_team)
    if not h or not a:
        return None
    h_xg = h.get("xg_home")
    h_xga = h.get("xga_home")
    a_xg = a.get("xg_away")
    a_xga = a.get("xga_away")
    if None in (h_xg, h_xga, a_xg, a_xga):
        return None
    # Forventet mål: (lagets attack-xG + motstanders defense-xGA) / 2
    home_expected = (h_xg + a_xga) / 2
    away_expected = (a_xg + h_xga) / 2
    # Skala til 1-5: jo mindre forventet motstand-xG, jo lettere kamp
    def to_difficulty(opp_expected: float) -> int:
        if opp_expected < 0.9:
            return 1
        if opp_expected < 1.2:
            return 2
        if opp_expected < 1.5:
            return 3
        if opp_expected < 1.9:
            return 4
        return 5
    return {
        "home_xg_expected": round(home_expected, 2),
        "away_xg_expected": round(away_expected, 2),
        "home_difficulty": to_difficulty(away_expected),
        "away_difficulty": to_difficulty(home_expected),
    }


def league_xg_table() -> list[dict[str, Any]]:
    """Returnerer lag sortert etter net xG — fra sterkest til svakest."""
    data = _load_if_changed()
    if not data or "teams" not in data:
        return []
    rows = []
    for name, vals in data["teams"].items():
        rows.append({
            "team": name,
            "xg_avg": vals.get("xg_avg"),
            "xga_avg": vals.get("xga_avg"),
            "net_xg": vals.get("net_xg"),
            "cs_pct_overall": vals.get("cs_pct_overall"),
            "goals_first_half_pct": vals.get("goals_first_half_pct"),
            "goals_second_half_pct": vals.get("goals_second_half_pct"),
            "home_advantage_xg": vals.get("home_advantage_xg"),
        })
    rows.sort(key=lambda r: r.get("net_xg") or -999, reverse=True)
    return rows


def team_dna_narrative(team_name: str) -> str | None:
    """Kort menneskelig beskrivelse av lagets DNA basert på empiriske tall."""
    t = get_team(team_name)
    if not t:
        return None
    parts = []
    net_xg = t.get("net_xg")
    if net_xg is not None:
        if net_xg >= 0.5:
            parts.append(f"Sterkt offensivt lag (net xG {net_xg:+.1f})")
        elif net_xg <= -0.3:
            parts.append(f"Sliter foran og bak (net xG {net_xg:+.1f})")
        else:
            parts.append(f"Balansert profil (net xG {net_xg:+.1f})")
    cs = t.get("cs_pct_overall")
    if cs is not None:
        if cs >= 0.40:
            parts.append(f"sterkt forsvar — clean sheet i {int(cs*100)}% av kampene")
        elif cs <= 0.20:
            parts.append(f"svakt forsvar — clean sheet i bare {int(cs*100)}% av kampene")
    second = t.get("goals_second_half_pct")
    first = t.get("goals_first_half_pct")
    if second is not None and first is not None:
        if second >= 0.60:
            parts.append(f"scorer {int(second*100)}% av målene etter pause — sent-blomstrende")
        elif first >= 0.55:
            parts.append(f"scorer {int(first*100)}% av målene før pause — kommer fort i gang")
    home_adv = t.get("home_advantage_xg")
    if home_adv is not None and home_adv >= 0.4:
        parts.append(f"klar hjemmebanefordel (+{home_adv:.1f} xG hjemme vs borte)")
    elif home_adv is not None and home_adv <= -0.2:
        parts.append("overraskende sterkere på borte enn hjemme")
    if not parts:
        return None
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."
