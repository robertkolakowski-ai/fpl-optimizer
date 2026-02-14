# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPL Optimizer — a Fantasy Premier League squad optimization tool with both a CLI and a Flask web UI. Uses linear programming (PuLP/CBC) to build optimal 15-man squads under FPL budget and position constraints, with player scoring based on xG, xA, BPS, ICT, fixture difficulty, and form.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI
python run.py

# Run web app locally (Flask dev server)
python -m flask --app fpl_optimizer.web run --debug

# Run web app via gunicorn (production / Render)
gunicorn fpl_optimizer.web:app --bind 0.0.0.0:8000
```

There are no tests, linter configs, or CI pipelines yet.

## Architecture

**Data flow:** FPL API → `api.py` (fetch & parse) → `models.py` (dataclasses) → `analyzer.py` (scoring) → `optimizer.py` (LP solver) → output (CLI or web)

### Key modules (`fpl_optimizer/`)

- **api.py** — All FPL API calls via `httpx`. Returns typed dataclass lists. `load_data()` is the main entry point that fetches players, teams, gameweeks, and fixtures.
- **models.py** — Dataclasses: `Player`, `Team`, `Fixture`, `Gameweek`, `Squad`. All use `from __future__ import annotations` for forward refs.
- **analyzer.py** — `score_players()` computes `composite_score` per player using position-specific weights across form, xG/xA, BPS, ICT, fixture difficulty, and clean sheet probability. Also handles rotation risk and fixture schedule analysis.
- **optimizer.py** — `select_squad()` solves a binary LP (PuLP CBC) maximizing total composite score subject to: 15-player squad, position limits (2 GK / 5 DEF / 5 MID / 3 FWD), max 3 per team, budget ≤ 100.0, and valid starting XI formation.
- **transfers.py** — `suggest_transfers()` recommends sells/buys by comparing current squad to optimal.
- **multi_gw.py** — Multi-gameweek transfer planning with chip strategy (Bench Boost, Triple Captain, Free Hit, Wildcard).
- **web.py** — Flask app with ~20 JSON API endpoints (`/api/squad`, `/api/players`, `/api/live`, `/api/fdr`, etc.) and an in-memory cache (5 min TTL). Single-page web UI served from `templates/web.html`.
- **cli.py** — Rich console interface. Entry point: `run.py`.
- **report_console.py / report_html.py** — Output formatters for CLI reports.

### Entry points

- `run.py` → CLI (`fpl_optimizer.cli:main`)
- `app.py` → Web deployment (imports `fpl_optimizer.web:app` for gunicorn)

### Deployment

Deployed on Render (see `render.yaml`). Python 3.11, gunicorn WSGI server.

## Conventions

- Python 3.11+ with `from __future__ import annotations` in all modules
- Dataclasses for all data models, with `to_dict()` via `dataclasses.asdict`
- Position IDs: 1=GK, 2=DEF, 3=MID, 4=FWD (matches FPL API)
- All FPL API data goes through `api.py` — never call the API directly from other modules
- Web endpoints return JSON; the single-page frontend calls them via fetch
