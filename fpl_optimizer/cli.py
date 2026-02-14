from __future__ import annotations

import argparse
import sys

from .analyzer import score_players
from .api import load_data
from .optimizer import select_squad
from .report_console import print_report
from .report_html import generate_html
from .transfers import suggest_transfers


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FPL Optimizer - Find the optimal Fantasy Premier League squad"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total budget in millions (default: 100.0)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=5,
        help="Number of gameweeks to look ahead for fixture difficulty (default: 5)",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        metavar="FILE",
        help="Generate HTML report to FILE",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Suppress console output",
    )
    parser.add_argument(
        "--transfers",
        action="store_true",
        help="Show transfer suggestions",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the web interface instead of CLI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web server (default: 5000)",
    )

    args = parser.parse_args(argv)

    if args.web:
        from .web import app

        print(f"Starting FPL Optimizer web interface on http://localhost:{args.port}")
        app.run(debug=True, port=args.port)
        return

    # Fetch data
    print("Fetching FPL data...")
    try:
        players, teams, gameweeks, fixtures = load_data()
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(players)} players, {len(fixtures)} fixtures")

    # Score players
    score_players(players, fixtures, gameweeks, lookahead=args.lookahead)

    # Optimize squad
    print("Optimizing squad...")
    squad = select_squad(players, budget=args.budget)

    # Transfer suggestions
    transfer_list = None
    if args.transfers:
        transfer_list = suggest_transfers(squad, players, budget=args.budget)

    # Console report
    if not args.no_console:
        print_report(squad, teams, transfers=transfer_list)

    # HTML report
    if args.html:
        generate_html(
            squad, teams, players, transfers=transfer_list, output_path=args.html
        )
        print(f"HTML report saved to {args.html}")
