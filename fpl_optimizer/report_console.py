from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .models import Squad
from .transfers import TransferSuggestion


def _player_table(title: str, players: list, teams: dict) -> Table:
    table = Table(title=title, show_lines=False)
    table.add_column("Pos", style="bold cyan", width=4)
    table.add_column("Player", style="bold white", min_width=14)
    table.add_column("Team", width=5)
    table.add_column("Cost", justify="right", width=5)
    table.add_column("Form", justify="right", width=5)
    table.add_column("PPG", justify="right", width=5)
    table.add_column("xG", justify="right", width=5)
    table.add_column("xA", justify="right", width=5)
    table.add_column("Score", justify="right", style="bold green", width=6)

    for p in players:
        team_name = teams.get(p.team)
        short = team_name.short_name if team_name else "?"
        table.add_row(
            p.position_name,
            p.name,
            short,
            f"{p.cost:.1f}",
            f"{p.form:.1f}",
            f"{p.points_per_game:.1f}",
            f"{p.xG:.1f}",
            f"{p.xA:.1f}",
            f"{p.composite_score:.3f}",
        )
    return table


def print_report(
    squad: Squad,
    teams: dict,
    transfers: list[TransferSuggestion] | None = None,
) -> None:
    console = Console()

    console.print()
    console.rule("[bold blue]FPL Optimizer Report[/bold blue]")
    console.print()

    # Starting XI
    console.print(_player_table("Starting XI", squad.starting, teams))
    console.print()

    # Bench
    console.print(_player_table("Bench", squad.bench, teams))
    console.print()

    # Summary
    console.print(f"[bold]Total cost:[/bold] {squad.total_cost:.1f}M")
    console.print(f"[bold]Budget remaining:[/bold] {squad.budget_remaining:.1f}M")
    console.print(
        f"[bold]Total score:[/bold] {sum(p.composite_score for p in squad.players):.3f}"
    )
    console.print()

    # Transfers
    if transfers:
        t_table = Table(title="Transfer Suggestions", show_lines=False)
        t_table.add_column("#", width=3)
        t_table.add_column("Out", style="red", min_width=12)
        t_table.add_column("In", style="green", min_width=12)
        t_table.add_column("Gain", justify="right", width=7)
        t_table.add_column("Cost Δ", justify="right", width=7)

        for i, s in enumerate(transfers, 1):
            t_table.add_row(
                str(i),
                f"{s.player_out.name} ({s.player_out.cost:.1f})",
                f"{s.player_in.name} ({s.player_in.cost:.1f})",
                f"+{s.score_gain:.3f}",
                f"{s.cost_change:+.1f}",
            )

        console.print(t_table)
        console.print()
