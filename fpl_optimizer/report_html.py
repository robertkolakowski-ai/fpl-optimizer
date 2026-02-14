from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .models import Player, Squad
from .transfers import TransferSuggestion

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_html(
    squad: Squad,
    teams: dict,
    all_players: list[Player],
    transfers: list[TransferSuggestion] | None = None,
    output_path: str = "report.html",
) -> None:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("report.html")

    # Top 10 per position
    pos_names = {1: "Goalkeepers", 2: "Defenders", 3: "Midfielders", 4: "Forwards"}
    top_by_position = {}
    for pos, name in pos_names.items():
        ranked = sorted(
            [p for p in all_players if p.position == pos],
            key=lambda p: p.composite_score,
            reverse=True,
        )[:10]
        top_by_position[name] = ranked

    html = template.render(
        squad=squad,
        teams=teams,
        transfers=transfers or [],
        top_by_position=top_by_position,
        total_score=sum(p.composite_score for p in squad.players),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    Path(output_path).write_text(html, encoding="utf-8")
