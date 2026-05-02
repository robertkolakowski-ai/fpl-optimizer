"""Web Push Notifications via VAPID.

Filosofi: minimum-friksjon, maks-verdi. Vi sender bare 3 typer varsler:
  1. Deadline-påminnelse fredag morgen (24t før kickoff)
  2. Skadevarsel når en spiller på laget får sin chance_of_playing < 75
  3. Prisendring nattetime hvis spiller du følger har endret pris

VAPID-nøklene genereres én gang og lagres som env-vars (VAPID_PRIVATE_KEY,
VAPID_PUBLIC_KEY, VAPID_SUBJECT). Hvis ingen er satt, fungerer alt ellers
som vanlig — push er bare disabled.

Subscriptions lagres i SQLite (data/push_subs.db) keyed by FPL team_id +
endpoint. Bruker `pywebpush` for selve sendingen.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _data_dir() -> Path:
    base = os.environ.get("FPL_DATA_DIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent.parent / "data"


def _db_path() -> Path:
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "push_subs.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subscriptions (
            team_id INTEGER NOT NULL,
            endpoint TEXT NOT NULL,
            keys_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, endpoint)
        )
    """)
    return conn


def get_public_key() -> str | None:
    """VAPID public key for client-side subscribe. None = push disabled."""
    return os.environ.get("VAPID_PUBLIC_KEY") or None


def is_enabled() -> bool:
    return bool(os.environ.get("VAPID_PUBLIC_KEY")) and bool(os.environ.get("VAPID_PRIVATE_KEY"))


def add_subscription(team_id: int, subscription: dict[str, Any]) -> None:
    """Lagre push-subscription for en bruker. Idempotent på (team_id, endpoint)."""
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys", {})
    if not endpoint or not keys:
        raise ValueError("Invalid subscription: missing endpoint or keys")
    with _LOCK, _conn() as conn:
        conn.execute(
            "INSERT INTO subscriptions (team_id, endpoint, keys_json) VALUES (?, ?, ?) "
            "ON CONFLICT(team_id, endpoint) DO UPDATE SET keys_json = excluded.keys_json",
            (team_id, endpoint, json.dumps(keys)),
        )


def remove_subscription(team_id: int, endpoint: str) -> None:
    with _LOCK, _conn() as conn:
        conn.execute(
            "DELETE FROM subscriptions WHERE team_id = ? AND endpoint = ?",
            (team_id, endpoint),
        )


def list_subscriptions(team_id: int | None = None) -> list[dict[str, Any]]:
    with _LOCK, _conn() as conn:
        if team_id is not None:
            rows = conn.execute(
                "SELECT team_id, endpoint, keys_json FROM subscriptions WHERE team_id = ?",
                (team_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT team_id, endpoint, keys_json FROM subscriptions"
            ).fetchall()
    return [
        {"team_id": r["team_id"], "endpoint": r["endpoint"], "keys": json.loads(r["keys_json"])}
        for r in rows
    ]


def send_push(subscription: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Send én push. Returnerer True ved success, False ved feil.

    Stille fallback hvis pywebpush ikke er installert eller VAPID-nøkler mangler.
    """
    if not is_enabled():
        return False
    try:
        from pywebpush import webpush, WebPushException  # type: ignore
    except ImportError:
        return False

    private_key = os.environ.get("VAPID_PRIVATE_KEY")
    subject = os.environ.get("VAPID_SUBJECT", "mailto:robert@kolakowski.no")
    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=private_key,
            vapid_claims={"sub": subject},
            ttl=60 * 60 * 24,  # 24t — varsler bør leveres innen 1 dag
        )
        return True
    except Exception:
        # Push kan feile av mange grunner (gone-410, expired-401, etc).
        # Vi catch'er alt for å ikke crashe ved en enkelt død sub.
        return False


def broadcast_to_team(team_id: int, payload: dict[str, Any]) -> dict[str, int]:
    """Send samme payload til alle subs for en team_id. Returnerer counts."""
    subs = list_subscriptions(team_id)
    sent = failed = 0
    dead_endpoints = []
    for sub in subs:
        info = {"endpoint": sub["endpoint"], "keys": sub["keys"]}
        ok = send_push(info, payload)
        if ok:
            sent += 1
        else:
            failed += 1
            dead_endpoints.append(sub["endpoint"])
    # Rydd opp endepunkter som returnerer 410 Gone (kan utvides senere med
    # detaljert feilkategorisering)
    return {"sent": sent, "failed": failed}
