"""
Lead-capture "tool" the agent calls once it has collected name + email + platform.

I originally wrote this as a one-line print() just to get the LangGraph tool-calling
flow working end to end. Once the routing/extraction logic was solid I came back and
gave it a real backing store (SQLite) so leads actually persist somewhere instead of
disappearing the moment the process exits. SQLite needs no server and ships with
Python, which made it the right choice for a project this size — swapping the
`_connect()` function for a Postgres/Mongo client later wouldn't touch any other file.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "leads.db"


def _mask_email(email):
    """user@example.com -> u***@example.com, for anything that hits stdout/logs."""
    local, _, domain = email.partition("@")
    if not domain or not local:
        return "***"
    return f"{local[0]}***@{domain}"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            platform TEXT NOT NULL,
            captured_at TEXT NOT NULL
        )
        """
    )
    return conn


def mock_lead_capture(name, email, platform):
    """
    Persist a captured lead to leads.db and return the stored row as a dict.

    Kept the original function name/signature (agent.py calls it unchanged) even
    though it's not "mock" anymore — this is a placeholder for whatever real CRM
    or webhook call would replace it in a production deployment (see README).
    """
    captured_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO leads (name, email, platform, captured_at) VALUES (?, ?, ?, ?)",
            (name, email, platform, captured_at),
        )
        lead_id = cursor.lastrowid

    # The full email is fine to store (it's the whole point of a lead), but
    # console/log output is easy to forget about, ship to a log aggregator,
    # or screenshot — so what appears there is masked, not the raw PII.
    logger.info(
        "Lead captured: id=%s name=%s email=%s platform=%s",
        lead_id, name, _mask_email(email), platform,
    )

    return {
        "id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": captured_at,
    }


def list_leads():
    """Return all captured leads, most recent first. Handy for a quick sanity check."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, email, platform, captured_at FROM leads ORDER BY id DESC"
        ).fetchall()

    return [
        {"id": r[0], "name": r[1], "email": r[2], "platform": r[3], "captured_at": r[4]}
        for r in rows
    ]


if __name__ == "__main__":
    # Quick manual check: `python tools.py` prints whatever's in leads.db so far.
    for lead in list_leads():
        print(lead)
