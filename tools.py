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
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
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
    # Safe migration: remove legacy duplicate records keeping earliest ID before creating unique index
    conn.execute(
        """
        DELETE FROM leads
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM leads
            GROUP BY lower(trim(email))
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_leads_email_unique ON leads(lower(trim(email)));")
    return conn


def mock_lead_capture(name, email, platform):
    """
    Persist a captured lead to leads.db and return the stored row as a dict.

    Normalizes email addresses and enforces database-level deduplication: if a lead with
    the same normalized email already exists, returns the existing record without inserting
    a duplicate row.
    """
    clean_name = str(name).strip()
    clean_email = str(email).strip().lower()
    clean_platform = str(platform).strip()
    captured_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        # Check for an existing lead with the same normalized email
        existing = conn.execute(
            "SELECT id, name, email, platform, captured_at FROM leads WHERE lower(trim(email)) = ? LIMIT 1",
            (clean_email,),
        ).fetchone()

        if existing:
            logger.info(
                "Lead deduplicated: existing id=%s name=%s email=%s",
                existing[0], existing[1], _mask_email(clean_email),
            )
            return {
                "id": existing[0],
                "name": existing[1],
                "email": existing[2],
                "platform": existing[3],
                "captured_at": existing[4],
                "is_duplicate": True,
            }

        try:
            cursor = conn.execute(
                "INSERT INTO leads (name, email, platform, captured_at) VALUES (?, ?, ?, ?)",
                (clean_name, clean_email, clean_platform, captured_at),
            )
            lead_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # Concurrent insert race condition caught by DB unique constraint
            existing = conn.execute(
                "SELECT id, name, email, platform, captured_at FROM leads WHERE lower(trim(email)) = ? LIMIT 1",
                (clean_email,),
            ).fetchone()
            if existing:
                return {
                    "id": existing[0],
                    "name": existing[1],
                    "email": existing[2],
                    "platform": existing[3],
                    "captured_at": existing[4],
                    "is_duplicate": True,
                }
            raise

    logger.info(
        "Lead captured: id=%s name=%s email=%s platform=%s",
        lead_id, clean_name, _mask_email(clean_email), clean_platform,
    )

    return {
        "id": lead_id,
        "name": clean_name,
        "email": clean_email,
        "platform": clean_platform,
        "captured_at": captured_at,
        "is_duplicate": False,
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
