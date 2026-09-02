"""Crawl progress: which users are left to expand, which leagues are left to fetch.

This is deliberately not Parquet. The corpus in ``iie.crawl.store`` is append-only
facts; the frontier is mutable bookkeeping that gets rewritten constantly -- a
league moves from discovered to accepted to fetched, a user goes from pending to
expanded. SQLite gives that plus crash-safe resume in a single file, and it makes
the visited set a primary key constraint rather than an in-memory set that dies
with the process.

Resumability is the whole point. A corpus crawl runs for tens of minutes to hours
and will be interrupted. Combined with the HTTP cache, a restart re-reads state
from disk and re-fetches nothing, so an interrupted crawl costs seconds rather
than the whole run.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Self

log = logging.getLogger(__name__)

# Users are pending until their leagues have been listed; leagues move
# discovered -> accepted -> fetched, or terminate at rejected with a reason.
USER_STATES = ("pending", "expanded")
LEAGUE_STATES = ("discovered", "accepted", "fetched", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    state         TEXT NOT NULL DEFAULT 'pending',
    depth         INTEGER NOT NULL DEFAULT 0,
    discovered_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS users_state ON users (state, depth);

CREATE TABLE IF NOT EXISTS leagues (
    league_id     TEXT PRIMARY KEY,
    season        TEXT,
    state         TEXT NOT NULL DEFAULT 'discovered',
    reason        TEXT,
    depth         INTEGER NOT NULL DEFAULT 0,
    discovered_at REAL NOT NULL,
    updated_at    REAL
);
CREATE INDEX IF NOT EXISTS leagues_state ON leagues (state, depth);
"""


class Frontier:
    """Persistent crawl state. Safe to reopen on the same file to resume."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        # WAL survives a hard kill mid-write without corrupting the file, which is
        # exactly the failure mode a long crawl invites.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # --- users ------------------------------------------------------------

    def add_users(self, user_ids: Iterable[str], depth: int = 0) -> int:
        """Queue users for expansion. Returns how many were newly added.

        Re-adding a known user is a no-op rather than an error: the same person
        turns up in many leagues, and that is the normal case, not a conflict.
        """
        rows = [(str(u), depth, time.time()) for u in user_ids if u]
        if not rows:
            return 0
        before = self._count("users")
        self.db.executemany(
            "INSERT OR IGNORE INTO users (user_id, depth, discovered_at) VALUES (?, ?, ?)", rows
        )
        self.db.commit()
        return self._count("users") - before

    def next_users(self, limit: int = 1) -> list[sqlite3.Row]:
        """Shallowest-first, so the crawl stays near its seeds before wandering."""
        return self.db.execute(
            "SELECT user_id, depth FROM users WHERE state = 'pending' ORDER BY depth, rowid LIMIT ?",
            (limit,),
        ).fetchall()

    def mark_user_expanded(self, user_id: str) -> None:
        self.db.execute("UPDATE users SET state = 'expanded' WHERE user_id = ?", (str(user_id),))
        self.db.commit()

    # --- leagues ----------------------------------------------------------

    def add_league(
        self,
        league_id: str,
        season: str | None,
        *,
        state: str = "discovered",
        reason: str | None = None,
        depth: int = 0,
    ) -> bool:
        """Record a league. Returns False if it was already known."""
        if state not in LEAGUE_STATES:
            raise ValueError(f"bad league state {state!r}; expected one of {LEAGUE_STATES}")
        now = time.time()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO leagues"
            " (league_id, season, state, reason, depth, discovered_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(league_id), season, state, reason, depth, now, now),
        )
        self.db.commit()
        return cur.rowcount > 0

    def next_leagues(self, state: str = "accepted", limit: int = 1) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT league_id, season, depth FROM leagues WHERE state = ? ORDER BY depth, rowid LIMIT ?",
            (state, limit),
        ).fetchall()

    def set_league_state(
        self,
        league_id: str,
        state: str,
        reason: str | None = None,
        season: str | None = None,
    ) -> None:
        """Update a league's state, and its season if it was not known at discovery.

        A league reached through ``previous_league_id`` is queued before anything
        about it is known, season included. Backfilling it here is what keeps its
        transactions labelled with a season they can be joined on.
        """
        if state not in LEAGUE_STATES:
            raise ValueError(f"bad league state {state!r}; expected one of {LEAGUE_STATES}")
        if season is None:
            self.db.execute(
                "UPDATE leagues SET state = ?, reason = ?, updated_at = ? WHERE league_id = ?",
                (state, reason, time.time(), str(league_id)),
            )
        else:
            self.db.execute(
                "UPDATE leagues SET state = ?, reason = ?, season = ?, updated_at = ?"
                " WHERE league_id = ?",
                (state, reason, str(season), time.time(), str(league_id)),
            )
        self.db.commit()

    def knows_league(self, league_id: str) -> bool:
        return (
            self.db.execute(
                "SELECT 1 FROM leagues WHERE league_id = ?", (str(league_id),)
            ).fetchone()
            is not None
        )

    # --- progress ---------------------------------------------------------

    def _count(self, table: str) -> int:
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def counts(self) -> dict[str, int]:
        """Progress summary: one key per user state and league state."""
        out = {"users_total": self._count("users"), "leagues_total": self._count("leagues")}
        for state in USER_STATES:
            out[f"users_{state}"] = self.db.execute(
                "SELECT COUNT(*) FROM users WHERE state = ?", (state,)
            ).fetchone()[0]
        for state in LEAGUE_STATES:
            out[f"leagues_{state}"] = self.db.execute(
                "SELECT COUNT(*) FROM leagues WHERE state = ?", (state,)
            ).fetchone()[0]
        return out

    def rejection_reasons(self) -> dict[str, int]:
        """Why leagues were dropped, so a filter that is too aggressive is visible."""
        rows = self.db.execute(
            "SELECT reason, COUNT(*) AS n FROM leagues WHERE state = 'rejected'"
            " GROUP BY reason ORDER BY n DESC"
        ).fetchall()
        return {row["reason"] or "unknown": row["n"] for row in rows}
