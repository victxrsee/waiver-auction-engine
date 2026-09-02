"""Normalization and persistence for the crawled Sleeper corpus.

The client in ``iie.sources.sleeper`` returns raw API shapes. This module is the
only place that decides what those shapes become on disk, which is two decisions:

**What is kept.** The engine we are building reasons about roster construction and
waiver behavior, not about people. So user IDs are kept -- they are the join key
that makes the league graph a graph -- and every free-text or identifying field
is dropped at the boundary: display names, avatars, team names, league names.
Dropping them here rather than at analysis time means a leaked corpus directory
is still just anonymous IDs and transactions.

**What shape it takes.** Nested API payloads are flattened into long tables
(``roster_players``, ``transaction_players``) instead of stored as blobs. A
waiver engine's central question -- who added whom, for how much, in what week --
is a row-per-player question, and answering it with a Parquet scan beats
re-parsing JSON on every run. Genuinely dict-shaped config (scoring settings,
league settings) stays as a JSON string in one column: lossless, schema-stable,
and rarely queried.

Writes are buffered and flushed to part files under ``<root>/<table>/``. Records
are immutable facts, so a re-crawl that writes a row twice is harmless --
``read_table`` de-duplicates on the primary key.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

DEFAULT_BUFFER_ROWS = 10_000


# --- schemas --------------------------------------------------------------
#
# Declared explicitly rather than inferred. Inference reads a column that happens
# to be all-null in one buffer as a different type than the same column in the
# next, and the resulting corpus will not load as a single dataset.

SCHEMAS: dict[str, pa.Schema] = {
    "leagues": pa.schema(
        [
            ("league_id", pa.string()),
            ("season", pa.string()),
            ("sport", pa.string()),
            ("status", pa.string()),
            ("total_rosters", pa.int32()),
            ("previous_league_id", pa.string()),
            ("draft_id", pa.string()),
            ("roster_positions", pa.list_(pa.string())),
            ("scoring_settings_json", pa.string()),
            ("settings_json", pa.string()),
        ]
    ),
    "league_members": pa.schema(
        [
            ("league_id", pa.string()),
            ("user_id", pa.string()),
            ("is_owner", pa.bool_()),
        ]
    ),
    "rosters": pa.schema(
        [
            ("league_id", pa.string()),
            ("roster_id", pa.int32()),
            ("owner_id", pa.string()),
            ("wins", pa.int32()),
            ("losses", pa.int32()),
            ("ties", pa.int32()),
            ("points_for", pa.float64()),
            ("points_against", pa.float64()),
            ("waiver_budget_used", pa.int32()),
            ("waiver_position", pa.int32()),
        ]
    ),
    "roster_players": pa.schema(
        [
            ("league_id", pa.string()),
            ("roster_id", pa.int32()),
            ("player_id", pa.string()),
            ("is_starter", pa.bool_()),
        ]
    ),
    "transactions": pa.schema(
        [
            ("transaction_id", pa.string()),
            ("league_id", pa.string()),
            ("season", pa.string()),
            ("week", pa.int32()),
            ("type", pa.string()),
            ("status", pa.string()),
            ("created_ms", pa.int64()),
            ("waiver_bid", pa.int32()),
            ("roster_ids", pa.list_(pa.int32())),
        ]
    ),
    "transaction_players": pa.schema(
        [
            ("transaction_id", pa.string()),
            ("league_id", pa.string()),
            ("season", pa.string()),
            ("week", pa.int32()),
            ("player_id", pa.string()),
            ("roster_id", pa.int32()),
            ("action", pa.string()),  # "add" | "drop"
            ("waiver_bid", pa.int32()),
        ]
    ),
    "draft_picks": pa.schema(
        [
            ("draft_id", pa.string()),
            ("league_id", pa.string()),
            ("season", pa.string()),
            ("pick_no", pa.int32()),
            ("round", pa.int32()),
            ("draft_slot", pa.int32()),
            ("roster_id", pa.int32()),
            ("player_id", pa.string()),
            ("is_keeper", pa.bool_()),
        ]
    ),
}

# Used by read_table to collapse rows a re-crawl wrote more than once.
PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "leagues": ("league_id",),
    "league_members": ("league_id", "user_id"),
    "rosters": ("league_id", "roster_id"),
    "roster_players": ("league_id", "roster_id", "player_id"),
    "transactions": ("transaction_id",),
    "transaction_players": ("transaction_id", "player_id", "action"),
    "draft_picks": ("draft_id", "pick_no"),
}


# --- normalization --------------------------------------------------------
#
# Each function takes a raw API payload and returns rows matching a schema above.
# They are pure and defensive: Sleeper omits fields freely on older seasons, so
# every lookup tolerates absence rather than assuming a key exists.


def _as_int(value: Any) -> int | None:
    """Coerce to int, mapping absent/blank/non-numeric to null rather than raising."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def normalize_league(raw: dict[str, Any]) -> dict[str, Any]:
    settings = raw.get("settings") or {}
    return {
        "league_id": _as_str(raw.get("league_id")),
        "season": _as_str(raw.get("season")),
        "sport": _as_str(raw.get("sport")),
        "status": _as_str(raw.get("status")),
        "total_rosters": _as_int(raw.get("total_rosters")),
        "previous_league_id": _as_str(raw.get("previous_league_id")),
        "draft_id": _as_str(raw.get("draft_id")),
        "roster_positions": [str(p) for p in (raw.get("roster_positions") or [])],
        "scoring_settings_json": json.dumps(raw.get("scoring_settings") or {}, sort_keys=True),
        "settings_json": json.dumps(settings, sort_keys=True),
    }


def normalize_league_members(
    league_id: str, raw_users: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """League membership edges. Display name, avatar and team name are dropped here."""
    return [
        {
            "league_id": league_id,
            "user_id": _as_str(u.get("user_id")),
            "is_owner": bool(u.get("is_owner")),
        }
        for u in raw_users
        if u.get("user_id")
    ]


def _roster_points(settings: dict[str, Any], whole_key: str, decimal_key: str) -> float | None:
    """Sleeper splits scores across an integer and a hundredths field."""
    whole = _as_int(settings.get(whole_key))
    if whole is None:
        return None
    return whole + (_as_int(settings.get(decimal_key)) or 0) / 100.0


def normalize_roster(league_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    settings = raw.get("settings") or {}
    return {
        "league_id": league_id,
        "roster_id": _as_int(raw.get("roster_id")),
        "owner_id": _as_str(raw.get("owner_id")),
        "wins": _as_int(settings.get("wins")),
        "losses": _as_int(settings.get("losses")),
        "ties": _as_int(settings.get("ties")),
        "points_for": _roster_points(settings, "fpts", "fpts_decimal"),
        "points_against": _roster_points(settings, "fpts_against", "fpts_against_decimal"),
        "waiver_budget_used": _as_int(settings.get("waiver_budget_used")),
        "waiver_position": _as_int(settings.get("waiver_position")),
    }


def normalize_roster_players(league_id: str, raw: dict[str, Any]) -> list[dict[str, Any]]:
    roster_id = _as_int(raw.get("roster_id"))
    starters = {str(p) for p in (raw.get("starters") or []) if p}
    return [
        {
            "league_id": league_id,
            "roster_id": roster_id,
            "player_id": str(player_id),
            "is_starter": str(player_id) in starters,
        }
        for player_id in (raw.get("players") or [])
        if player_id
    ]


def normalize_transaction(
    league_id: str, season: str, week: int, raw: dict[str, Any]
) -> dict[str, Any]:
    settings = raw.get("settings") or {}
    return {
        "transaction_id": _as_str(raw.get("transaction_id")),
        "league_id": league_id,
        "season": season,
        "week": week,
        "type": _as_str(raw.get("type")),
        "status": _as_str(raw.get("status")),
        "created_ms": _as_int(raw.get("created")),
        "waiver_bid": _as_int(settings.get("waiver_bid")),
        "roster_ids": [
            i for i in (_as_int(r) for r in (raw.get("roster_ids") or [])) if i is not None
        ],
    }


def normalize_transaction_players(
    league_id: str, season: str, week: int, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per player moved.

    ``adds``/``drops`` are ``{player_id: roster_id}`` maps, and either may be null
    on a transaction that only moves draft picks or FAAB.
    """
    settings = raw.get("settings") or {}
    bid = _as_int(settings.get("waiver_bid"))
    txn_id = _as_str(raw.get("transaction_id"))
    rows: list[dict[str, Any]] = []
    for action in ("add", "drop"):
        for player_id, roster_id in (raw.get(f"{action}s") or {}).items():
            rows.append(
                {
                    "transaction_id": txn_id,
                    "league_id": league_id,
                    "season": season,
                    "week": week,
                    "player_id": str(player_id),
                    "roster_id": _as_int(roster_id),
                    "action": action,
                    # Carried onto the add row so bid-vs-outcome needs no join back.
                    "waiver_bid": bid if action == "add" else None,
                }
            )
    return rows


def normalize_draft_picks(
    draft_id: str, league_id: str, season: str, raw_picks: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "draft_id": draft_id,
            "league_id": league_id,
            "season": season,
            "pick_no": _as_int(p.get("pick_no")),
            "round": _as_int(p.get("round")),
            "draft_slot": _as_int(p.get("draft_slot")),
            "roster_id": _as_int(p.get("roster_id")),
            "player_id": _as_str(p.get("player_id")),
            "is_keeper": bool(p.get("is_keeper")),
        }
        for p in raw_picks
    ]


# --- persistence ----------------------------------------------------------


class _TableWriter:
    """Buffers rows for one table and flushes them to Parquet part files."""

    def __init__(self, directory: Path, schema: pa.Schema, buffer_rows: int) -> None:
        self.directory = directory
        self.schema = schema
        self.buffer_rows = buffer_rows
        self._buffer: list[dict[str, Any]] = []
        self.rows_written = 0

    def add(self, rows: Iterable[dict[str, Any]]) -> None:
        self._buffer.extend(rows)
        if len(self._buffer) >= self.buffer_rows:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        # Written under a temp name and renamed, so a crawl killed mid-flush never
        # leaves a truncated file that poisons the whole dataset on read.
        final = self.directory / f"part-{uuid.uuid4().hex}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(final)
        self.rows_written += len(self._buffer)
        self._buffer.clear()


class CorpusStore:
    """Write side of the corpus. Use as a context manager so buffers always flush.

        with CorpusStore("data/corpus") as store:
            store.add_league(raw_league)

    Tables land in ``<root>/<table>/part-*.parquet``.
    """

    def __init__(self, root: Path | str, *, buffer_rows: int = DEFAULT_BUFFER_ROWS) -> None:
        self.root = Path(root)
        self._writers = {
            name: _TableWriter(self.root / name, schema, buffer_rows)
            for name, schema in SCHEMAS.items()
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        for writer in self._writers.values():
            writer.flush()
        log.info("corpus written to %s: %s", self.root, self.stats)

    @property
    def stats(self) -> dict[str, int]:
        return {name: w.rows_written for name, w in self._writers.items() if w.rows_written}

    # -- one call per API payload, so the crawler never touches a schema ----

    def add_league(self, raw_league: dict[str, Any]) -> None:
        self._writers["leagues"].add([normalize_league(raw_league)])

    def add_league_users(self, league_id: str, raw_users: Iterable[dict[str, Any]]) -> None:
        self._writers["league_members"].add(normalize_league_members(league_id, raw_users))

    def add_rosters(self, league_id: str, raw_rosters: Iterable[dict[str, Any]]) -> None:
        for raw in raw_rosters:
            self._writers["rosters"].add([normalize_roster(league_id, raw)])
            self._writers["roster_players"].add(normalize_roster_players(league_id, raw))

    def add_transactions(
        self, league_id: str, season: str, week: int, raw_transactions: Iterable[dict[str, Any]]
    ) -> None:
        for raw in raw_transactions:
            self._writers["transactions"].add([normalize_transaction(league_id, season, week, raw)])
            self._writers["transaction_players"].add(
                normalize_transaction_players(league_id, season, week, raw)
            )

    def add_draft_picks(
        self, draft_id: str, league_id: str, season: str, raw_picks: Iterable[dict[str, Any]]
    ) -> None:
        self._writers["draft_picks"].add(
            normalize_draft_picks(draft_id, league_id, season, raw_picks)
        )


# --- read side ------------------------------------------------------------


def read_table(root: Path | str, name: str, *, dedupe: bool = True):
    """Load one table as a DataFrame, collapsing rows a re-crawl duplicated.

    Returns an empty, correctly-typed frame when the table was never written, so
    callers can analyze a partial corpus without special-casing.
    """
    if name not in SCHEMAS:
        raise KeyError(f"unknown table {name!r}; expected one of {sorted(SCHEMAS)}")

    directory = Path(root) / name
    if not directory.exists() or not any(directory.glob("*.parquet")):
        return SCHEMAS[name].empty_table().to_pandas()

    frame = pq.read_table(directory, schema=SCHEMAS[name]).to_pandas()
    if dedupe:
        frame = frame.drop_duplicates(subset=list(PRIMARY_KEYS[name]))
    return frame.reset_index(drop=True)
