"""Player-week panel: the trailing and forward views a claim is scored against.

Two nflverse id spaces have to be reconciled first. ``player_week`` keys on
``player_id``, which is already the gsis id, so it joins directly. ``snap_counts``
keys on ``pfr_player_id``, a different space entirely, and reaches gsis only
through the ``players`` table -- which covers 99.7% of the snap-count ids.

Everything is computed on a **complete week grid** rather than on the rows that
happen to exist. A player on bye has no stat row, so a three-row rolling window
would silently span four calendar weeks and mix in older evidence. Reindexing to
every week first makes the window mean what it says.

Trailing aggregates are inclusive of their week and forward aggregates are strictly
exclusive of it, which is what keeps the as-of boundary intact: a claim resolved
against week ``W`` sees trailing stats through ``W`` and outcomes from ``W+1`` on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Volume stats a claim is judged on. Absent weeks are real zeros -- a player who
# did not play produced nothing -- rather than missing data.
VOLUME_COLUMNS = ("fantasy_points", "receptions", "targets", "carries", "receiving_yards")

MAX_WEEK = 18


def build_player_week(
    cache,
    seasons: list[int],
    *,
    gsis_ids: set[str] | None = None,
) -> pd.DataFrame:
    """One row per (gsis_id, season, week) with production and snap share."""
    stats = cache.load_seasons(
        "player_week",
        seasons,
        columns=["player_id", "season", "week", "season_type", "position", *VOLUME_COLUMNS],
    )
    if stats.empty:
        return stats

    stats = stats[stats["season_type"] == "REG"].rename(columns={"player_id": "gsis_id"})
    if gsis_ids is not None:
        stats = stats[stats["gsis_id"].isin(gsis_ids)]

    snaps = _snap_share(cache, seasons)
    if not snaps.empty:
        stats = stats.merge(snaps, on=["gsis_id", "season", "week"], how="left")
    else:
        # Explicitly float: pd.NA would make this an object column, which rolling
        # aggregation rejects. Snap counts are simply absent before 2012 and for
        # the live season until nflverse publishes them.
        stats["snap_pct"] = np.nan

    return stats.reset_index(drop=True)


def _snap_share(cache, seasons: list[int]) -> pd.DataFrame:
    """Snap counts, translated from pfr ids into gsis ids."""
    snaps = cache.load_seasons(
        "snap_counts",
        seasons,
        columns=["pfr_player_id", "season", "week", "game_type", "offense_pct"],
    )
    if snaps.empty:
        return snaps

    players = cache.load("players", columns=["gsis_id", "pfr_id"]).dropna(
        subset=["gsis_id", "pfr_id"]
    )
    snaps = snaps[snaps["game_type"] == "REG"].merge(
        players, left_on="pfr_player_id", right_on="pfr_id", how="inner"
    )
    matched = snaps["gsis_id"].nunique()
    log.info("snap counts: %d players resolved to gsis ids", matched)

    return (
        snaps.groupby(["gsis_id", "season", "week"], as_index=False)["offense_pct"]
        .max()
        .rename(columns={"offense_pct": "snap_pct"})
    )


def on_full_week_grid(panel: pd.DataFrame, max_week: int = MAX_WEEK) -> pd.DataFrame:
    """Reindex to every week of every season a player appears in.

    Without this a bye week collapses the calendar and a rolling window reaches
    further back than it claims to.
    """
    if panel.empty:
        return panel

    spans = panel.groupby(["gsis_id", "season"], as_index=False).size()[["gsis_id", "season"]]
    weeks = pd.DataFrame({"week": range(1, max_week + 1)})
    grid = spans.merge(weeks, how="cross")

    # Marked before the merge so a genuinely absent week is distinguishable from
    # a week whose stat columns merely happen to be null.
    present = panel.assign(_present=1)
    full = grid.merge(present, on=["gsis_id", "season", "week"], how="left")
    full["played"] = full["_present"].fillna(0).astype(int)
    for column in VOLUME_COLUMNS:
        if column in full.columns:
            full[column] = full[column].fillna(0.0)

    return (
        full.drop(columns="_present")
        .sort_values(["gsis_id", "season", "week"])
        .reset_index(drop=True)
    )


def add_trailing(panel: pd.DataFrame, windows: tuple[int, ...] = (1, 3)) -> pd.DataFrame:
    """Rolling sums/means over the trailing ``n`` weeks, inclusive of each week."""
    out = panel.copy()
    grouped = out.groupby(["gsis_id", "season"], sort=False)

    for n in windows:
        for column in ("fantasy_points", "targets", "carries", "receptions"):
            if column in out.columns:
                out[f"{column}_l{n}"] = grouped[column].transform(
                    lambda s, n=n: s.rolling(n, min_periods=1).sum()
                )
        if "snap_pct" in out.columns:
            out[f"snap_pct_l{n}"] = grouped["snap_pct"].transform(
                lambda s, n=n: s.rolling(n, min_periods=1).mean()
            )
        out[f"played_l{n}"] = grouped["played"].transform(
            lambda s, n=n: s.rolling(n, min_periods=1).sum()
        )

    # The opportunity shock the engine exists to price: this week's snap share
    # against the weeks before it. Positive means a role just opened up.
    if {"snap_pct_l1", "snap_pct_l3"} <= set(out.columns):
        out["snap_pct_delta"] = out["snap_pct_l1"] - out["snap_pct_l3"]
    return out


def add_forward(panel: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Sum of the ``horizon`` weeks strictly after each week.

    Strictly after is the whole point: this is the outcome a bidder was buying,
    and it must contain nothing they could already see.
    """
    out = panel.copy()
    grouped = out.groupby(["gsis_id", "season"], sort=False)

    for column in ("fantasy_points", "receptions"):
        if column not in out.columns:
            continue
        forward = grouped[column].transform(
            lambda s, h=horizon: s[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
        )
        out[f"{column}_next{horizon}"] = forward.fillna(0.0)

    out[f"played_next{horizon}"] = (
        grouped["played"]
        .transform(lambda s, h=horizon: s[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1))
        .fillna(0.0)
    )
    return out
