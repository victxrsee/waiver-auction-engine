"""Claim observations: one row per waiver bid, with what was known and what happened.

This is the table the models are fitted on. Each row is a single bid -- won or
lost -- carrying the features a bidder could have seen and the production the
player actually delivered afterwards.

Losing bids matter as much as winning ones. Sleeper reports failed waiver claims
with their bid attached, so roughly half of these rows are offers that did not
win. Without them the price model would see only the maximum of each auction and
have to infer the rest of the distribution; with them it observes the competing
bids directly.

Points are recomputed per league rather than taken from nflverse's presets,
because the same player is a different asset under different scoring. A reception
worth half a point in one league is worth a full point in the next, and a price
model trained on the wrong one is learning someone else's market.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..crawl.store import read_table
from .asof import NflCalendar, attach_asof_week, check_no_lookahead
from .opportunity import attach_opportunity
from .panel import add_forward, add_trailing, build_player_week, on_full_week_grid

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimSpec:
    """Feature windows and the horizon a claim's value is judged over."""

    trailing_windows: tuple[int, ...] = (1, 3)
    # Defenses are kept by default. They carry no player features -- nothing in
    # nflverse is keyed on a team abbreviation -- but they are 12% of all claims,
    # and a price model that predicts what leaguemates will bid needs no player
    # features at all. Dropping them here would hide that gap instead of scoping it.
    include_defense: bool = True
    # Six weeks is most of what remains of a mid-season roster spot, and short
    # enough that a claim in week 12 still has an outcome to be scored against.
    outcome_horizon: int = 6


# nflverse fantasy_points is standard scoring, so a league's PPR setting is
# applied on top rather than choosing between two fixed presets.
DEFAULT_RECEPTION_POINTS = 0.0


def _league_context(leagues: pd.DataFrame) -> pd.DataFrame:
    """Per-league scoring and FAAB budget, from the stored settings blobs.

    League size matters too, and is carried here for the same reason: it is a
    model feature, and a feature that is declared but never joined trains nothing.

    The budget matters as much as the scoring. Bids are denominated in each
    league's own currency, and pooling raw amounts across leagues destroys the
    signal: measured on the corpus, bid-vs-realized-value correlation is +0.045
    on raw bids and +0.102 once expressed as a share of budget. Most leagues run
    a 100 budget but some run 1000, so a bid of 50 means entirely different
    things in each.
    """
    sizes = pd.to_numeric(leagues["total_rosters"], errors="coerce")
    points, budgets = [], []
    for scoring, settings in zip(
        leagues["scoring_settings_json"], leagues["settings_json"], strict=True
    ):
        try:
            points.append(float((json.loads(scoring) or {}).get("rec", DEFAULT_RECEPTION_POINTS)))
        except (TypeError, ValueError, json.JSONDecodeError):
            points.append(DEFAULT_RECEPTION_POINTS)
        try:
            budget = (json.loads(settings) or {}).get("waiver_budget")
            budgets.append(float(budget) if budget else None)
        except (TypeError, ValueError, json.JSONDecodeError):
            budgets.append(None)

    return pd.DataFrame(
        {
            "league_id": leagues["league_id"],
            "reception_points": points,
            "waiver_budget": budgets,
            # League size is a first-order driver of price -- a player is scarcer
            # in a 14-team league than a 10-team one -- and it was declared as a
            # model feature long before anything actually joined it on.
            "total_rosters": sizes.to_numpy(),
        }
    )


def build_claims(
    corpus_root: Path | str,
    cache,
    bridge,
    calendar: NflCalendar,
    *,
    spec: ClaimSpec | None = None,
) -> pd.DataFrame:
    """Assemble the claim table from a crawled corpus plus nflverse."""
    spec = spec or ClaimSpec()

    transactions = read_table(corpus_root, "transactions")
    players = read_table(corpus_root, "transaction_players")
    leagues = read_table(corpus_root, "leagues")
    if transactions.empty or players.empty:
        log.warning("corpus has no transactions; nothing to build")
        return pd.DataFrame()

    claims = _waiver_adds(transactions, players)
    if claims.empty:
        log.warning("corpus has no waiver bids")
        return claims

    claims = attach_asof_week(claims, calendar)
    claims = _resolve_players(claims, bridge)

    if not spec.include_defense:
        claims = claims[~claims["is_defense"]].copy()

    seasons = sorted({int(s) for s in claims["season"].unique()})
    panel = build_player_week(cache, seasons, gsis_ids=set(claims["gsis_id"].dropna()))
    if panel.empty:
        log.warning("no nflverse player weeks for seasons %s", seasons)
        return pd.DataFrame()

    panel = on_full_week_grid(panel)
    panel = add_trailing(panel, spec.trailing_windows)
    panel = add_forward(panel, spec.outcome_horizon)

    claims["season_int"] = claims["season"].astype(int)
    merged = claims.merge(
        panel,
        left_on=["gsis_id", "season_int", "asof_week"],
        right_on=["gsis_id", "season", "week"],
        how="left",
        suffixes=("", "_panel"),
    )

    merged = merged.merge(_league_context(leagues), on="league_id", how="left")
    merged["reception_points"] = merged["reception_points"].fillna(DEFAULT_RECEPTION_POINTS)
    merged = _apply_league_scoring(merged, spec)

    # The comparable price across leagues, and what a bid model should target.
    budget = merged["waiver_budget"].where(merged["waiver_budget"] > 0)
    merged["bid_pct"] = merged["waiver_bid"] / budget

    # Depth chart, injury status and cross-league demand: the signals that precede
    # a role change rather than describing it after the fact.
    merged = attach_opportunity(merged, cache, seasons=seasons, calendar=calendar)

    # Read the week back from the *panel* side of the join rather than copying
    # asof_week. Copying it made the assertion compare a column to itself, so it
    # would still have passed if the merge were changed to join on the Sleeper
    # week label -- which is exactly the regression it exists to catch.
    merged["feature_week"] = merged["week_panel"]
    check_no_lookahead(merged)

    return merged.reset_index(drop=True)


def _resolve_players(claims: pd.DataFrame, bridge) -> pd.DataFrame:
    """Attach gsis ids, separating unpriceable positions from real failures.

    Three outcomes, and conflating them is how a whole position disappears
    silently: ``resolved`` joins to nflverse, ``defense`` never will because
    nothing there is keyed on a team abbreviation, and ``unresolved`` is a genuine
    bridge miss worth watching.
    """
    out = claims.copy()
    out["is_defense"] = [bridge.is_defense(pid) for pid in out["player_id"]]
    out["gsis_id"] = [
        None if is_def else (r.gsis_id if (r := bridge.resolve(pid)) else None)
        for pid, is_def in zip(out["player_id"], out["is_defense"], strict=True)
    ]

    out["resolution"] = "resolved"
    out.loc[out["is_defense"], "resolution"] = "defense"
    out.loc[~out["is_defense"] & out["gsis_id"].isna(), "resolution"] = "unresolved"

    counts = out["resolution"].value_counts()
    total = len(out)
    log.info(
        "claims: %d resolved, %d defenses (unpriceable), %d unresolved (of %d)",
        counts.get("resolved", 0),
        counts.get("defense", 0),
        counts.get("unresolved", 0),
        total,
    )
    if counts.get("unresolved", 0):
        log.warning(
            "dropping %d claims whose players did not resolve (%.1f%%)",
            counts["unresolved"],
            100 * counts["unresolved"] / total,
        )
    return out[out["resolution"] != "unresolved"].copy()


def _waiver_adds(transactions: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Every bid: the adds of waiver transactions, won and lost alike."""
    header = transactions[transactions["type"] == "waiver"][
        ["transaction_id", "league_id", "season", "week", "status", "created_ms", "roster_ids"]
    ]
    adds = players[players["action"] == "add"][
        ["transaction_id", "player_id", "roster_id", "waiver_bid"]
    ]

    claims = header.merge(adds, on="transaction_id", how="inner")
    claims = claims[claims["waiver_bid"].notna()]

    # A league crawled before its season was known writes rows that cannot be
    # joined to any NFL week. Rare, but they must be dropped rather than coerced.
    labelled = claims["season"].notna() & (claims["season"].astype(str).str.strip() != "")
    if not labelled.all():
        log.warning("dropping %d claims with no season label", (~labelled).sum())

    claims = claims[labelled].copy()
    claims["won"] = (claims["status"] == "complete").astype(int)
    return claims


def _apply_league_scoring(frame: pd.DataFrame, spec: ClaimSpec) -> pd.DataFrame:
    """Restate nflverse standard points in each league's own scoring."""
    out = frame.copy()
    rec = out["reception_points"]

    for n in spec.trailing_windows:
        points, receptions = f"fantasy_points_l{n}", f"receptions_l{n}"
        if points in out.columns and receptions in out.columns:
            out[f"points_l{n}"] = out[points] + rec * out[receptions]

    horizon = spec.outcome_horizon
    points, receptions = f"fantasy_points_next{horizon}", f"receptions_next{horizon}"
    if points in out.columns and receptions in out.columns:
        # The target: what the bidder actually bought.
        out["points_gained"] = out[points] + rec * out[receptions]
    return out
