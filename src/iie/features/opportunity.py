"""Opportunity shocks: the reason a waiver claim is worth making at all.

The panel in ``iie.features.panel`` measures production and snap share, which
describe opportunity *after* it has already shifted. By the time a backup's snap
count rises he has already played the game, and the waiver wire has moved. This
module supplies the signals that precede that.

**Depth chart movement is the primary signal, and it needs no injury data.** A
player moving from third on the depth chart to first encodes the starter being
hurt *and* the coach benching him, without having to distinguish the two or to
join anything whose publication time is uncertain.

nflverse publishes two incompatible schemas for this: 2025 onward is daily
snapshots with a real timestamp, 2024 and earlier is weekly rows with no timestamp
at all. Both are normalised to one row per player-week here. That forfeits the
daily precision the newer format allows, deliberately -- a feature present only in
the test season cannot be trained on, and differing granularity between train and
test is itself a distribution shift. Concatenating the raw frames instead loses a
whole season silently, which is what happened before the normalisers existed.

Measured on the corpus: this is where the mispricing lives. Controlling for current
depth rank, a player promoted to first string within the week costs 66% more than
an established starter and returns 26% less.

**Injury status is secondary and deliberately lagged.** The official report is
week-level with no timestamp, and it publishes Wednesday-Friday -- the same window
waivers run in. Worse, an injury sustained in week *W*'s game first appears in the
week *W+1* report, which is published *after* the Wednesday claim that reacts to
it. So the report lags the event it describes, and using it unlagged would be
lookahead rather than signal. It is joined only for weeks the bidder had certainly
seen, and it answers "is this player himself healthy", not "did a role just open".

**Cross-league demand is the news proxy.** There is no structured feed for "the
coach says he's benched", but hundreds of leagues react to that news within hours
and the corpus observes all of it. Measured on 2025, demand correlates +0.298 with
price against +0.108 for price against realized value -- the crowd knows what is
happening, and pays too much for it. Counted strictly from claims that precede the
one being scored.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Positions a fantasy waiver claim can plausibly concern. Depth charts carry every
# position group including special teams and the offensive line. Note "PK": nflverse
# labels kickers PK here while Sleeper calls them K, and filtering on K alone drops
# every kicker silently.
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "FB", "PK", "K")

# Ordered by severity so the code is monotone: a larger number is worse news about
# this player's own availability.
INJURY_SEVERITY = {
    "Questionable": 1,
    "Doubtful": 2,
    "Out": 3,
    "Injured Reserve": 4,
    "IR": 4,
    "PUP": 4,
}

# How far back to look for the previous depth position. A week is long enough for a
# chart to be republished and short enough that a change is recent news.
PROMOTION_LOOKBACK = pd.Timedelta(days=7)


# Every normaliser returns this shape, so the two schemas concatenate cleanly and an
# empty result from either path does not break the columns the caller expects.
_CHART_COLUMNS = ["gsis_id", "season", "week", "club", "pos_abb", "pos_rank", "dt"]


def _empty_charts() -> pd.DataFrame:
    return pd.DataFrame(columns=_CHART_COLUMNS)


def _normalise_modern_charts(charts: pd.DataFrame, calendar) -> pd.DataFrame:
    """2025-onward schema: daily snapshots keyed by ``dt``, ``pos_abb``, ``pos_rank``."""
    charts = charts[charts["pos_abb"].isin(FANTASY_POSITIONS)].dropna(subset=["gsis_id", "dt"])
    charts = charts.copy()
    charts["dt"] = pd.to_datetime(charts["dt"], errors="coerce", utc=True)
    charts["pos_rank"] = pd.to_numeric(charts["pos_rank"], errors="coerce")
    if charts.empty:
        return _empty_charts()

    # Reuse the as-of boundary rather than inventing a second time convention: a
    # snapshot is labelled with the last week whose games had finished when it was
    # taken, which is exactly how claims are labelled.
    charts["season"] = charts["dt"].dt.year.where(
        charts["dt"].dt.month >= 3, charts["dt"].dt.year - 1
    )
    charts["week"] = [
        calendar.completed_week(season, instant)
        for season, instant in zip(charts["season"], charts["dt"], strict=True)
    ]
    return charts.rename(columns={"team": "club"})[_CHART_COLUMNS]


def _normalise_legacy_charts(charts: pd.DataFrame) -> pd.DataFrame:
    """Pre-2025 schema: weekly rows keyed by ``club_code``, ``position``, ``depth_team``.

    nflverse changed this format entirely between seasons. Concatenating the two
    without normalising produces NaN columns that the filters then drop, losing a
    whole season silently -- which is what happened here before this existed.
    """
    if charts.empty or "position" not in charts.columns:
        return _empty_charts()
    # DataFrame.get returns the *default* when the column is absent, and a bare
    # string has no .eq -- so the membership test has to come first.
    if "game_type" in charts.columns:
        charts = charts[charts["game_type"] == "REG"]
    charts = charts[charts["position"].isin(FANTASY_POSITIONS)]
    charts = charts.dropna(subset=["gsis_id", "week"]).copy()

    charts["pos_rank"] = pd.to_numeric(charts["depth_team"], errors="coerce")
    charts = charts.dropna(subset=["pos_rank"])
    # A player is listed once per role (RB and KR, say); his depth is the best of them.
    charts = charts.groupby(["gsis_id", "season", "week", "club_code", "position"], as_index=False)[
        "pos_rank"
    ].min()

    return charts.rename(columns={"club_code": "club", "position": "pos_abb"}).assign(dt=pd.NaT)[
        _CHART_COLUMNS
    ]


def build_depth_charts(cache, seasons: list[int], calendar) -> pd.DataFrame:
    """Depth charts normalised to one row per (player, season, week).

    Week-level rather than per-snapshot, deliberately. The 2025 format carries daily
    timestamps and the 2024 format does not, and a feature that exists only in the
    test season cannot be trained on -- differing granularity between train and test
    is itself a distribution shift. Both are reduced to the last chart known as of
    each completed week, which is safely in the past for any claim at that boundary.
    """
    frames = []
    for season in seasons:
        try:
            raw = cache.load("depth_charts", season)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("no depth charts for %s: %s", season, exc)
            continue
        modern = "pos_abb" in raw.columns and "dt" in raw.columns
        frames.append(
            _normalise_modern_charts(raw, calendar) if modern else _normalise_legacy_charts(raw)
        )

    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_charts().assign(depth_size=None)

    charts = pd.concat(frames, ignore_index=True)
    charts = charts.dropna(subset=["season", "week"]).astype({"season": int, "week": int})

    # Several snapshots can fall in one week; the latest is what a bidder saw.
    charts = charts.sort_values("dt", na_position="first").drop_duplicates(
        ["gsis_id", "season", "week"], keep="last"
    )

    # How many players the club lists at this position, so a rank of 2 out of 2
    # reads differently from 2 out of 6.
    charts["depth_size"] = charts.groupby(["season", "week", "club", "pos_abb"])[
        "pos_rank"
    ].transform("size")

    columns = ["gsis_id", "season", "week", "club", "pos_abb", "pos_rank", "depth_size"]
    log.info(
        "depth charts: %d player-weeks across seasons %s",
        len(charts),
        sorted(charts["season"].unique()),
    )
    return charts[columns].reset_index(drop=True)


def build_injury_status(cache, seasons: list[int]) -> pd.DataFrame:
    """Weekly injury report, reduced to a severity code per player-week."""
    frames = []
    for season in seasons:
        try:
            frames.append(cache.load("injuries", season))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("no injury report for %s: %s", season, exc)
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "injury_severity"])

    injuries = pd.concat(frames, ignore_index=True)
    if injuries.empty or "report_status" not in injuries.columns:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "injury_severity"])

    injuries = injuries.dropna(subset=["gsis_id"])
    # Same trap as above: .get returns the string default when absent.
    if "season_type" in injuries.columns:
        injuries = injuries[injuries["season_type"] == "REG"]
    injuries["injury_severity"] = (
        injuries["report_status"].map(INJURY_SEVERITY).fillna(0.0).astype(float)
    )

    # A player can appear more than once in a week's report; keep the worst news.
    return (
        injuries.groupby(["gsis_id", "season", "week"], as_index=False)["injury_severity"]
        .max()
        .astype({"season": int, "week": int})
    )


def attach_depth_features(
    claims: pd.DataFrame, charts: pd.DataFrame, *, asof_col: str = "asof_week"
) -> pd.DataFrame:
    """Depth position as of each claim, and how it changed in the preceding week.

    Joined at ``asof_week`` -- the last week whose games had finished when the claim
    was made -- and again at the week before it. The difference is the signal:
    ``depth_promotion`` is positive when a player has moved up the chart, which is
    what a role opening looks like *before* it shows up in a box score.
    """
    columns = ("depth_rank", "depth_size", "depth_promotion", "depth_is_starter")
    out = claims.copy()
    if charts.empty:
        for column in columns:
            out[column] = np.nan
        return out

    out["_season_int"] = out["season"].astype(int)
    keys = ["gsis_id", "_season_int", "_week"]
    lookup = charts.rename(columns={"season": "_season_int", "week": "_week"})

    def _rank_at(week_offset: int, name: str) -> pd.Series:
        probe = out[["gsis_id", "_season_int", asof_col]].copy()
        probe["_week"] = probe[asof_col] + week_offset
        merged = probe.merge(
            lookup[[*keys, "pos_rank", "depth_size"]], on=keys, how="left"
        ).set_axis(out.index)
        out[f"{name}_size"] = merged["depth_size"]
        return merged["pos_rank"]

    out["depth_rank"] = _rank_at(0, "depth")
    previous = _rank_at(-1, "_prev")
    out = out.drop(columns=["_prev_size"])

    # Rank counts downward, so a fall in the rank number is a promotion.
    out["depth_promotion"] = previous - out["depth_rank"]
    out["depth_is_starter"] = (out["depth_rank"] == 1).astype(float)
    out.loc[out["depth_rank"].isna(), "depth_is_starter"] = np.nan
    return out.drop(columns="_season_int")


def attach_injury_features(
    claims: pd.DataFrame, injuries: pd.DataFrame, *, asof_col: str = "asof_week"
) -> pd.DataFrame:
    """Own injury severity from the most recent report the bidder had certainly seen.

    Joined at ``asof_week``, never later. The week *W* report is published before
    week *W*'s games, so a claim made after those games has seen it; the week *W+1*
    report has not been published yet and using it would be lookahead.
    """
    out = claims.copy()
    if injuries.empty:
        out["injury_severity"] = np.nan
        return out

    out["_season_int"] = out["season"].astype(int)
    # A left merge keeps row order but discards the index; restore it so callers
    # can align the result against what they passed in.
    merged = out.merge(
        injuries.rename(columns={"season": "_season_int", "week": asof_col}),
        on=["gsis_id", "_season_int", asof_col],
        how="left",
    ).set_axis(out.index)
    # Absent from the report means not listed as injured, which is information.
    merged["injury_severity"] = merged["injury_severity"].fillna(0.0)
    merged.loc[merged["gsis_id"].isna(), "injury_severity"] = np.nan
    return merged.drop(columns="_season_int")


def attach_cross_league_demand(
    claims: pd.DataFrame, *, created_col: str = "created_ms"
) -> pd.DataFrame:
    """How many other leagues had already claimed this player, before this claim.

    The historical stand-in for a news feed. Strictly backward-looking: a claim is
    counted only if it happened before the one being scored. Leagues are counted
    once each, so a single contested auction with five bidders is one league's
    worth of demand rather than five.
    """
    out = claims.copy()
    if out.empty:
        out["demand_prior"] = np.nan
        return out

    # One row per (season, player, league) claim event, at its earliest timestamp:
    # several bidders inside one league are one league's demand, not several.
    events = (
        out.groupby(["season", "week", "player_id", "league_id"], as_index=False)[created_col]
        .min()
        .sort_values(["season", "week", "player_id", created_col])
    )
    events["prior_leagues"] = events.groupby(["season", "week", "player_id"]).cumcount()

    merged = out.merge(
        events.rename(columns={created_col: "_first_claim"}),
        on=["season", "week", "player_id", "league_id"],
        how="left",
    ).set_axis(out.index)
    # A league's own earlier bidders must not count toward its own demand, so the
    # count is taken at the league's first claim on that player.
    merged["demand_prior"] = merged["prior_leagues"].astype(float)
    return merged.drop(columns=["prior_leagues", "_first_claim"])


def attach_opportunity(
    claims: pd.DataFrame, cache, seasons: list[int] | None = None, calendar=None
) -> pd.DataFrame:
    """Attach every opportunity feature. Convenience wrapper over the three above."""
    seasons = seasons or sorted({int(s) for s in claims["season"].unique()})
    if calendar is None:
        from .asof import NflCalendar

        calendar = NflCalendar.from_cache(cache)

    out = attach_depth_features(claims, build_depth_charts(cache, seasons, calendar))
    out = attach_injury_features(out, build_injury_status(cache, seasons))
    out = attach_cross_league_demand(out)

    log.info(
        "opportunity features: depth rank on %.1f%% of claims, injury status on %.1f%%",
        100 * out["depth_rank"].notna().mean(),
        100 * out["injury_severity"].notna().mean(),
    )
    return out
