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
at all. Concatenating the raw frames loses a whole season silently, which is what
happened before the normalisers existed.

**The timestamp is kept, and the join is made as of the claim itself.** An earlier
version labelled each snapshot with the week that had completed when it was
published and joined on that label. But such a label names a ~7-day bucket running
from the end of week *W*'s games to the end of week *W+1*'s, and a claim sits at
the *near* end of it -- so the snapshot a claim read was published a median 5.0
days after it, and later than it 97.7% of the time. Timestamped charts are now
joined with ``merge_asof`` on the claim's own instant. Untimestamped ones keep the
week join, which is sound there because their label is the week the chart
*describes* rather than a window it was published in.

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

# Return and holder roles in the legacy schema. Being first-choice punt returner
# says nothing about a player's offensive depth, which is the thing being measured.
# "K"/"PK" are absent deliberately: for a kicker that is the real position.
SPECIAL_TEAMS_ROLES = frozenset({"PR", "KR", "KOR", "KO", "P", "H", "LS"})

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
    # A player is listed once per role, and the return roles are the trap: a WR who
    # is his club's first-choice punt returner is PR1, and taking the best rank over
    # all his roles promotes him to WR1. That made 52% of 2024 claims read as
    # starters against 36% of 2025's -- a fabricated distribution shift across the
    # train/test boundary, in the feature the model leans on hardest.
    if "depth_position" in charts.columns:
        charts = charts[~charts["depth_position"].isin(SPECIAL_TEAMS_ROLES)]
    charts = charts.dropna(subset=["gsis_id", "week"]).copy()

    charts["pos_rank"] = pd.to_numeric(charts["depth_team"], errors="coerce")
    charts = charts.dropna(subset=["pos_rank"])
    # What remains is his offensive depth, best across however it is spelled.
    charts = charts.groupby(["gsis_id", "season", "week", "club_code", "position"], as_index=False)[
        "pos_rank"
    ].min()

    return charts.rename(columns={"club_code": "club", "position": "pos_abb"}).assign(dt=pd.NaT)[
        _CHART_COLUMNS
    ]


def build_depth_charts(cache, seasons: list[int], calendar) -> pd.DataFrame:
    """Depth charts from both schemas, one row per observation, timestamps intact.

    Every snapshot is kept rather than reduced to one per player-week. The reduction
    is what caused the leak: it retained the *last* chart carrying a week label,
    which sits at the far end of a window ending after the following week's games.
    ``attach_depth_features`` picks the right one per claim instead, as of the claim.

    ``dt`` is NaT for the 2024-and-earlier schema, which publishes no timestamp; those
    rows are joined by week.
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
        return _empty_charts().assign(depth_size=None, dt=pd.NaT)

    charts = pd.concat(frames, ignore_index=True)
    charts = charts.dropna(subset=["season", "week"]).astype({"season": int, "week": int})

    # Snapshots are NOT collapsed to one row per player-week. Doing that used to
    # keep the *last* snapshot carrying a week label -- but that label names a
    # ~7-day bucket running from the end of week W's games to the end of week
    # W+1's, so the kept snapshot sat at the far end of the window while the claim
    # sat at the near end. Measured: the chart a claim read was published a median
    # 5.0 days *after* it, and later than it 97.7% of the time. The timestamp is
    # kept so the join can be made as-of the claim itself.
    charts = charts.sort_values("dt", na_position="first")

    # How many players the club lists at this position, so a rank of 2 out of 2
    # reads differently from 2 out of 6. Scoped to one published chart, not to a
    # week, or snapshots inside the week would be counted together.
    charts["depth_size"] = charts.groupby(
        ["season", "week", "club", "pos_abb", "dt"], dropna=False
    )["pos_rank"].transform("size")

    # "dt" is load-bearing: it is what lets the join be made as of the claim rather
    # than as of a week label. NaT marks the untimestamped legacy schema.
    columns = ["gsis_id", "season", "week", "club", "pos_abb", "pos_rank", "depth_size", "dt"]
    log.info(
        "depth charts: %d player-weeks across seasons %s",
        len(charts),
        sorted(charts["season"].unique()),
    )
    return charts[columns].reset_index(drop=True)


def _regular_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Regular-season rows, using whichever column this season's schema carries.

    2024 has ``game_type`` only; 2025 has both it and ``season_type``. The choice
    has to be made per season, before any concatenation -- see ``build_injury_status``.
    """
    for column in ("season_type", "game_type"):
        if column in frame.columns:
            return frame[frame[column].isin(("REG",))]
    return frame


def build_injury_status(cache, seasons: list[int]) -> pd.DataFrame:
    """Weekly injury report, reduced to a severity code per player-week.

    Each season is filtered *before* the concatenation, not after. nflverse added
    ``season_type`` in 2025, so a post-concat filter on it silently discards every
    2024 row -- the column exists on the combined frame only because 2025 supplied
    it, and 2024's rows are all NaN there. That is the same trap the two depth-chart
    normalisers exist to avoid, and it cost the whole training season once already:
    ``injury_severity`` was uniformly 0.0 across 2024 while looking perfectly healthy.
    """
    frames = []
    for season in seasons:
        try:
            raw = cache.load("injuries", season)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("no injury report for %s: %s", season, exc)
            continue
        if not raw.empty and "report_status" in raw.columns:
            frames.append(_regular_season(raw))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "season", "week", "injury_severity"])

    injuries = pd.concat(frames, ignore_index=True).dropna(subset=["gsis_id"])
    injuries["injury_severity"] = (
        injuries["report_status"].map(INJURY_SEVERITY).fillna(0.0).astype(float)
    )

    # A player can appear more than once in a week's report; keep the worst news.
    return (
        injuries.groupby(["gsis_id", "season", "week"], as_index=False)["injury_severity"]
        .max()
        .astype({"season": int, "week": int})
    )


DEPTH_COLUMNS = ("depth_rank", "depth_size", "depth_promotion", "depth_is_starter")
# How far back to look for the rank a promotion is measured against. A week, because
# depth charts are published roughly weekly and the signal is "he moved up since the
# last time anyone looked".
PROMOTION_LOOKBACK = pd.Timedelta(days=7)


def _asof_rank(claims: pd.DataFrame, charts: pd.DataFrame, instants: pd.Series) -> pd.DataFrame:
    """The last chart published at or before each instant, per player.

    ``merge_asof`` on the raw timestamp rather than on a week label, because a week
    label cannot express "before this Wednesday" -- it names a window that ends
    after the following Sunday's games.
    """
    # merge_asof refuses to join datetime64[ms] against datetime64[us], and the two
    # sides arrive at different resolutions: claim instants come from epoch
    # milliseconds, chart timestamps from parsed strings. Pin both.
    unit = "datetime64[us, UTC]"
    left = pd.DataFrame(
        {"gsis_id": claims["gsis_id"], "_at": pd.Series(instants).astype(unit)}
    ).dropna(subset=["_at"])
    left = left.sort_values("_at", kind="stable")
    right = charts.assign(dt=charts["dt"].astype(unit)).sort_values("dt", kind="stable")
    merged = pd.merge_asof(
        left,
        right[["dt", "gsis_id", "pos_rank", "depth_size"]],
        left_on="_at",
        right_on="dt",
        by="gsis_id",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = left.index
    # The property a week-granularity guard cannot check. Cheap, and this is exactly
    # the assertion whose absence let a five-day leak ship.
    late = merged["dt"].notna() & (merged["dt"] > merged["_at"])
    if late.any():
        raise AssertionError(f"{late.sum()} depth charts joined from after the claim")
    return merged.reindex(claims.index)


def attach_depth_features(
    claims: pd.DataFrame,
    charts: pd.DataFrame,
    *,
    asof_col: str = "asof_week",
    created_col: str = "created_ms",
) -> pd.DataFrame:
    """Depth position as of each claim, and how it changed in the preceding week.

    Two join paths, because the two nflverse schemas carry different time
    information and forcing them through one convention is what caused the leak
    this function was rewritten to fix.

    * **Timestamped charts (2025 onward)** are joined ``merge_asof`` on the claim's
      own instant: the last chart actually published when the bid was placed.
    * **Untimestamped charts (2024 and earlier)** have no publication time, only a
      game week, and are joined at ``asof_week`` -- the last week whose games had
      finished. That is genuinely backward-looking, since the label is the week the
      chart describes rather than a bucket it was published in.

    ``depth_promotion`` is positive when a player has moved up the chart, which is
    what a role opening looks like *before* it reaches a box score.
    """
    out = claims.copy()
    if charts.empty or "dt" not in charts.columns:
        for column in DEPTH_COLUMNS:
            out[column] = np.nan
        return out

    # One row per player per observation. A player can be listed at two positions or
    # for two clubs in the same week, and a left merge on a duplicated key fans out:
    # the claim table silently grew by 48 rows before this existed. Best rank wins,
    # matching what the legacy normaliser does across roles.
    def _one_per(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if frame.empty:
            return frame
        return (
            frame.sort_values("pos_rank").drop_duplicates(keys, keep="first").reset_index(drop=True)
        )

    timestamped = _one_per(charts[charts["dt"].notna()], ["gsis_id", "dt"])
    legacy = _one_per(charts[charts["dt"].isna()], ["gsis_id", "season", "week"])

    rank = pd.Series(np.nan, index=out.index, dtype=float)
    size = pd.Series(np.nan, index=out.index, dtype=float)
    previous = pd.Series(np.nan, index=out.index, dtype=float)

    if not timestamped.empty and created_col in out.columns:
        instants = pd.to_datetime(out[created_col], unit="ms", utc=True)
        now = _asof_rank(out, timestamped, instants)
        before = _asof_rank(out, timestamped, instants - PROMOTION_LOOKBACK)
        rank = rank.fillna(now["pos_rank"])
        size = size.fillna(now["depth_size"])
        previous = previous.fillna(before["pos_rank"])

    if not legacy.empty:
        probe = pd.DataFrame(
            {"gsis_id": out["gsis_id"], "_s": out["season"].astype(int), "_w": out[asof_col]}
        )
        lookup = legacy.rename(columns={"season": "_s", "week": "_w"})
        keys = ["gsis_id", "_s", "_w"]
        for offset, target in ((0, "now"), (-1, "before")):
            got = (
                probe.assign(_w=probe["_w"] + offset)
                .merge(lookup[[*keys, "pos_rank", "depth_size"]], on=keys, how="left")
                .set_axis(out.index)
            )
            if target == "now":
                rank = rank.fillna(got["pos_rank"])
                size = size.fillna(got["depth_size"])
            else:
                previous = previous.fillna(got["pos_rank"])

    out["depth_rank"] = rank
    out["depth_size"] = size
    # Rank counts downward, so a fall in the rank number is a promotion.
    out["depth_promotion"] = previous - rank
    out["depth_is_starter"] = (rank == 1).astype(float)
    out.loc[rank.isna(), "depth_is_starter"] = np.nan
    return out


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
    """What *share* of leagues had already claimed this player, before this claim.

    The historical stand-in for a news feed. Strictly backward-looking: a claim is
    counted only if it happened before the one being scored. Leagues are counted
    once each, so a single contested auction with five bidders is one league's
    worth of demand rather than five.

    A share rather than a count, because the count is not a property of the player
    at all -- it is a property of how many leagues the crawler happened to reach
    that season. The corpus holds 174 leagues in 2024 and 1,823 in 2025, so the raw
    count averaged 8.6 in training and 278 in test: the model learned thresholds on
    a scale that did not exist in the season it was scored on. Dividing by the
    leagues active that week makes the feature mean the same thing in both, and
    survives the corpus growing again.
    """
    out = claims.copy()
    if out.empty:
        out["demand_share"] = np.nan
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
    active = out.groupby(["season", "week"])["league_id"].transform("nunique").astype(float)
    merged["demand_share"] = merged["prior_leagues"].astype(float) / active.where(active > 0)
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
