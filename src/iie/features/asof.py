"""The as-of boundary: what a bidder could actually have known.

Every feature this engine learns from has to be restricted to information that
existed before the claim was made. Getting that wrong does not raise an error --
it produces a model that backtests beautifully and loses money, because it was
quietly reading the future.

The obvious approach, bucketing by Sleeper's transaction ``week``, does not work.
Measured on real leagues, a Sleeper week is a Wednesday-to-Wednesday window that
*straddles* the NFL week it is named for: week 5 of 2025 runs Oct 1 to Oct 8,
while the week 5 games are Oct 2 to Oct 6. Claims land throughout that window --
about 10% within six hours of its start, roughly 40% in its final 24 hours -- so
two transactions sharing a week label can have completely different information
behind them. One was made before the week's games, the other after them.

So the cut is made on ``created_ms`` against the real NFL schedule, not on the
week label. ``completed_week`` answers the only question that matters: at this
instant, which NFL weeks had finished?

Where the answer is uncertain the error is deliberately biased toward knowing
*less*. Games are assumed to run four hours, longer than a real one, so a claim
made shortly after the final whistle is attributed to the previous week. That
costs a little signal at the margin and cannot invent any.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

# nflverse publishes kickoff as local US Eastern wall-clock time.
NFL_TZ = ZoneInfo("America/New_York")

# Deliberately longer than a real game. Erring long means a claim made just after
# the last whistle is treated as pre-game, which loses signal but never leaks it.
ASSUMED_GAME_LENGTH = dt.timedelta(hours=4)

REGULAR_SEASON = "REG"


@dataclass(frozen=True)
class WeekWindow:
    """When an NFL week's games started and were certainly over."""

    season: int
    week: int
    first_kickoff: pd.Timestamp
    last_whistle: pd.Timestamp

    def contains(self, instant: pd.Timestamp) -> bool:
        return self.first_kickoff <= instant <= self.last_whistle


class NflCalendar:
    """Maps an instant to the NFL weeks that were finished by then."""

    def __init__(self, games: pd.DataFrame, *, regular_season_only: bool = True) -> None:
        if regular_season_only and "game_type" in games.columns:
            games = games[games["game_type"] == REGULAR_SEASON]
        self._windows = self._build_windows(games)

    @classmethod
    def from_cache(cls, cache, **kwargs) -> NflCalendar:
        """Build from an :class:`~iie.sources.nflverse.NflverseCache`."""
        return cls(cache.load("schedules"), **kwargs)

    @staticmethod
    def _build_windows(games: pd.DataFrame) -> dict[tuple[int, int], WeekWindow]:
        needed = {"season", "week", "gameday", "gametime"}
        missing = needed - set(games.columns)
        if missing:
            raise ValueError(f"schedule is missing columns: {sorted(missing)}")

        frame = games.dropna(subset=["gameday", "gametime"]).copy()
        kickoff_local = pd.to_datetime(
            frame["gameday"].astype(str) + " " + frame["gametime"].astype(str),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        frame["kickoff"] = kickoff_local.dt.tz_localize(
            NFL_TZ, ambiguous=True, nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
        frame = frame.dropna(subset=["kickoff"])

        windows: dict[tuple[int, int], WeekWindow] = {}
        for (season, week), group in frame.groupby(["season", "week"]):
            windows[(int(season), int(week))] = WeekWindow(
                season=int(season),
                week=int(week),
                first_kickoff=group["kickoff"].min(),
                last_whistle=group["kickoff"].max() + ASSUMED_GAME_LENGTH,
            )
        return windows

    # --- queries ----------------------------------------------------------

    def window(self, season: int, week: int) -> WeekWindow | None:
        return self._windows.get((int(season), int(week)))

    def weeks(self, season: int) -> list[int]:
        return sorted(w for (s, w) in self._windows if s == int(season))

    def completed_week(self, season: int, instant: pd.Timestamp) -> int:
        """Highest week whose games had certainly finished by ``instant``.

        Returns 0 before the season's first game is over, which is the correct
        answer for a pre-season or week 1 claim: no games have been played, so
        there is no in-season evidence to condition on.
        """
        if instant is None or pd.isna(instant):
            raise ValueError("completed_week needs a timestamp")
        instant = pd.Timestamp(instant)
        if instant.tzinfo is None:
            instant = instant.tz_localize("UTC")

        done = [
            w.week
            for w in self._windows.values()
            if w.season == int(season) and w.last_whistle <= instant
        ]
        return max(done, default=0)

    def completed_week_from_ms(self, season: int, created_ms: int) -> int:
        """Convenience for Sleeper transactions, which carry epoch milliseconds."""
        return self.completed_week(season, pd.Timestamp(int(created_ms), unit="ms", tz="UTC"))


def attach_asof_week(
    transactions: pd.DataFrame,
    calendar: NflCalendar,
    *,
    season_col: str = "season",
    created_col: str = "created_ms",
    out_col: str = "asof_week",
) -> pd.DataFrame:
    """Add the as-of week to a transaction frame.

    ``asof_week`` is the last week of evidence a bidder could have had.
    Features must be built from weeks <= asof_week; outcomes from later weeks.
    """
    frame = transactions.copy()
    instants = pd.to_datetime(frame[created_col], unit="ms", utc=True)
    seasons = frame[season_col].astype(int)
    frame[out_col] = [
        calendar.completed_week(season, instant)
        for season, instant in zip(seasons, instants, strict=True)
    ]
    return frame


def check_no_lookahead(
    frame: pd.DataFrame, *, asof_col: str = "asof_week", feature_week_col: str = "feature_week"
) -> None:
    """Raise if any feature was drawn from a week the bidder could not have seen.

    Cheap to call and worth calling: this is the one bug in the pipeline that
    silently improves every metric while destroying the model's real value.
    """
    if frame.empty:
        return
    offending = frame[frame[feature_week_col] > frame[asof_col]]
    if not offending.empty:
        raise ValueError(
            f"lookahead: {len(offending)} rows use features from after their as-of week "
            f"(worst: feature_week={offending[feature_week_col].max()} vs "
            f"asof_week={offending[asof_col].min()})"
        )
