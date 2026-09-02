"""Tests for the as-of boundary.

Lookahead is the one bug here that makes every metric better while making the
model worthless, so these tests are mostly about boundaries: exactly when a week
becomes known, and that the error always points toward knowing less.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from iie.features.asof import (
    ASSUMED_GAME_LENGTH,
    NflCalendar,
    attach_asof_week,
    check_no_lookahead,
)

# Two weeks of a toy season. Times are US Eastern wall clock, as nflverse publishes.
GAMES = pd.DataFrame(
    {
        "season": [2025, 2025, 2025, 2025],
        "week": [1, 1, 2, 2],
        "game_type": ["REG", "REG", "REG", "REG"],
        "gameday": ["2025-09-07", "2025-09-08", "2025-09-14", "2025-09-15"],
        "gametime": ["13:00", "20:15", "13:00", "20:15"],
        "away_team": ["A", "B", "C", "D"],
        "home_team": ["E", "F", "G", "H"],
    }
)


@pytest.fixture
def calendar():
    return NflCalendar(GAMES)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_week_window_spans_first_kickoff_to_last_whistle(calendar):
    window = calendar.window(2025, 1)
    # 13:00 ET on the 7th is 17:00 UTC; the Monday nighter kicks at 20:15 ET.
    assert window.first_kickoff == ts("2025-09-07 17:00")
    assert window.last_whistle == ts("2025-09-09 00:15") + ASSUMED_GAME_LENGTH


def test_no_week_is_complete_before_the_season(calendar):
    assert calendar.completed_week(2025, ts("2025-09-01 12:00")) == 0


def test_week_is_not_complete_while_its_last_game_is_playing(calendar):
    """Kickoff of the final game is not the end of it."""
    assert calendar.completed_week(2025, ts("2025-09-09 01:00")) == 0


def test_week_becomes_complete_after_the_assumed_game_length(calendar):
    last_kick = ts("2025-09-09 00:15")
    assert calendar.completed_week(2025, last_kick + ASSUMED_GAME_LENGTH) == 1
    assert calendar.completed_week(2025, last_kick + dt.timedelta(hours=3)) == 0


def test_error_is_biased_toward_knowing_less(calendar):
    """A claim just after a realistic 3h game must still read as pre-completion."""
    realistic_end = ts("2025-09-09 00:15") + dt.timedelta(hours=3, minutes=10)
    assert calendar.completed_week(2025, realistic_end) == 0, "must never over-credit knowledge"


def test_later_weeks_accumulate(calendar):
    assert calendar.completed_week(2025, ts("2025-09-16 12:00")) == 2


def test_naive_timestamps_are_treated_as_utc(calendar):
    assert calendar.completed_week(2025, pd.Timestamp("2025-09-16 12:00")) == 2


def test_unknown_season_has_no_completed_weeks(calendar):
    assert calendar.completed_week(1999, ts("2025-12-01 12:00")) == 0


def test_epoch_milliseconds_helper_agrees(calendar):
    instant = ts("2025-09-16 12:00")
    assert calendar.completed_week_from_ms(2025, int(instant.timestamp() * 1000)) == 2


def test_missing_timestamp_is_an_error(calendar):
    with pytest.raises(ValueError, match="needs a timestamp"):
        calendar.completed_week(2025, None)


def test_postseason_is_excluded_by_default():
    games = pd.concat(
        [
            GAMES,
            pd.DataFrame(
                {
                    "season": [2025],
                    "week": [3],
                    "game_type": ["POST"],
                    "gameday": ["2026-01-11"],
                    "gametime": ["13:00"],
                    "away_team": ["X"],
                    "home_team": ["Y"],
                }
            ),
        ],
        ignore_index=True,
    )
    assert NflCalendar(games).weeks(2025) == [1, 2]


def test_schedule_missing_columns_fails_loudly():
    with pytest.raises(ValueError, match="missing columns"):
        NflCalendar(pd.DataFrame({"season": [2025], "week": [1]}))


def test_attach_asof_week_splits_a_shared_week_label(calendar):
    """The whole point: one Sleeper week holds claims with different knowledge."""
    before = int(ts("2025-09-10 12:00").timestamp() * 1000)  # after wk1, before wk2
    after = int(ts("2025-09-16 12:00").timestamp() * 1000)  # after wk2
    frame = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "season": ["2025", "2025"],
            "week": [2, 2],
            "created_ms": [before, after],
        }
    )
    out = attach_asof_week(frame, calendar)
    assert out["asof_week"].tolist() == [1, 2]


def test_check_no_lookahead_passes_when_features_are_in_the_past():
    frame = pd.DataFrame({"asof_week": [5, 5], "feature_week": [4, 5]})
    check_no_lookahead(frame)


def test_check_no_lookahead_catches_a_leak():
    frame = pd.DataFrame({"asof_week": [5, 5], "feature_week": [4, 6]})
    with pytest.raises(ValueError, match="lookahead"):
        check_no_lookahead(frame)


def test_check_no_lookahead_tolerates_empty():
    check_no_lookahead(pd.DataFrame(columns=["asof_week", "feature_week"]))
