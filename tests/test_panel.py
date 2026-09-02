"""Tests for the player-week panel.

The two things that can silently corrupt every downstream model: a rolling window
that reaches further back than it claims because a bye week collapsed the
calendar, and a forward window that includes the week a bidder was betting on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from iie.features.panel import add_forward, add_trailing, on_full_week_grid

# Weeks 1, 2 and 4: week 3 is a bye, and must not be closed over.
SPARSE = pd.DataFrame(
    {
        "gsis_id": ["p1"] * 3,
        "season": [2025] * 3,
        "week": [1, 2, 4],
        "position": ["RB"] * 3,
        "fantasy_points": [10.0, 20.0, 40.0],
        "receptions": [1.0, 2.0, 4.0],
        "targets": [2.0, 3.0, 5.0],
        "carries": [5.0, 6.0, 7.0],
        "receiving_yards": [10.0, 20.0, 30.0],
        "snap_pct": [0.5, 0.6, 0.9],
    }
)


@pytest.fixture
def grid():
    return on_full_week_grid(SPARSE, max_week=5)


def test_grid_fills_every_week(grid):
    assert grid["week"].tolist() == [1, 2, 3, 4, 5]


def test_absent_weeks_are_zero_production_not_missing(grid):
    bye = grid[grid.week == 3].iloc[0]
    assert bye["fantasy_points"] == 0.0
    assert bye["played"] == 0


def test_played_flag_marks_real_appearances(grid):
    assert grid.set_index("week")["played"].tolist() == [1, 1, 0, 1, 0]


def test_trailing_window_respects_the_calendar_not_the_rows(grid):
    """The bye must consume a slot, or the window reaches too far back."""
    out = add_trailing(grid, windows=(3,))
    week4 = out[out.week == 4].iloc[0]
    # Weeks 2, 3, 4 = 20 + 0 + 40. Row-based rolling would have given 10+20+40.
    assert week4["fantasy_points_l3"] == 60.0


def test_trailing_window_is_inclusive_of_its_own_week(grid):
    out = add_trailing(grid, windows=(1,))
    assert out[out.week == 2].iloc[0]["fantasy_points_l1"] == 20.0


def test_snap_delta_flags_an_opportunity_shock(grid):
    """A jump in snap share against recent weeks is the signal being priced."""
    out = add_trailing(grid, windows=(1, 3))
    assert out[out.week == 4].iloc[0]["snap_pct_delta"] > 0


def test_forward_window_excludes_the_current_week(grid):
    """The outcome must contain nothing the bidder could already see."""
    out = add_forward(grid, horizon=6)
    week1 = out[out.week == 1].iloc[0]
    # Weeks 2..5 only: 20 + 0 + 40 + 0. Including week 1 would add its 10.
    assert week1["fantasy_points_next6"] == 60.0


def test_forward_window_is_bounded_by_the_horizon(grid):
    out = add_forward(grid, horizon=2)
    # Weeks 2 and 3 only.
    assert out[out.week == 1].iloc[0]["fantasy_points_next2"] == 20.0


def test_final_week_has_no_future(grid):
    out = add_forward(grid, horizon=6)
    assert out[out.week == 5].iloc[0]["fantasy_points_next6"] == 0.0


def test_players_do_not_leak_into_each_other():
    two = pd.concat([SPARSE, SPARSE.assign(gsis_id="p2", fantasy_points=1.0)], ignore_index=True)
    out = add_trailing(on_full_week_grid(two, max_week=5), windows=(3,))
    at_week4 = out[out.week == 4].set_index("gsis_id")["fantasy_points_l3"]
    # p2 scores 1.0 in weeks 2 and 4 with the bye between, so 2.0; p1 is untouched.
    assert at_week4["p2"] == 2.0
    assert at_week4["p1"] == 60.0


def test_empty_panel_survives():
    assert on_full_week_grid(pd.DataFrame()).empty
