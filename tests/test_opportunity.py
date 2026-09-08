"""Tests for opportunity-shock features.

Two things dominate here. The first is schema drift: nflverse publishes depth
charts in two incompatible formats and concatenating them silently loses a whole
season, so both normalisers are pinned. The second is as-of discipline -- these
features describe a moment, and reading one week too late turns signal into
lookahead.
"""

from __future__ import annotations

import pandas as pd
import pytest

from iie.features.opportunity import (
    attach_cross_league_demand,
    attach_depth_features,
    attach_injury_features,
    build_depth_charts,
    build_injury_status,
)


class FakeCalendar:
    """Maps an instant to a completed week by date, cheaply."""

    def completed_week(self, season, instant):
        return int(pd.Timestamp(instant).dayofyear // 7)


# The 2025-onward format: daily snapshots, ranked, with a real timestamp.
MODERN = pd.DataFrame(
    {
        "dt": pd.to_datetime(["2025-09-10", "2025-09-10", "2025-09-17", "2025-09-17"], utc=True),
        "team": ["SF"] * 4,
        "gsis_id": ["p1", "p2", "p1", "p2"],
        "pos_abb": ["RB"] * 4,
        "pos_rank": [1, 2, 2, 1],
    }
)

# The 2024-and-earlier format: weekly rows, different column names entirely.
LEGACY = pd.DataFrame(
    {
        "season": [2024] * 5,
        "club_code": ["SF"] * 5,
        "week": [5, 5, 5, 5, 5],
        "game_type": ["REG", "REG", "REG", "REG", "POST"],
        "position": ["RB", "RB", "K", "WR", "RB"],
        "depth_position": ["RB", "KR", "K", "WR", "RB"],
        "depth_team": ["2", "1", "1", "1", "1"],
        "gsis_id": ["p1", "p1", "k1", "w1", "p9"],
    }
)


class FakeCache:
    def __init__(self, by_season):
        self.by_season = by_season

    def load(self, name, season=None, columns=None):
        try:
            return self.by_season[season]
        except KeyError:
            raise FileNotFoundError(f"no {name} for {season}") from None


# --- schema normalisation ---------------------------------------------------


def test_both_nflverse_schemas_are_normalised_to_one_shape():
    """Concatenating the raw frames drops a whole season silently; this is the guard."""
    cache = FakeCache({2024: LEGACY, 2025: MODERN})
    charts = build_depth_charts(cache, [2024, 2025], FakeCalendar())

    assert set(charts["season"]) == {2024, 2025}, "neither season may vanish"
    assert set(charts.columns) >= {"gsis_id", "season", "week", "club", "pos_abb", "pos_rank"}


def test_return_roles_do_not_promote_a_player():
    """p1 is RB2 and KR1. Taking the best rank over all his roles would call him a
    starter, but returning kicks is not offensive depth -- and doing so made 52% of
    2024 claims read as starters against 36% of 2025's, a distribution shift
    manufactured entirely by the schema difference."""
    charts = build_depth_charts(FakeCache({2024: LEGACY}), [2024], FakeCalendar())
    assert charts.set_index("gsis_id").loc["p1", "pos_rank"] == 2


def test_legacy_postseason_rows_are_excluded():
    charts = build_depth_charts(FakeCache({2024: LEGACY}), [2024], FakeCalendar())
    assert "p9" not in set(charts["gsis_id"]), "POST game_type must not be counted"


def test_kickers_survive_the_position_filter():
    """nflverse labels kickers PK in one schema and K in the other; filtering on
    one alone drops every kicker without raising."""
    charts = build_depth_charts(FakeCache({2024: LEGACY}), [2024], FakeCalendar())
    assert "k1" in set(charts["gsis_id"])


def test_a_missing_season_is_skipped_not_fatal():
    charts = build_depth_charts(FakeCache({2025: MODERN}), [2019, 2025], FakeCalendar())
    assert set(charts["season"]) == {2025}


def test_no_seasons_at_all_returns_a_usable_empty_frame():
    charts = build_depth_charts(FakeCache({}), [2024], FakeCalendar())
    assert charts.empty
    assert "pos_rank" in charts.columns, "callers index this frame even when it is empty"


def test_every_snapshot_is_kept_not_just_the_last_in_a_week():
    """Collapsing to one row per player-week is what caused the five-day leak: the
    kept snapshot sat at the far end of the week bucket while the claim sat at the
    near end. The join needs every snapshot so it can pick the right one."""
    same_week = MODERN.assign(dt=pd.to_datetime(["2025-09-10"] * 2 + ["2025-09-11"] * 2, utc=True))
    charts = build_depth_charts(FakeCache({2025: same_week}), [2025], FakeCalendar())
    assert len(charts) == 4, "both snapshots must survive"
    assert charts["dt"].notna().all()


# --- depth features ---------------------------------------------------------


CHARTS = pd.DataFrame(
    {
        "gsis_id": ["p1", "p1", "p2"],
        "season": [2025, 2025, 2025],
        "week": [4, 5, 5],
        "club": ["SF"] * 3,
        "pos_abb": ["RB"] * 3,
        "pos_rank": [3.0, 1.0, 2.0],
        "depth_size": [4.0, 4.0, 4.0],
        # No publication timestamp: the 2024-and-earlier schema, joined by week.
        "dt": [pd.NaT] * 3,
    }
)

CLAIMS = pd.DataFrame(
    {
        "gsis_id": ["p1", "p2", "p3"],
        "season": ["2025", "2025", "2025"],
        "asof_week": [5, 5, 5],
        "week": [6, 6, 6],
        "player_id": ["a", "b", "c"],
        "league_id": ["L1", "L1", "L1"],
        "created_ms": [1_700_000_000_000] * 3,
    }
)


def test_depth_rank_is_read_at_the_asof_week():
    out = attach_depth_features(CLAIMS, CHARTS)
    assert out.set_index("gsis_id").loc["p1", "depth_rank"] == 1.0


def test_promotion_compares_against_the_previous_week():
    """p1 went from third in week 4 to first in week 5: a promotion of two."""
    out = attach_depth_features(CLAIMS, CHARTS)
    assert out.set_index("gsis_id").loc["p1", "depth_promotion"] == 2.0


def test_promotion_is_null_without_a_prior_week():
    """p2 has no week-4 chart, so his movement is unknown rather than zero."""
    out = attach_depth_features(CLAIMS, CHARTS)
    assert pd.isna(out.set_index("gsis_id").loc["p2", "depth_promotion"])


def test_a_player_absent_from_the_chart_gets_nulls_not_zeros():
    out = attach_depth_features(CLAIMS, CHARTS)
    row = out.set_index("gsis_id").loc["p3"]
    assert pd.isna(row["depth_rank"]) and pd.isna(row["depth_is_starter"])


def test_starter_flag_tracks_rank_one():
    out = attach_depth_features(CLAIMS, CHARTS).set_index("gsis_id")
    assert out.loc["p1", "depth_is_starter"] == 1.0
    assert out.loc["p2", "depth_is_starter"] == 0.0


def test_row_order_is_preserved():
    """Merges reset the index; callers align the result against what they passed."""
    out = attach_depth_features(CLAIMS, CHARTS)
    assert (out.index == CLAIMS.index).all()


def test_empty_charts_still_produce_the_columns():
    out = attach_depth_features(CLAIMS, pd.DataFrame())
    assert out["depth_rank"].isna().all()


# --- as-of joining, the leak this module was rewritten to fix ----------------

# A Wednesday claim, with charts published on either side of it.
CLAIM_AT = pd.Timestamp("2025-10-08 12:00", tz="UTC")
TIMESTAMPED = pd.DataFrame(
    {
        "gsis_id": ["p1", "p1", "p1"],
        "season": [2025] * 3,
        "week": [5, 5, 5],
        "club": ["SF"] * 3,
        "pos_abb": ["RB"] * 3,
        "pos_rank": [3.0, 2.0, 1.0],
        "depth_size": [4.0, 4.0, 4.0],
        "dt": pd.to_datetime(
            ["2025-09-29 12:00", "2025-10-06 12:00", "2025-10-12 12:00"], utc=True
        ),
    }
)
TIMED_CLAIM = pd.DataFrame(
    {
        "gsis_id": ["p1"],
        "season": ["2025"],
        "asof_week": [5],
        "week": [6],
        "player_id": ["a"],
        "league_id": ["L1"],
        "created_ms": [int(CLAIM_AT.timestamp() * 1000)],
    }
)


def test_a_chart_published_after_the_claim_is_never_used():
    """The defect: labelling a snapshot with its completed week meant a claim read
    the *last* chart in a ~7-day bucket ending after the following Sunday's games.
    Measured on the corpus, 97.7% of depth features came from a chart published
    after the claim that read it, a median of 5.0 days later."""
    out = attach_depth_features(TIMED_CLAIM, TIMESTAMPED)
    # Oct 6 is the last chart at or before the Oct 8 claim; Oct 12 is the future.
    assert out["depth_rank"].iloc[0] == 2.0


def test_promotion_is_measured_against_a_week_earlier_by_clock_not_by_label():
    """Sep 29 (rank 3) is the chart a week before the claim; Oct 6 is rank 2."""
    out = attach_depth_features(TIMED_CLAIM, TIMESTAMPED)
    assert out["depth_promotion"].iloc[0] == 1.0


def test_a_claim_before_any_chart_gets_nulls():
    early = TIMED_CLAIM.assign(
        created_ms=[int(pd.Timestamp("2025-09-01", tz="UTC").timestamp() * 1000)]
    )
    out = attach_depth_features(early, TIMESTAMPED)
    assert pd.isna(out["depth_rank"].iloc[0])


def test_both_schemas_can_be_joined_in_one_call():
    """2024 rows have no timestamp and 2025 rows do; each must take its own path."""
    charts = pd.concat([CHARTS.assign(season=2024), TIMESTAMPED], ignore_index=True)
    claims = pd.concat([CLAIMS.assign(season="2024").head(1), TIMED_CLAIM], ignore_index=True)
    out = attach_depth_features(claims, charts)
    assert out["depth_rank"].notna().all(), "neither path may come back empty"


def test_an_out_of_order_claim_frame_still_joins_as_of():
    """merge_asof requires sorted keys; the claim table is not sorted by time."""
    shuffled = pd.concat([TIMED_CLAIM] * 3, ignore_index=True)
    shuffled["created_ms"] = [
        int(pd.Timestamp(t, tz="UTC").timestamp() * 1000)
        for t in ("2025-10-08", "2025-09-30", "2025-10-13")
    ]
    out = attach_depth_features(shuffled, TIMESTAMPED)
    assert list(out["depth_rank"]) == [2.0, 3.0, 1.0]


# --- injuries ---------------------------------------------------------------


INJURIES = pd.DataFrame(
    {
        "gsis_id": ["p1", "p1", "p2"],
        "season": [2025, 2025, 2025],
        "week": [5, 6, 5],
        "season_type": ["REG"] * 3,
        "report_status": ["Questionable", "Out", None],
    }
)


def test_injury_severity_is_ordered():
    status = build_injury_status(FakeCache({2025: INJURIES}), [2025])
    by_week = status.set_index(["gsis_id", "week"])["injury_severity"]
    assert by_week[("p1", 5)] < by_week[("p1", 6)], "Out is worse news than Questionable"


def test_injury_is_read_at_the_asof_week_not_later():
    """The week W+1 report is published after a claim at asof_week W. Using it
    would be lookahead: p1 is Out in week 6 but only Questionable as of week 5."""
    status = build_injury_status(FakeCache({2025: INJURIES}), [2025])
    out = attach_injury_features(CLAIMS, status).set_index("gsis_id")
    assert out.loc["p1", "injury_severity"] == 1.0, "week 6's 'Out' must not leak back"


def test_absence_from_the_report_means_healthy():
    status = build_injury_status(FakeCache({2025: INJURIES}), [2025])
    out = attach_injury_features(CLAIMS, status).set_index("gsis_id")
    assert out.loc["p2", "injury_severity"] == 0.0


def test_injury_frame_without_the_status_column_is_tolerated():
    """DataFrame.get returns its string default when a column is absent, and a bare
    string has no .eq -- that combination used to raise here."""
    assert build_injury_status(FakeCache({2025: pd.DataFrame({"gsis_id": ["p1"]})}), [2025]).empty


# --- cross-league demand ----------------------------------------------------


DEMAND_CLAIMS = pd.DataFrame(
    {
        "season": ["2025"] * 5,
        "week": [6] * 5,
        "player_id": ["a"] * 4 + ["b"],
        "league_id": ["L1", "L1", "L2", "L3", "L1"],
        "created_ms": [100, 150, 200, 300, 100],
    }
)


def test_demand_counts_leagues_not_bidders():
    """Two bidders inside one league are one league's demand, not two."""
    out = attach_cross_league_demand(DEMAND_CLAIMS)
    by_league = out.set_index("league_id")["demand_share"]
    # Three leagues are active in the week, so the shares are thirds.
    assert by_league.loc["L2"] == pytest.approx(1 / 3), "only L1 preceded it"
    assert by_league.loc["L3"] == pytest.approx(2 / 3), "L1 and L2 preceded it"


def _corpus(n_leagues: int, claiming: int) -> pd.DataFrame:
    """`claiming` of `n_leagues` claim player 'a'; the rest claim someone else."""
    return pd.DataFrame(
        {
            "season": ["2025"] * n_leagues,
            "week": [6] * n_leagues,
            "player_id": ["a"] * claiming + ["z"] * (n_leagues - claiming),
            "league_id": [f"L{i}" for i in range(n_leagues)],
            "created_ms": list(range(n_leagues)),
        }
    )


def test_demand_is_a_share_so_it_survives_the_corpus_growing():
    """Half the leagues want him, in a corpus and in one four times its size.

    As a raw count this is the same fact reported as 12 and as 49, and the price
    model trains on the season the crawler reached 174 leagues in and is scored on
    the one it reached 1,823 in. The share is the same fact either way.
    """
    small = attach_cross_league_demand(_corpus(50, 25))
    big = attach_cross_league_demand(_corpus(200, 100))

    def mean_share(frame):
        return frame[frame.player_id == "a"]["demand_share"].mean()

    assert mean_share(big) == pytest.approx(mean_share(small), abs=0.02)


def test_the_first_league_to_claim_sees_no_prior_demand():
    out = attach_cross_league_demand(DEMAND_CLAIMS)
    assert out[out.league_id == "L1"]["demand_share"].iloc[0] == 0.0


def test_a_leagues_own_later_bidders_share_its_count():
    """Both L1 rows describe one league's decision, so both carry the same demand."""
    out = attach_cross_league_demand(DEMAND_CLAIMS)
    l1 = out[(out.league_id == "L1") & (out.player_id == "a")]["demand_share"]
    assert l1.nunique() == 1


def test_demand_is_scoped_to_the_player_and_week():
    """A player claimed widely in week 2 says nothing about his price in week 10."""
    out = attach_cross_league_demand(DEMAND_CLAIMS)
    assert out[out.player_id == "b"]["demand_share"].iloc[0] == 0.0


def test_demand_preserves_row_order():
    out = attach_cross_league_demand(DEMAND_CLAIMS)
    assert (out.index == DEMAND_CLAIMS.index).all()


def test_empty_claims_do_not_crash():
    empty = DEMAND_CLAIMS.iloc[:0]
    assert attach_cross_league_demand(empty)["demand_share"].empty
