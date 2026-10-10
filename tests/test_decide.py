"""Tests for the decision layer.

The arithmetic here is short and the places it can be silently wrong are specific:
bidding full value instead of shading below it, scoring a free win at the wrong
probability, and ranking by value when the decision is about surplus. Each has its
own test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iie.models.decide import (
    ExchangeRate,
    FixedRate,
    MarketImpliedRate,
    ReplacementLevel,
    optimal_bids,
    rank_claims,
)
from iie.models.price import BidSchedule

GRID = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])


def schedule(probability, curve) -> BidSchedule:
    """A BidSchedule with a stated contest probability and conditional price curve."""
    probability = np.asarray(probability, dtype=float)
    curves = np.tile(np.asarray(curve, dtype=float), (len(probability), 1))
    return BidSchedule(probability=probability, grid=GRID, curves=curves)


# --- the trade ---------------------------------------------------------------


def test_the_bid_is_shaded_below_the_player_s_value():
    """Bidding your full valuation wins often and gains nothing, which is why a value
    model alone cannot produce a bid."""
    sched = schedule([1.0], np.linspace(0.01, 0.40, 7))
    got = optimal_bids(sched, np.array([100.0]), np.array([1000.0]))
    value_budget = 100.0 / 1000.0
    assert got["claim"].iat[0]
    assert 0 < got["bid"].iat[0] < value_budget


def test_a_player_worth_less_than_any_winning_bid_is_not_claimed():
    """Being priced out is a real answer, not a failure of the model."""
    sched = schedule([1.0], np.linspace(0.20, 0.90, 7))
    got = optimal_bids(sched, np.array([1.0]), np.array([1000.0]))
    assert not got["claim"].iat[0]
    assert got["bid"].iat[0] == 0.0


def test_expected_surplus_is_the_quantity_being_maximised():
    """Recompute it by hand at the chosen operating point."""
    sched = schedule([0.8], np.linspace(0.02, 0.50, 7))
    got = optimal_bids(sched, np.array([60.0]), np.array([600.0])).iloc[0]
    expected = got["win_prob"] * (got["value_budget"] - got["bid"])
    assert got["expected_surplus"] == pytest.approx(expected)


def test_a_more_valuable_player_earns_a_higher_bid():
    sched = schedule([1.0, 1.0], np.linspace(0.01, 0.40, 7))
    got = optimal_bids(sched, np.array([40.0, 120.0]), np.array([800.0, 800.0]))
    assert got["bid"].iat[1] > got["bid"].iat[0]


def test_a_dearer_market_buys_less_surplus_and_wins_less_often():
    """The bid itself is **not** monotone in the market price, which is worth pinning
    because the intuition runs the other way.

    Same player, value 0.1 of budget. Where prices run 0.01-0.10 the optimum bids
    0.055 and wins half the time. Where they run 0.05-0.50 the optimum bids *less*,
    0.05, because winning is no longer worth what it costs -- so the right move is a
    long shot at the cheap end of the distribution rather than a bigger bid. What
    falls monotonically is the surplus and the chance of winning, not the bid.
    """
    cheap = optimal_bids(
        schedule([1.0], np.linspace(0.01, 0.10, 7)), np.array([80.0]), np.array([800.0])
    ).iloc[0]
    dear = optimal_bids(
        schedule([1.0], np.linspace(0.05, 0.50, 7)), np.array([80.0]), np.array([800.0])
    ).iloc[0]
    assert dear["expected_surplus"] < cheap["expected_surplus"]
    assert dear["win_prob"] < cheap["win_prob"]
    assert dear["bid"] < cheap["bid"], "shades down rather than chasing"


# --- the free-win subtlety ----------------------------------------------------


def test_an_uncontested_claim_is_won_at_one_minus_the_contest_probability():
    """`at(q)` returns 0 whenever q <= 1 - p, so the bid is zero while the true win
    probability is 1 - p rather than q. Scoring those at q would understate every
    free win, and most of the board is free wins."""
    sched = schedule([0.1], np.linspace(0.05, 0.60, 7))
    got = optimal_bids(sched, np.array([50.0]), np.array([500.0])).iloc[0]
    assert got["bid"] == 0.0, "nothing to outbid"
    assert got["win_prob"] == pytest.approx(0.9), "wins whenever nobody else bids"
    assert got["expected_surplus"] == pytest.approx(0.9 * 0.1)


def test_a_certain_contest_gets_no_free_win_credit():
    sched = schedule([1.0], np.linspace(0.01, 0.40, 7))
    got = optimal_bids(sched, np.array([100.0]), np.array([1000.0])).iloc[0]
    assert got["bid"] > 0.0
    assert got["win_prob"] < 1.0


# --- exchange rate -----------------------------------------------------------


def test_fixed_and_market_rates_both_satisfy_the_protocol():
    """The rate is swappable on purpose: a dynamic-programming shadow price has to be
    able to replace the market-implied one without the caller changing."""
    assert isinstance(FixedRate(10.0), ExchangeRate)
    assert isinstance(MarketImpliedRate(), ExchangeRate)


def test_the_market_rate_is_points_per_unit_of_budget_spent():
    claims = pd.DataFrame({"won": [1, 1], "bid_pct": [0.1, 0.3], "points_gained": [20.0, 60.0]})
    rate = MarketImpliedRate().fit(claims)
    assert rate.points_per_budget(claims)[0] == pytest.approx(200.0)


def test_free_wins_are_excluded_from_the_rate():
    """They return points for no budget, so including them drives the rate toward
    infinity and makes every bid look wasteful."""
    priced = pd.DataFrame({"won": [1], "bid_pct": [0.2], "points_gained": [20.0]})
    with_free = pd.DataFrame({"won": [1, 1], "bid_pct": [0.2, 0.0], "points_gained": [20.0, 500.0]})
    assert MarketImpliedRate().fit(priced).points_per_budget(priced)[0] == pytest.approx(
        MarketImpliedRate().fit(with_free).points_per_budget(priced)[0]
    )


def test_losing_claims_are_excluded_from_the_rate():
    """A losing bid was never paid, so it bought nothing."""
    claims = pd.DataFrame({"won": [1, 0], "bid_pct": [0.1, 0.9], "points_gained": [20.0, 0.0]})
    assert MarketImpliedRate().fit(claims).points_per_budget(claims)[0] == pytest.approx(200.0)


def test_a_corpus_with_no_priced_wins_falls_back_to_the_floor():
    empty = pd.DataFrame({"won": [1], "bid_pct": [0.0], "points_gained": [5.0]})
    rate = MarketImpliedRate(floor=2.0).fit(empty)
    assert rate.points_per_budget(empty)[0] == 2.0


def test_using_the_rate_before_fitting_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        MarketImpliedRate().points_per_budget(pd.DataFrame({"a": [1]}))


def test_a_richer_rate_makes_a_player_cheaper_to_justify():
    """If a unit of budget buys more points elsewhere, this player is worth less of
    it -- so the bid falls. That is the opportunity cost doing its job."""
    sched = schedule([1.0], np.linspace(0.01, 0.40, 7))
    poor = optimal_bids(sched, np.array([80.0]), np.array([400.0]))["bid"].iat[0]
    rich = optimal_bids(sched, np.array([80.0]), np.array([4000.0]))["bid"].iat[0]
    assert rich < poor


def test_a_zero_rate_yields_no_recommendation_rather_than_dividing_by_zero():
    sched = schedule([1.0], np.linspace(0.01, 0.40, 7))
    got = optimal_bids(sched, np.array([80.0]), np.array([0.0])).iloc[0]
    assert not got["claim"]
    assert np.isnan(got["value_budget"])


# --- ranking -----------------------------------------------------------------


FRAME = pd.DataFrame(
    {
        "league_id": ["L1"] * 3,
        "season": ["2025"] * 3,
        "week": [6, 6, 6],
        "player_id": ["cheap", "star", "priced_out"],
        "position": ["RB", "WR", "TE"],
    }
)


def test_ranking_is_by_surplus_not_by_value():
    """The most valuable player is usually the wrong claim: everyone wants him and
    the price says so. The right claim is the one furthest below his worth."""
    # The star is worth more but costs far more; the cheap one is a bargain.
    sched = BidSchedule(
        probability=np.array([1.0, 1.0, 1.0]),
        grid=GRID,
        curves=np.vstack(
            [
                np.linspace(0.01, 0.05, 7),  # cheap
                np.linspace(0.30, 0.80, 7),  # star
                np.linspace(0.50, 0.95, 7),  # priced out
            ]
        ),
    )
    ranked = rank_claims(
        FRAME,
        price_schedule=sched,
        value_points=np.array([40.0, 90.0, 5.0]),
        rate=FixedRate(500.0),
    )
    assert list(ranked["player_id"]) == ["cheap", "star", "priced_out"]
    assert ranked["value_points"].iat[0] < ranked["value_points"].iat[1]
    assert not ranked["claim"].iat[2]


def test_the_ranked_list_carries_the_identifiers_a_manager_needs():
    sched = schedule([1.0, 1.0, 1.0], np.linspace(0.01, 0.20, 7))
    ranked = rank_claims(
        FRAME,
        price_schedule=sched,
        value_points=np.array([10.0, 20.0, 30.0]),
        rate=FixedRate(200.0),
    )
    for column in ("league_id", "week", "player_id", "position", "bid", "win_prob"):
        assert column in ranked.columns


def test_cumulative_spend_counts_expected_cost_not_committed_bids():
    """A bid is paid only when it wins, so the bids down a list sum to more than
    following the list actually costs."""
    sched = schedule([1.0, 1.0, 1.0], np.linspace(0.01, 0.20, 7))
    ranked = rank_claims(
        FRAME,
        price_schedule=sched,
        value_points=np.array([30.0, 30.0, 30.0]),
        rate=FixedRate(200.0),
    )
    assert (ranked["expected_spend"] <= ranked["bid"] + 1e-12).all()
    assert ranked["cumulative_expected_spend"].is_monotonic_increasing
    assert ranked["cumulative_expected_spend"].iat[-1] == pytest.approx(
        ranked["expected_spend"].sum()
    )


def test_an_unclaimed_row_costs_nothing():
    sched = schedule([1.0], np.linspace(0.50, 0.95, 7))
    ranked = rank_claims(
        FRAME.head(1),
        price_schedule=sched,
        value_points=np.array([1.0]),
        rate=FixedRate(1000.0),
    )
    assert not ranked["claim"].iat[0]
    assert ranked["expected_spend"].iat[0] == 0.0


# --- value over replacement --------------------------------------------------


def panel_with(values, positions, *, week=6, trailing=0.0) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "season": ["2025"] * len(values),
            "week": [week] * len(values),
            "position": positions,
            "fantasy_points_l3": [trailing] * len(values),
        }
    )
    return frame, np.asarray(values, dtype=float)


def test_replacement_is_the_nth_best_available_at_the_position():
    """Losing a claim drops you to the next player your leaguemates have not taken,
    not to the average of the wire."""
    frame, values = panel_with([100.0, 90.0, 80.0, 70.0, 60.0], ["TE"] * 5)
    level = ReplacementLevel(rank=3).fit(frame, values)
    # Third best available is 80, so the best TE is worth 20 more than doing nothing.
    assert level.subtract(frame.head(1), np.array([100.0]))[0] == pytest.approx(20.0)


def test_a_player_no_better_than_free_is_worth_nothing_to_claim():
    """Floored at zero, not negative: you simply do not claim him."""
    frame, values = panel_with([100.0, 90.0, 80.0], ["RB"] * 3)
    level = ReplacementLevel(rank=2).fit(frame, values)
    assert level.subtract(frame.head(1), np.array([50.0]))[0] == 0.0


def test_replacement_is_computed_per_position():
    """A thin position makes a claim worth more, which is the point of the adjustment."""
    frame, values = panel_with([100.0, 20.0, 100.0, 95.0], ["TE", "TE", "RB", "RB"])
    level = ReplacementLevel(rank=2).fit(frame, values)
    probe = frame.iloc[[0, 2]]
    got = level.subtract(probe, np.array([100.0, 100.0]))
    assert got[0] == pytest.approx(80.0), "TE replacement is 20"
    assert got[1] == pytest.approx(5.0), "RB replacement is 95"


def test_a_position_thinner_than_the_rank_uses_its_worst_available():
    """Never invents a number, and never a level better than the best available."""
    frame, values = panel_with([50.0, 30.0], ["K"] * 2)
    level = ReplacementLevel(rank=12).fit(frame, values)
    assert level.subtract(frame.head(1), np.array([50.0]))[0] == pytest.approx(20.0)


def test_an_unseen_position_week_falls_back_rather_than_dropping_the_row():
    frame, values = panel_with([100.0, 60.0, 40.0], ["WR"] * 3, week=5)
    level = ReplacementLevel(rank=2).fit(frame, values)
    later = frame.assign(week=14)
    got = level.subtract(later.head(1), np.array([100.0]))
    assert np.isfinite(got[0]) and got[0] == pytest.approx(40.0)


def test_subtracting_before_fitting_raises():
    frame, _ = panel_with([1.0], ["RB"])
    with pytest.raises(RuntimeError, match="before fit"):
        ReplacementLevel().subtract(frame, np.array([1.0]))


def test_replacement_changes_the_decision_not_just_the_number():
    """The failure it exists to prevent: with the alternative treated as zero, every
    productive player is an arbitrage. Run against real week-10 claims before this
    existed, all 50,337 candidates cleared a positive surplus."""
    frame, values = panel_with([80.0, 78.0, 76.0], ["RB"] * 3)
    level = ReplacementLevel(rank=2).fit(frame, values)
    sched = schedule([1.0], np.linspace(0.05, 0.30, 7))

    raw = optimal_bids(sched, np.array([80.0]), np.array([400.0]))
    over = optimal_bids(sched, level.subtract(frame.head(1), np.array([80.0])), np.array([400.0]))
    assert raw["claim"].iat[0], "raw points make him look worth claiming"
    assert not over["claim"].iat[0], "a 2-point edge does not justify the price"


# --- one row per player ------------------------------------------------------


def test_several_bids_on_one_player_collapse_to_one_recommendation():
    """The claim table holds one row per *bid*, so a contested player appears once
    per rival who wanted him. That is information about his price, not five separate
    decisions to make."""
    repeated = pd.DataFrame(
        {
            "league_id": ["L1"] * 4,
            "season": ["2025"] * 4,
            "week": [10] * 4,
            "player_id": ["star", "star", "star", "other"],
            "position": ["TE", "TE", "TE", "RB"],
        }
    )
    sched = schedule([1.0] * 4, np.linspace(0.01, 0.20, 7))
    ranked = rank_claims(
        repeated,
        price_schedule=sched,
        value_points=np.array([60.0, 60.0, 60.0, 30.0]),
        rate=FixedRate(300.0),
    )
    assert len(ranked) == 2
    assert list(ranked["player_id"]) == ["star", "other"]
