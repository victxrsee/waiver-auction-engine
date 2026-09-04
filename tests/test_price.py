"""Tests for the price model.

The target is subtle enough to be worth pinning down: the price is the best
*losing* bid, not the winning one, and auctions nobody won carry no price
information at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iie.models.price import (
    BidSchedule,
    ComparablesPrice,
    HurdlePrice,
    build_auction_table,
    evaluate,
    simulate_policy,
)


def claim(league, week, player, bid, won, **kw):
    row = {
        "league_id": league,
        "season": "2025",
        "week": week,
        "player_id": player,
        "bid_pct": bid,
        "won": won,
        "franchise_id": f"f{league}",
        "position": kw.get("position", "WR"),
        "asof_week": week - 1,
        "points_l3": kw.get("points_l3", 10.0),
        "is_defense": False,
        "total_rosters": 12,
        "reception_points": 0.5,
        "waiver_budget": 100.0,
    }
    return row


CLAIMS = pd.DataFrame(
    [
        # Contested: winner at 0.30, best loser at 0.12 -> price to beat is 0.12.
        claim("L1", 3, "p1", 0.30, 1),
        claim("L1", 3, "p1", 0.12, 0),
        claim("L1", 3, "p1", 0.05, 0),
        # Uncontested: nobody else bid, so the price to beat is zero.
        claim("L1", 4, "p2", 0.20, 1),
        # Nobody won: not an auction in the price sense at all.
        claim("L2", 5, "p3", 0.40, 0),
        claim("L2", 5, "p3", 0.10, 0),
    ]
)


@pytest.fixture
def auctions():
    return build_auction_table(CLAIMS)


def test_price_is_the_best_losing_bid_not_the_winning_one(auctions):
    """A winner may have overbid wildly; what mattered was beating the runner-up."""
    row = auctions[auctions.player_id == "p1"].iloc[0]
    assert row["price_to_beat"] == pytest.approx(0.12)
    assert row["winning_bid"] == pytest.approx(0.30)


def test_uncontested_auction_costs_nothing_to_win(auctions):
    row = auctions[auctions.player_id == "p2"].iloc[0]
    assert row["price_to_beat"] == 0.0
    assert row["contested"] == 0


def test_auctions_nobody_won_are_dropped(auctions):
    """A lone failed claim is a roster or budget problem, not a lost contest."""
    assert "p3" not in set(auctions.player_id)
    assert len(auctions) == 2


def test_contested_flag_matches_a_nonzero_price(auctions):
    assert auctions.set_index("player_id").loc["p1", "contested"] == 1


def test_bidder_count_is_recorded_but_stays_out_of_the_features():
    """n_bidders is unknowable at decision time and close to the target itself."""
    from iie.models.price import FEATURE_COLUMNS

    assert "n_bidders" not in FEATURE_COLUMNS
    assert "n_winners" not in FEATURE_COLUMNS


# --- baselines --------------------------------------------------------------


def test_comparables_defaults_to_the_mean_not_the_median():
    """Four auctions in five are uncontested, so a median baseline is all zeros."""
    frame = pd.DataFrame({"position": ["WR"] * 5, "week": [3] * 5})
    target = pd.Series([0.0, 0.0, 0.0, 0.0, 0.5])
    assert ComparablesPrice().fit(frame, target).predict(frame)[0] == pytest.approx(0.1)
    assert ComparablesPrice(statistic="median").fit(frame, target).predict(frame)[0] == 0.0


def test_comparables_falls_back_for_unseen_groups():
    train = pd.DataFrame({"position": ["WR"], "week": [3]})
    model = ComparablesPrice().fit(train, pd.Series([0.2]))
    unseen = pd.DataFrame({"position": ["TE"], "week": [11]})
    assert np.isfinite(model.predict(unseen)[0])


# --- hurdle -----------------------------------------------------------------


def _training_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    week = rng.integers(1, 15, n)
    points = rng.normal(10, 5, n)
    contested = rng.random(n) < (0.1 + 0.02 * points.clip(0, 20))
    price = np.where(contested, np.abs(rng.normal(0.05, 0.03, n)) + 0.001 * points, 0.0)
    return pd.DataFrame(
        {
            "week": week,
            "asof_week": week - 1,
            "points_l3": points,
            "position": rng.choice(["WR", "RB"], n),
            "is_defense": False,
            "total_rosters": 12,
            "reception_points": 0.5,
            "waiver_budget": 100.0,
        }
    ), pd.Series(price)


def test_hurdle_fits_and_predicts_nonnegative():
    frame, target = _training_frame()
    predictions = HurdlePrice().fit(frame, target).predict(frame)
    assert len(predictions) == len(frame)
    assert (predictions >= 0).all()


def test_higher_confidence_never_bids_less():
    """The bid-to-win curve must be monotone in the confidence asked for."""
    frame, target = _training_frame()
    model = HurdlePrice().fit(frame, target)
    low = model.predict_to_win(frame, 0.7)
    high = model.predict_to_win(frame, 0.95)
    assert high.mean() > low.mean()
    assert (high >= low - 1e-9).mean() > 0.95


def test_hurdle_handles_missing_player_features():
    """Defenses have no player features at all and must not crash the model."""
    frame, target = _training_frame()
    frame.loc[:50, "points_l3"] = np.nan
    assert np.isfinite(HurdlePrice().fit(frame, target).predict(frame)).all()


# --- policy scoring ---------------------------------------------------------


POLICY_AUCTIONS = pd.DataFrame({"price_to_beat": [0.0, 0.10, 0.20, 0.0]})


def test_a_tie_is_not_a_win():
    """Sleeper breaks equal bids on waiver priority, so ties are scored as losses."""
    result = simulate_policy(POLICY_AUCTIONS, np.array([0.0, 0.10, 0.25, 0.01]))
    assert result["win_rate"] == pytest.approx(0.5)


def test_contested_win_rate_excludes_the_free_wins():
    """Any positive bid wins an uncontested auction; pooling flatters every policy."""
    result = simulate_policy(POLICY_AUCTIONS, np.array([0.01, 0.15, 0.05, 0.01]))
    assert result["win_rate"] == pytest.approx(0.75)
    assert result["contested_win_rate"] == pytest.approx(0.5)


def test_spend_is_only_charged_on_wins():
    """First-price: you pay what you bid, and only when you win."""
    result = simulate_policy(POLICY_AUCTIONS, np.array([0.0, 0.50, 0.0, 0.0]))
    assert result["spend_per_win"] == pytest.approx(0.50)
    assert result["overpay_per_win"] == pytest.approx(0.40)


def test_evaluate_reports_error_and_ranking():
    truth = pd.Series([0.0, 0.1, 0.2, 0.3])
    metrics = evaluate(np.array([0.0, 0.1, 0.2, 0.3]), truth)
    assert metrics["mae"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)


# --- bid schedule -----------------------------------------------------------


def test_bid_schedule_matches_direct_prediction():
    """Precomputing the curves must not change the answer, only the cost."""
    frame, target = _training_frame()
    model = HurdlePrice().fit(frame, target)
    schedule = model.bid_schedule(frame)
    assert isinstance(schedule, BidSchedule)
    assert np.allclose(schedule.at(0.85), model.predict_to_win(frame, 0.85))


def test_bid_schedule_is_monotone_in_confidence():
    frame, target = _training_frame()
    schedule = HurdlePrice().fit(frame, target).bid_schedule(frame)
    means = [schedule.at(c).mean() for c in (0.6, 0.7, 0.8, 0.9, 0.95)]
    assert means == sorted(means)
