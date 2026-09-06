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
    _design_matrix,
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


def _training_frame(n=400, seed=0, positions=("WR", "RB")):
    """A frame carrying every declared feature, as production does.

    Fixtures that carried only a few features are why a declared-but-absent
    feature went unnoticed in production for as long as it did.
    """
    rng = np.random.default_rng(seed)
    week = rng.integers(1, 15, n)
    points = rng.normal(10, 5, n)
    contested = rng.random(n) < (0.1 + 0.02 * points.clip(0, 20))
    price = np.where(contested, np.abs(rng.normal(0.05, 0.03, n)) + 0.001 * points, 0.0)
    return pd.DataFrame(
        {
            "week": week,
            "asof_week": week - 1,
            "points_l1": points / 3,
            "points_l3": points,
            "snap_pct_l1": rng.random(n),
            "snap_pct_l3": rng.random(n),
            "snap_pct_delta": rng.normal(0, 0.1, n),
            "targets_l1": rng.integers(0, 5, n),
            "targets_l3": rng.integers(0, 15, n),
            "carries_l3": rng.integers(0, 20, n),
            "receptions_l3": rng.integers(0, 12, n),
            "played_l3": rng.integers(0, 4, n),
            "position": rng.choice(list(positions), n),
            "is_defense": False,
            "total_rosters": 12,
            "reception_points": 0.5,
            "waiver_budget": 100.0,
            "depth_rank": rng.integers(1, 5, n).astype(float),
            "depth_size": rng.integers(2, 7, n).astype(float),
            "depth_promotion": rng.integers(-1, 2, n).astype(float),
            "depth_is_starter": (rng.random(n) < 0.3).astype(float),
            "demand_prior": rng.integers(0, 60, n).astype(float),
            "injury_severity": rng.integers(0, 4, n).astype(float),
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


def test_a_tie_on_a_contested_auction_is_not_a_win():
    """Sleeper breaks equal bids on waiver priority, so contested ties are losses."""
    # prices [0.0, 0.10, 0.20, 0.0]; bids tie on #1 and clear #2.
    result = simulate_policy(POLICY_AUCTIONS, np.array([0.0, 0.10, 0.25, 0.01]))
    # Both uncontested auctions won, #2 won on price, #1 lost the tie.
    assert result["win_rate"] == pytest.approx(0.75)
    assert result["contested_win_rate"] == pytest.approx(0.5)


def test_an_uncontested_auction_is_won_by_any_bid_including_zero():
    """There is no rival to outbid. Scoring these with a strict > discards every
    claim a policy correctly wins for free -- most of them, since four auctions in
    five are uncontested."""
    result = simulate_policy(POLICY_AUCTIONS, np.zeros(4))
    assert result["win_rate"] == pytest.approx(0.5), "both uncontested auctions won"
    assert result["contested_win_rate"] == pytest.approx(0.0)
    assert result["spend_per_win"] == pytest.approx(0.0), "free wins cost nothing"


def test_contested_win_rate_excludes_the_free_wins():
    """Any positive bid wins an uncontested auction; pooling flatters every policy."""
    result = simulate_policy(POLICY_AUCTIONS, np.array([0.01, 0.15, 0.05, 0.01]))
    assert result["win_rate"] == pytest.approx(0.75)
    assert result["contested_win_rate"] == pytest.approx(0.5)


def test_spend_is_only_charged_on_wins():
    """First-price: you pay what you bid, and only when you win."""
    contested_only = pd.DataFrame({"price_to_beat": [0.10, 0.20]})
    result = simulate_policy(contested_only, np.array([0.50, 0.05]))
    assert result["win_rate"] == pytest.approx(0.5), "0.05 does not clear 0.20"
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


# --- fit/predict on disjoint frames ------------------------------------------
#
# The only thing the model is ever asked to do in production, and the one thing no
# test did. Its absence is why an unstable categorical encoding survived: every
# other test fits and predicts on the same frame, where the bug is invisible.


def test_categorical_codes_are_stable_across_disjoint_frames():
    """A split missing a rare position must not renumber the others.

    Deriving categories per-frame meant WR was code 17 at fit and 14 at predict,
    so the model scored every skill position under a label it never learned.
    """
    train, target = _training_frame(positions=("WR", "RB", "TE", "QB"), seed=1)
    # Test data lacking QB entirely -- the real 2024/2025 split shape.
    test, _ = _training_frame(n=120, positions=("WR", "RB", "TE"), seed=2)

    model = HurdlePrice().fit(train, target)
    train_codes = _design_matrix(train, model._vocabulary)["position"]
    test_codes = _design_matrix(test, model._vocabulary)["position"]

    def mapping(frame, codes):
        return dict(
            pd.Series(codes.to_numpy(), index=frame["position"].to_numpy()).groupby(level=0).first()
        )

    shared = mapping(train, train_codes)
    for position, code in mapping(test, test_codes).items():
        assert shared[position] == code, f"{position} encoded differently at predict time"


def test_an_unseen_category_becomes_nan_not_a_collision():
    """A position absent from training must not be silently aliased onto a known one."""
    train, target = _training_frame(positions=("WR", "RB"), seed=1)
    model = HurdlePrice().fit(train, target)

    novel = pd.DataFrame([{**train.iloc[0].to_dict(), "position": "LS"}])
    assert pd.isna(_design_matrix(novel, model._vocabulary)["position"].iloc[0])


def test_single_row_inference_matches_the_batch():
    """The actual product call is one player at a time. A per-frame encoding gave
    that row code 0 regardless of position, so it disagreed with the batch."""
    train, target = _training_frame(positions=("WR", "RB", "TE"), seed=1)
    test, _ = _training_frame(n=40, positions=("WR", "RB", "TE"), seed=3)

    model = HurdlePrice().fit(train, target)
    batch = model.predict(test)
    one_at_a_time = np.array([model.predict(test.iloc[[i]])[0] for i in range(len(test))])
    assert np.allclose(batch, one_at_a_time)


def test_a_declared_feature_that_is_absent_raises():
    """Silently training on fewer features than declared is how one goes missing."""
    frame, _ = _training_frame()
    with pytest.raises(KeyError, match="declared features absent"):
        _design_matrix(frame.drop(columns=["total_rosters"]))
