"""Tests for the value model.

Two things carry most of the weight. The first is the as-of boundary: the features
describe weeks up to and including *W* and the target describes weeks after it, and
if those windows ever overlap the model is reading its own answer. The second is the
component decomposition, which is what lets one fit serve leagues with different
scoring -- a model that silently learned one league's rules would still look fine on
that league.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iie.features.panel import add_forward, add_trailing
from iie.models.value import (
    COMPONENTS,
    FEATURE_COLUMNS,
    PositionalValue,
    ValueModel,
    add_horizon,
    evaluate_value,
    top_k_realized,
)


def panel(n_players: int = 40, weeks: int = 18, seed: int = 0) -> pd.DataFrame:
    """A rectangular player-week panel, as on_full_week_grid produces."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        skill = rng.random()
        for week in range(1, weeks + 1):
            played = int(rng.random() < 0.3 + 0.6 * skill)
            rows.append(
                {
                    "gsis_id": f"p{p}",
                    "season": 2025,
                    "week": week,
                    "position": rng.choice(["RB", "WR", "TE", "QB"]),
                    "played": played,
                    "fantasy_points": played * rng.gamma(2, 4 * skill + 0.5),
                    "receptions": played * rng.poisson(3 * skill),
                    "targets": played * rng.poisson(5 * skill),
                    "carries": played * rng.poisson(4 * skill),
                    "snap_pct": played * (0.2 + 0.7 * skill),
                }
            )
    return pd.DataFrame(rows)


def ready(**kw) -> pd.DataFrame:
    """A panel with trailing features, forward targets and the horizon attached."""
    return add_horizon(add_forward(add_trailing(panel(**kw)), 6))


# One fit covers every test that only reads predictions off a fitted model. A
# ValueModel trains eighteen boosted models (two components x classifier, mean and
# seven quantiles), so refitting per test cost the suite well over two minutes.
@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return ready()


@pytest.fixture(scope="module")
def fitted(frame) -> ValueModel:
    return ValueModel().fit(frame)


# --- the as-of boundary -----------------------------------------------------


def test_trailing_features_and_the_target_never_overlap():
    """The feature windows cover weeks <= W and the target weeks > W. If they ever
    met, the model would be reading part of its own answer -- and this repo has
    already shipped two leaks that came from a window boundary being off by one."""
    frame = ready(n_players=3)
    one = frame[frame.gsis_id == "p0"].sort_values("week")

    # Reconstruct both sides by hand for a middle week and compare.
    week = 9
    row = one[one.week == week].iloc[0]
    past = one[one.week <= week]
    future = one[(one.week > week) & (one.week <= week + 6)]

    assert row["fantasy_points_l3"] == pytest.approx(past.tail(3)["fantasy_points"].sum())
    assert row["fantasy_points_next6"] == pytest.approx(future["fantasy_points"].sum())


def test_the_final_week_has_no_future_and_says_so():
    frame = ready(n_players=3)
    last = frame[frame.week == 18]
    assert (last["weeks_remaining"] == 0).all()
    assert (last["fantasy_points_next6"] == 0).all()


def test_weeks_remaining_is_the_window_length_not_the_weeks_left():
    """`18 - week` is a strictly decreasing function of `week`, which is already a
    feature, so it told a tree ensemble nothing: the two measured at correlation
    -1.000 and an arm given both scored bit-identically to an arm given one. It was
    also plainly wrong as described, claiming a 17-week window at week 1.

    Clipping at the horizon is what makes it informative -- flat where the window is
    full, falling only where it truly truncates, which `week` cannot express.
    """
    frame = ready(n_players=2)
    expected = (18 - frame["week"]).clip(lower=0, upper=6)
    assert (frame["weeks_remaining"] == expected).all()
    assert (frame.loc[frame.week <= 12, "weeks_remaining"] == 6).all(), "flat while full"
    assert frame.loc[frame.week == 16, "weeks_remaining"].iat[0] == 2
    # Not a monotone transform of week, which is the whole point.
    early = frame[frame.week.between(1, 12)]
    assert early["weeks_remaining"].nunique() == 1 and early["week"].nunique() == 12


def test_no_forward_column_is_a_feature():
    """A single forward column in the feature list would make the model perfect and
    useless. Cheap to assert, and the kind of thing a refactor introduces."""
    assert not [f for f in FEATURE_COLUMNS if "next" in f]


# --- component decomposition ------------------------------------------------


def test_zero_reception_points_returns_the_points_component_alone(frame, fitted):
    model = fitted
    parts = model.predict_components(frame)
    assert np.allclose(model.predict_points(frame, 0.0), parts["fantasy_points"])


def test_one_fit_serves_leagues_with_different_scoring(frame, fitted):
    """The whole reason components are predicted separately. A full-PPR league must
    value a receiver above what a zero-PPR league does, from the same fit."""
    model = fitted
    standard = model.predict_points(frame, 0.0)
    ppr = model.predict_points(frame, 1.0)
    parts = model.predict_components(frame)
    assert np.allclose(ppr - standard, parts["receptions"])
    assert (ppr >= standard - 1e-9).all(), "receptions cannot subtract value"


def test_reception_points_may_vary_per_row(frame, fitted):
    """A backtest spans many leagues at once, so the rate has to be a series too."""
    model = fitted
    rates = pd.Series(np.where(np.arange(len(frame)) % 2 == 0, 0.0, 1.0), index=frame.index)
    mixed = model.predict_points(frame, rates)
    assert np.allclose(mixed[::2], model.predict_points(frame, 0.0)[::2])
    assert np.allclose(mixed[1::2], model.predict_points(frame, 1.0)[1::2])


def test_every_component_gets_its_own_hurdle(frame, fitted):
    model = fitted
    parts = model.predict_components(frame)
    assert set(parts.columns) == set(COMPONENTS)


# --- horizon conditioning ---------------------------------------------------


def test_the_model_predicts_less_late_in_the_season(frame, fitted):
    """Not a bug to correct but the quantity to capture: a week-16 claim really does
    buy two more weeks, and an engine blind to that overpays for it every time."""
    model = fitted
    predicted = model.predict_points(frame, 0.5)
    early = predicted[frame["week"].to_numpy() <= 6].mean()
    late = predicted[frame["week"].to_numpy() >= 16].mean()
    assert late < early


# --- quantiles --------------------------------------------------------------


def test_quantiles_are_monotone_in_the_level_asked_for(frame, fitted):
    model = fitted
    low = model.predict_quantile(frame, 0.5)
    high = model.predict_quantile(frame, 0.9)
    assert high.mean() > low.mean()
    assert (high >= low - 1e-9).mean() > 0.95


def test_a_quantile_inside_the_spike_at_zero_is_zero(frame, fitted):
    """A share ``1 - p`` of outcomes sit at exactly zero, so any quantile below that
    falls inside the spike and the honest answer is zero -- reporting a positive
    number there would invent production the player never had.

    Asserted as the invariant rather than as a share of rows: how many rows land in
    the spike is a fact about the data (~80% zeros in the real panel) and not about
    this code, so a fixture with livelier players must not be able to hide a bug.
    """
    model = fitted
    probability, _, _ = model._hurdles["fantasy_points"].quantile_curves(frame)

    for quantile in (0.02, 0.1, 0.3):
        got = model.predict_quantile(frame, quantile)
        inside = quantile <= 1 - probability
        assert (got[inside] == 0.0).all(), f"q={quantile} inside the spike must be 0"
        if inside.sum() < len(frame):
            assert (got[~inside] > 0.0).all(), f"q={quantile} above the spike must be positive"


# --- contract -----------------------------------------------------------------


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        ValueModel().predict_components(ready(n_players=2))


def test_a_panel_without_forward_targets_raises_with_the_fix():
    """Forgetting add_forward would otherwise train on a KeyError deep in sklearn."""
    bare = add_horizon(add_trailing(panel(n_players=3)))
    with pytest.raises(KeyError, match="add_forward"):
        ValueModel().fit(bare)


def test_predictions_keep_the_frame_index(frame, fitted):
    shuffled = frame.sample(frac=1.0, random_state=1)
    assert (fitted.predict_components(shuffled).index == shuffled.index).all()


def test_a_declared_feature_that_is_absent_raises():
    frame = ready(n_players=3)
    with pytest.raises(KeyError, match="declared features absent"):
        ValueModel().fit(frame.drop(columns=["snap_pct_delta"]))


# --- baseline and scoring -----------------------------------------------------


def test_the_baseline_uses_the_mean_not_the_median():
    """The target is ~80% zeros, so a median baseline predicts zero everywhere and
    is indistinguishable from refusing to play."""
    frame = ready(n_players=20)
    predicted = PositionalValue().fit(frame).predict_points(frame, 0.5)
    assert predicted.mean() > 0


def test_the_baseline_falls_back_for_an_unseen_group():
    frame = ready(n_players=20)
    model = PositionalValue().fit(frame[frame.position == "RB"])
    assert np.isfinite(model.predict_points(frame, 0.5)).all()


def test_the_model_beats_the_baseline_on_within_week_ranking():
    """Ranking inside the set a manager chooses between is the real task; a pooled
    correlation is flattered by easy contrasts between early and late weeks."""
    train = ready(n_players=40, seed=1)
    test = ready(n_players=40, seed=2)
    truth = test["fantasy_points_next6"] + 0.5 * test["receptions_next6"]

    base = evaluate_value(
        PositionalValue().fit(train).predict_points(test, 0.5), truth, groups=test["week"]
    )
    model = evaluate_value(
        ValueModel().fit(train).predict_points(test, 0.5), truth, groups=test["week"]
    )
    assert model["spearman_within_group"] > base["spearman_within_group"]
    assert model["mae"] < base["mae"]


def test_a_group_with_no_ranking_information_is_skipped_not_scored_zero():
    """Week 18's outcomes are all zero, so its correlation is undefined. Counting it
    as zero would drag the reported ranking down by a constant that is not a result."""
    truth = pd.Series([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    groups = pd.Series(["dead", "dead", "dead", "live", "live", "live"])
    out = evaluate_value(np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]), truth, groups=groups)
    assert out["groups_ranked"] == 1
    assert out["spearman_within_group"] == pytest.approx(1.0)


# --- the decision metric, and the trap it replaced ---------------------------


def test_top_k_averages_the_outcome_of_the_rows_it_would_have_picked():
    truth = pd.Series([10.0, 0.0, 5.0, 0.0])
    groups = pd.Series(["w1"] * 4)
    # Ranks row 0 first and row 2 second, which realise 10 and 5.
    got = top_k_realized(np.array([4.0, 1.0, 3.0, 2.0]), truth, groups, k=2)
    assert got["top_k"] == pytest.approx(7.5)


def test_a_perfect_ranking_reaches_the_oracle():
    truth = pd.Series([1.0, 9.0, 3.0, 7.0])
    groups = pd.Series(["w1"] * 4)
    got = top_k_realized(truth.to_numpy(), truth, groups, k=2)
    assert got["top_k"] == pytest.approx(got["oracle"])


def test_the_oracle_and_floor_bracket_the_result():
    """A bare "27.5 points" means nothing without the best and average available."""
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.gamma(2, 3, 200))
    groups = pd.Series(np.repeat(["w1", "w2"], 100))
    got = top_k_realized(rng.random(200), truth, groups, k=5)
    assert got["floor"] <= got["top_k"] <= got["oracle"]


def test_a_group_smaller_than_k_is_skipped():
    truth = pd.Series([1.0, 2.0, 5.0, 6.0, 7.0])
    groups = pd.Series(["small", "small", "big", "big", "big"])
    got = top_k_realized(np.arange(5.0), truth, groups, k=3)
    assert got["groups_scored"] == 1


def test_declining_to_discriminate_beats_spearman_and_loses_the_decision():
    """The trap that made a one-line heuristic look better than the fitted model.

    Most of a waiver pool returns nothing, so a predictor that leaves the whole
    undifferentiated mass tied is scored on *matched* ties and rates well on rank --
    while picking no better than chance. Measured on the real 2025 pool, trailing
    production left 91% of rows in one tie and scored 0.780 against the model's
    0.656, yet its top-5 claims returned 23.5 points against 27.5.
    """
    rng = np.random.default_rng(0)
    n = 600
    truth = pd.Series(np.where(rng.random(n) < 0.88, 0.0, rng.gamma(3, 6, n)))
    groups = pd.Series(np.repeat([f"w{i}" for i in range(6)], n // 6))

    # Leaves every row tied; cannot pick, but matches the mass of zeros.
    blunt = np.zeros(n)
    # Orders the producers correctly and the rest arbitrarily.
    sharp = truth.to_numpy() + rng.normal(0, 2, n)

    blunt_rho = evaluate_value(blunt, truth, groups=groups)["spearman_within_group"]
    sharp_rho = evaluate_value(sharp, truth, groups=groups)["spearman_within_group"]
    assert np.isnan(blunt_rho), "a constant has no rank correlation at all"
    assert sharp_rho > 0.5

    blunt_pick = top_k_realized(blunt, truth, groups, k=5)["top_k"]
    sharp_pick = top_k_realized(sharp, truth, groups, k=5)["top_k"]
    assert sharp_pick > blunt_pick, "top-k must reward actually choosing well"
