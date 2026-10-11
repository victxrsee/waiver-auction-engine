"""Tests for cluster-aware evaluation.

The properties worth guarding: chained league-seasons collapse into one franchise,
resampling respects clusters, and no split ever puts a cluster on both sides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iie.eval.clustered import (
    Estimate,
    attach_clusters,
    block_bootstrap,
    franchise_ids,
    group_kfold,
    naive_se,
    season_split,
)

LEAGUES = pd.DataFrame(
    {
        # A chains 2023 -> 2024 -> 2025; B stands alone; C points outside the corpus.
        "league_id": ["A23", "A24", "A25", "B25", "C25"],
        "season": ["2023", "2024", "2025", "2025", "2025"],
        "previous_league_id": [None, "A23", "A24", "0", "GONE"],
    }
)


def test_chained_seasons_collapse_into_one_franchise():
    """The same league across years is the same people with the same habits."""
    ids = franchise_ids(LEAGUES)
    assert ids["A23"] == ids["A24"] == ids["A25"]
    assert ids["B25"] != ids["A23"]


def test_unchained_leagues_are_their_own_franchise():
    ids = franchise_ids(LEAGUES)
    assert len({ids["A23"], ids["B25"], ids["C25"]}) == 3


def test_previous_league_outside_the_corpus_is_ignored():
    """C25 points at a league we never crawled; it must not vanish or link wrongly."""
    assert franchise_ids(LEAGUES)["C25"] == "C25"


def test_mapping_is_total():
    assert set(franchise_ids(LEAGUES).index) == set(LEAGUES.league_id)


def test_attach_clusters_reduces_the_group_count():
    claims = pd.DataFrame({"league_id": ["A23", "A24", "A25", "B25"], "x": [1, 2, 3, 4]})
    out = attach_clusters(claims, LEAGUES)
    assert out["league_id"].nunique() == 4
    assert out["franchise_id"].nunique() == 2


def test_unknown_league_falls_back_to_itself():
    claims = pd.DataFrame({"league_id": ["ZZZ"], "x": [1]})
    assert attach_clusters(claims, LEAGUES)["franchise_id"].iloc[0] == "ZZZ"


# --- bootstrap --------------------------------------------------------------


def _clustered_frame(n_clusters=40, per=25, seed=0, slope=0.2, cluster_sd=3.0):
    """Rows correlated within cluster: the situation naive errors mishandle."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        offset = rng.normal(0, cluster_sd)  # a cluster-level habit shared by every row
        for _ in range(per):
            x = rng.normal()
            rows.append({"franchise_id": f"f{c}", "x": x, "y": x * slope + offset})
    return pd.DataFrame(rows)


def _corr(frame):
    return frame["x"].corr(frame["y"])


def test_bootstrap_returns_an_interval_around_the_estimate():
    frame = _clustered_frame()
    est = block_bootstrap(frame, _corr, draws=200)
    assert isinstance(est, Estimate)
    assert est.ci_low < est.value < est.ci_high
    assert est.n_clusters == 40


def test_clustered_interval_is_wider_than_the_naive_one():
    """The entire point: shared cluster effects are not independent evidence."""
    frame = _clustered_frame()
    est = block_bootstrap(frame, _corr, draws=200)
    assert est.se > naive_se(est.value, len(frame)), "clustering must not shrink uncertainty"


def test_bootstrap_is_reproducible():
    frame = _clustered_frame()
    a = block_bootstrap(frame, _corr, draws=100, seed=7)
    b = block_bootstrap(frame, _corr, draws=100, seed=7)
    assert (a.se, a.ci_low, a.ci_high) == (b.se, b.ci_low, b.ci_high)


def test_paired_difference_is_computed_on_the_same_resample():
    """A model comparison must resample both models' rows together."""
    frame = _clustered_frame()
    est = block_bootstrap(frame, lambda f: _corr(f) - _corr(f), draws=50)
    assert est.value == 0.0 and est.se == 0.0


def test_excludes_zero_flags_a_real_effect():
    frame = _clustered_frame(slope=2.0)
    assert block_bootstrap(frame, _corr, draws=200).excludes_zero


def test_large_cluster_effects_swamp_a_weak_pooled_signal():
    """A real within-cluster slope can still be undetectable pooled.

    With cluster offsets far larger than the effect, the pooled correlation is
    genuinely uninformative -- and the interval must say so rather than certify
    the noise. This is the case naive errors get confidently wrong.
    """
    frame = _clustered_frame(slope=0.2, cluster_sd=3.0)
    est = block_bootstrap(frame, _corr, draws=200)
    assert not est.excludes_zero
    assert naive_se(est.value, len(frame)) < est.se


def test_empty_frame_is_an_error():
    with pytest.raises(ValueError, match="empty frame"):
        block_bootstrap(pd.DataFrame(), _corr)


# --- splits -----------------------------------------------------------------


def test_kfold_never_splits_a_cluster():
    """A league on both sides of a split leaks its quirks into validation."""
    frame = _clustered_frame(n_clusters=20, per=10)
    for train, test in group_kfold(frame, n_splits=4):
        assert not set(frame.loc[train, "franchise_id"]) & set(frame.loc[test, "franchise_id"])


def test_kfold_covers_every_row_exactly_once_across_folds():
    frame = _clustered_frame(n_clusters=20, per=10)
    seen = pd.Index([])
    for _, test in group_kfold(frame, n_splits=4):
        seen = seen.append(test)
    assert sorted(seen) == sorted(frame.index)


def test_kfold_folds_are_roughly_balanced():
    """Leagues differ a lot in size, so balancing by rows matters."""
    frame = _clustered_frame(n_clusters=40, per=10)
    sizes = [len(test) for _, test in group_kfold(frame, n_splits=5)]
    assert max(sizes) - min(sizes) <= 0.25 * np.mean(sizes)


def test_kfold_rejects_a_single_split():
    with pytest.raises(ValueError, match="at least 2"):
        list(group_kfold(_clustered_frame(), n_splits=1))


def test_season_split_separates_years():
    frame = pd.DataFrame({"season": ["2024", "2024", "2025"], "x": [1, 2, 3]})
    train, test = season_split(frame, ["2024"], ["2025"])
    assert train["x"].tolist() == [1, 2]
    assert test["x"].tolist() == [3]


def test_season_split_refuses_overlap():
    frame = pd.DataFrame({"season": ["2024"], "x": [1]})
    with pytest.raises(ValueError, match="overlap"):
        season_split(frame, ["2024"], ["2024"])


def test_a_statistic_that_regroups_on_the_cluster_id_is_inflated():
    """A hazard worth a regression test, because it produced an interval that did
    not contain its own point estimate (+704 true, against [+1071, +1155]).

    Resampling draws whole clusters with replacement, so a cluster picked three
    times appears as three copies of its rows. A statistic that then groups on the
    cluster id collapses those copies into one group carrying triple the totals, so
    the resampled distribution runs far above the truth while the full-sample
    estimate stays correct -- which is why the symptom was a bad interval rather
    than an obviously wrong number.

    Note that `sum() / nunique()` is **also** wrong, and in the same direction: the
    numerator grows with duplicates while the denominator shrinks. Pre-aggregating
    to one row per cluster is the only form that resamples what it claims to.
    """
    clusters = pd.DataFrame(
        {
            "franchise_id": np.repeat(["f1", "f2", "f3", "f4"], 3),
            "value": np.repeat([1.0, 3.0, 7.0, 9.0], 3),
        }
    )
    totals = clusters.groupby("franchise_id", as_index=False)["value"].sum()

    bad = block_bootstrap(
        clusters, lambda s: float(s.groupby("franchise_id")["value"].sum().mean()), draws=400
    )
    good = block_bootstrap(totals, lambda s: float(s["value"].mean()), draws=400)

    truth = float(totals["value"].mean())
    assert bad.value == pytest.approx(truth) == pytest.approx(good.value)
    assert good.ci_low <= truth <= good.ci_high, "the honest form brackets the truth"
    assert bad.ci_high > 2 * good.ci_high - truth, "merged duplicates run far above it"


def test_pre_aggregating_to_one_row_per_cluster_is_the_fix():
    """With one row per cluster, resampling rows *is* resampling clusters."""
    totals = pd.DataFrame({"franchise_id": ["f1", "f2", "f3", "f4"], "gap": [2.0, 4.0, 6.0, 8.0]})
    got = block_bootstrap(totals, lambda f: float(f["gap"].mean()), draws=200)
    assert got.ci_low <= got.value <= got.ci_high, "the interval must contain its estimate"
    assert got.value == pytest.approx(5.0)
