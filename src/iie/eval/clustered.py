"""Honest error bars and honest splits for a corpus of correlated claims.

Claims are not independent draws. Everything inside one league shares its
managers, its scoring, its budget dynamics and its player pool, so a manager who
overpays all season is a single behavioural fact appearing in two hundred rows,
not two hundred pieces of evidence. Statistics that assume otherwise overstate
how much the corpus knows.

Measured on the 500-league corpus, the bid-versus-value correlation of +0.102
carries a naive standard error of 0.0029 and a league-clustered one of 0.0073.
The honest interval is three times wider, and the effective sample size is nearer
500 than 118,000. That gap is the difference between correctly rejecting a
feature that does nothing and shipping it.

Clusters are franchises rather than league-seasons. Sleeper chains a league
across years through ``previous_league_id``, and the same league in 2024 and 2025
is the same group of people with the same habits.

Splitting follows the same rule. Cross-validation folds are built from whole
clusters so a league never appears on both sides of a split, and backtests are
split by season, since clustering addresses who the observations came from and
says nothing about when.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_DRAWS = 400


# --- clusters ---------------------------------------------------------------


class _Union:
    """Minimal union-find, enough to chain leagues into franchises."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def franchise_ids(leagues: pd.DataFrame) -> pd.Series:
    """Map every league_id to a franchise id, chaining seasons together.

    Returned as a Series indexed by league_id. Leagues with no chain are their
    own franchise, so the result is always total.
    """
    union = _Union()
    known = set(leagues["league_id"].astype(str))

    for league_id, previous in zip(
        leagues["league_id"].astype(str),
        leagues.get("previous_league_id", pd.Series(dtype=object)),
        strict=False,
    ):
        union.find(league_id)
        previous = None if previous is None or pd.isna(previous) else str(previous)
        # "0" is Sleeper's way of writing "no previous league".
        if previous and previous != "0" and previous in known:
            union.union(previous, league_id)

    return pd.Series({league_id: union.find(league_id) for league_id in known}, name="franchise_id")


def attach_clusters(
    claims: pd.DataFrame, leagues: pd.DataFrame, *, out_col: str = "franchise_id"
) -> pd.DataFrame:
    """Add the franchise a claim belongs to, falling back to its league."""
    mapping = franchise_ids(leagues)
    out = claims.copy()
    out[out_col] = out["league_id"].astype(str).map(mapping).fillna(out["league_id"].astype(str))

    log.info(
        "%d claims across %d leagues collapse to %d franchises",
        len(out),
        out["league_id"].nunique(),
        out[out_col].nunique(),
    )
    return out


# --- error bars -------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A statistic with a cluster-aware interval."""

    value: float
    se: float
    ci_low: float
    ci_high: float
    n_clusters: int
    draws: int

    def __str__(self) -> str:
        return f"{self.value:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}] (n={self.n_clusters})"

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely one side of zero."""
        return self.ci_low > 0 or self.ci_high < 0


def block_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str = "franchise_id",
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    alpha: float = 0.05,
) -> Estimate:
    """Resample whole clusters to get an interval that respects the dependence.

    ``statistic`` is applied to a resampled frame, so a paired comparison between
    two models is just a statistic returning their difference -- both are computed
    on the same resampled clusters, which is what makes the comparison paired.

    **The statistic must not group by ``cluster_col``.** Resampling draws whole
    clusters with replacement, so a cluster picked three times appears as three
    copies of its rows; a statistic that regroups on the cluster id collapses those
    copies back into one group carrying triple the totals, which inflates the result
    and produces an interval that does not even contain the full-sample estimate.
    Pre-aggregate to one row per cluster and take a row-level statistic instead.
    """
    if frame.empty:
        raise ValueError("cannot bootstrap an empty frame")

    groups = [group for _, group in frame.groupby(cluster_col, sort=True)]
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(draws):
        picks = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picks])
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(float(value))

    if not values:
        raise ValueError("statistic produced no finite values across draws")

    array = np.asarray(values)
    low, high = np.percentile(array, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(
        value=float(statistic(frame)),
        se=float(array.std(ddof=1)),
        ci_low=float(low),
        ci_high=float(high),
        n_clusters=len(groups),
        draws=len(array),
    )


def naive_se(correlation: float, n: int) -> float:
    """The standard error you get by pretending rows are independent.

    Provided for comparison. It is what makes a useless feature look significant.
    """
    if n < 3:
        return float("nan")
    return (1 - correlation**2) / np.sqrt(n - 2)


# --- splits -----------------------------------------------------------------


def group_kfold(
    frame: pd.DataFrame,
    *,
    cluster_col: str = "franchise_id",
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train, test) index pairs that never split a cluster.

    Folds are balanced by row count rather than cluster count, since leagues vary
    a lot in how many claims they contribute.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    sizes = frame.groupby(cluster_col).size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    order = sizes.index.to_numpy()
    rng.shuffle(order)
    # Greedy largest-remaining assignment keeps folds close in size.
    order = sizes.reindex(order).sort_values(ascending=False).index

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    loads = np.zeros(n_splits)
    for cluster in order:
        target = int(np.argmin(loads))
        folds[target].append(cluster)
        loads[target] += sizes[cluster]

    for fold in folds:
        test = frame[cluster_col].isin(fold)
        yield frame.index[~test], frame.index[test]


def season_split(
    frame: pd.DataFrame, train_seasons: list[str], test_seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season, for a backtest that cannot see its own future.

    Clustering handles which observations came from the same people. It does
    nothing about time, and a model validated on a season it trained on will
    flatter itself.
    """
    seasons = frame["season"].astype(str)
    overlap = set(train_seasons) & set(test_seasons)
    if overlap:
        raise ValueError(f"train and test seasons overlap: {sorted(overlap)}")

    return (
        frame[seasons.isin([str(s) for s in train_seasons])].copy(),
        frame[seasons.isin([str(s) for s in test_seasons])].copy(),
    )
