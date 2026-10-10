"""The decision layer: what to claim, and what to bid.

V says what a player is worth in points. P says what it costs in budget to outbid
everyone else. Neither alone is a decision. This combines them into a ranked claim
list with bid amounts, which is the thing the project set out to produce.

**The trade.** Bidding ``b`` on a player worth ``v`` points wins with probability
``w(b)`` and costs ``b`` only when it wins -- a first-price auction is paid at the
amount bid, and nothing is paid for losing. So the expected gain, in budget units, is

    w(b) * (v / rate - b)

where ``rate`` converts points into budget. Maximising that over ``b`` is the whole
decision, and the maximum sits strictly below ``v / rate``: bidding your full
valuation wins often and gains nothing, which is why a value model alone cannot
produce a bid.

**Why this is a sweep and not calculus.** ``HurdlePrice.bid_schedule`` already gives
the inverse curve, the bid required to win at each confidence, for every row at once.
So the optimum is found by walking that curve rather than by differentiating an
assumed functional form, and it costs array arithmetic.

**The exchange rate is deliberately swappable.** FAAB is only redeemable in this one
market, so there is no external price for it; a rate has to come from somewhere.
``MarketImpliedRate`` reads it off what leagues actually paid, which is defensible
and has a known flaw: the market overpays for recency and hype -- measured at +0.270
correlation with price against +0.029 with realised value -- so a market-derived
rate inherits that error as its reference point. It still finds *relative*
mispricing, which is most of the available edge in a fixed-budget auction. What it
cannot do is decide to sit a week out and bank budget for later; that needs the
opportunity cost of budget over the rest of the season, which is a dynamic program.
Hence the protocol: a ``ShadowPriceRate`` can replace it without the caller changing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .price import BidSchedule

log = logging.getLogger(__name__)

# Operating points swept when searching for the best bid. Finer than the price
# model's quantile grid on purpose: the curve is interpolated between grid knots, so
# a fine sweep costs only arithmetic and avoids landing on a knot by accident.
DEFAULT_SWEEP = np.round(np.arange(0.02, 0.999, 0.005), 4)


@runtime_checkable
class ExchangeRate(Protocol):
    """Points per unit of budget: what a point is worth in FAAB.

    The one number that lets value and price be compared. Implementations differ in
    what they consider the cost of budget -- what the market charged, or what the
    rest of the season could have bought with it.
    """

    def points_per_budget(self, context: pd.DataFrame) -> np.ndarray:
        """One rate per row of ``context``, so it may vary by league or by week."""
        ...


@dataclass(frozen=True)
class FixedRate:
    """A constant rate. For tests, and for asking "what if a point were worth X"."""

    rate: float

    def points_per_budget(self, context: pd.DataFrame) -> np.ndarray:
        return np.full(len(context), float(self.rate))


@dataclass
class MarketImpliedRate:
    """The rate leagues actually paid: realised points per unit of budget spent.

    Fitted on **won claims that cost something**. Free wins are excluded
    deliberately: they return points for no budget, so including them drives the
    rate toward infinity and makes every bid look wasteful. The question this
    answers is "when budget was actually spent, what did a unit of it buy", which is
    the opportunity cost a bid is traded against.

    Known limitation, stated rather than hidden: this is the market's own rate, and
    the market overpays. A rate derived from it is therefore too generous about what
    budget should buy. It still ranks claims correctly relative to one another, which
    is where the edge is; it is the absolute level that is suspect. See the module
    docstring.
    """

    floor: float = 1.0
    _rate: float | None = None

    def fit(self, claims: pd.DataFrame) -> MarketImpliedRate:
        won = claims[(claims["won"] == 1) & (claims["bid_pct"] > 0)]
        spent = float(won["bid_pct"].sum())
        gained = float(won["points_gained"].sum())
        if spent <= 0 or not np.isfinite(gained):
            log.warning("no priced wins to fit an exchange rate on; using the floor")
            self._rate = self.floor
        else:
            self._rate = max(gained / spent, self.floor)
        log.info(
            "market-implied rate: %.1f points per unit of budget (%d priced wins)",
            self._rate,
            len(won),
        )
        return self

    def points_per_budget(self, context: pd.DataFrame) -> np.ndarray:
        if self._rate is None:
            raise RuntimeError("MarketImpliedRate.points_per_budget called before fit")
        return np.full(len(context), self._rate)


@dataclass
class ReplacementLevel:
    """Points obtainable for nothing, so a claim is valued as a gain and not a total.

    **Without this the engine is nonsense, and the way it fails is instructive.** A
    manager who claims a player projected for 40 points does not gain 40 points --
    they gain 40 minus whatever they would have started instead. Treating the
    alternative as zero makes every productive player an arbitrage: run against real
    week-10 claims, *every one* of 50,337 candidates cleared a positive surplus,
    which is a statement about the units rather than a discovery.

    Defined by **rank, not by quantile**. If you lose a claim on the best available
    tight end, you do not fall to the average of the waiver wire -- you fall to
    roughly the next one your leaguemates have not taken, so replacement is the
    ``rank``-th best available player at that position that week. A quantile over
    the available pool is the wrong instrument because that pool is padded with
    thousands of players who never play: measured, its 90th percentile is 0.0 points
    for most position-weeks, so subtracting it changes almost nothing.

    It is a *decision* concern rather than a valuation one, which is why it lives
    here: the player's projection does not change, only what claiming him is worth
    to you. Falls back to the position's overall level, then to zero, where the panel
    does not cover a position-week.
    """

    rank: int = 12
    pool_threshold: float = 6.0
    pool_window: int = 3
    _levels: pd.Series | None = None
    _by_position: pd.Series | None = None

    def fit(self, panel: pd.DataFrame, value_points: np.ndarray) -> ReplacementLevel:
        frame = pd.DataFrame(
            {
                "season": panel["season"].astype(str).to_numpy(),
                "week": panel["week"].to_numpy(),
                "position": panel.get(
                    "position", pd.Series(index=panel.index, dtype=object)
                ).to_numpy(),
                "value": np.asarray(value_points, dtype=float),
            }
        )
        available = frame[
            panel[f"fantasy_points_l{self.pool_window}"].to_numpy() <= self.pool_threshold
        ]
        if available.empty:
            available = frame

        def nth_best(values: pd.Series) -> float:
            ordered = values.sort_values(ascending=False).to_numpy()
            if len(ordered) == 0:
                return float("nan")
            # The rank-th best, or the worst available if the position is thinner
            # than that -- never better than the best, never an invented number.
            return float(ordered[min(self.rank, len(ordered)) - 1])

        self._levels = available.groupby(["season", "week", "position"], observed=True)[
            "value"
        ].agg(nth_best)
        self._by_position = available.groupby("position", observed=True)["value"].agg(nth_best)
        log.info(
            "replacement level (rank %d available): %d position-weeks, median %.1f points",
            self.rank,
            len(self._levels),
            float(self._levels.median()) if len(self._levels) else float("nan"),
        )
        return self

    def subtract(self, frame: pd.DataFrame, value_points: np.ndarray) -> np.ndarray:
        """``value_points`` restated as a gain over the free alternative, floored at 0."""
        if self._levels is None:
            raise RuntimeError("ReplacementLevel.subtract called before fit")
        index = pd.MultiIndex.from_arrays(
            [
                frame["season"].astype(str),
                frame["week"],
                frame.get("position", pd.Series(index=frame.index, dtype=object)),
            ]
        )
        level = self._levels.reindex(index).to_numpy(dtype=float)
        fallback = (
            self._by_position.reindex(
                frame.get("position", pd.Series(index=frame.index, dtype=object))
            ).to_numpy(dtype=float)
            if self._by_position is not None
            else np.zeros(len(frame))
        )
        level = np.where(np.isfinite(level), level, fallback)
        level = np.where(np.isfinite(level), level, 0.0)
        # A player no better than what is free is worth nothing to claim, not a
        # negative amount -- you simply do not claim him.
        return np.clip(np.asarray(value_points, dtype=float) - level, 0.0, None)


def optimal_bids(
    schedule: BidSchedule,
    value_points: np.ndarray,
    rate: np.ndarray,
    *,
    sweep: np.ndarray | None = None,
) -> pd.DataFrame:
    """The surplus-maximising bid for each row, and what it is expected to gain.

    Returns ``bid``, ``win_prob``, ``value_budget``, ``expected_surplus`` and
    ``claim`` -- the last false where no bid has positive expected surplus, which is
    a real answer and not a failure: a player can be correctly priced out.

    The subtlety worth stating, because getting it wrong is invisible. ``at(q)``
    returns exactly zero whenever ``q <= 1 - p``: the chance of facing nobody
    already covers the confidence asked for, so a minimal bid suffices. At those
    operating points the *true* win probability is ``1 - p``, not ``q`` -- a zero bid
    wins precisely the uncontested auctions. Scoring them at ``q`` would understate
    every free win, which is most of the board. Hence the ``maximum``.
    """
    grid = DEFAULT_SWEEP if sweep is None else np.asarray(sweep, dtype=float)
    value_points = np.asarray(value_points, dtype=float)
    rate = np.asarray(rate, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        value_budget = np.where(rate > 0, value_points / rate, np.nan)

    # (rows x operating points). Built once; the search is then a row-wise argmax.
    bids = np.column_stack([schedule.at(q) for q in grid])
    uncontested = 1.0 - np.asarray(schedule.probability, dtype=float)
    wins = np.maximum(grid[None, :], uncontested[:, None])
    surplus = wins * (value_budget[:, None] - bids)

    # A row that is all-NaN (no rate) must not pick an arbitrary column.
    usable = np.isfinite(surplus)
    surplus_filled = np.where(usable, surplus, -np.inf)
    best = np.argmax(surplus_filled, axis=1)
    rows = np.arange(len(value_points))

    chosen_surplus = surplus_filled[rows, best]
    claim = np.isfinite(chosen_surplus) & (chosen_surplus > 0)
    return pd.DataFrame(
        {
            "bid": np.where(claim, bids[rows, best], 0.0),
            "win_prob": np.where(claim, wins[rows, best], 0.0),
            "value_points": value_points,
            "value_budget": value_budget,
            "expected_surplus": np.where(np.isfinite(chosen_surplus), chosen_surplus, np.nan),
            "confidence": np.where(claim, grid[best], np.nan),
            "claim": claim,
        }
    )


def rank_claims(
    frame: pd.DataFrame,
    *,
    price_schedule: BidSchedule,
    value_points: np.ndarray,
    rate: ExchangeRate,
    identifiers: tuple[str, ...] = ("league_id", "season", "week", "player_id", "position"),
) -> pd.DataFrame:
    """A ranked claim list: best expected surplus first, with the bid to place.

    Ranked by expected surplus rather than by value or by price. The most valuable
    player is often the wrong claim, because everyone else wants him too and the
    price reflects it; the right claim is the one whose price is furthest below what
    he is worth.
    """
    decided = optimal_bids(price_schedule, value_points, rate.points_per_budget(frame))
    carried = [c for c in identifiers if c in frame.columns]
    out = pd.concat([frame[carried].reset_index(drop=True), decided], axis=1)

    # One row per player. The claim table holds one row per *bid*, so a contested
    # player appears once per rival who wanted him -- which is information about the
    # price, not five separate decisions to make.
    if "player_id" in out.columns:
        keys = [c for c in ("league_id", "season", "week", "player_id") if c in out.columns]
        out = out.sort_values("expected_surplus", ascending=False).drop_duplicates(keys)

    # Cumulative expected spend over the ranked list, so a reader can see where the
    # budget goes. Expected, not committed: a bid is only paid when it wins, so the
    # bids down a list sum to more than the budget it actually costs to follow it.
    out = out.sort_values("expected_surplus", ascending=False, na_position="last")
    out = out.reset_index(drop=True)
    out["expected_spend"] = (out["bid"] * out["win_prob"]).where(out["claim"], 0.0)
    out["cumulative_expected_spend"] = out["expected_spend"].cumsum()
    return out
