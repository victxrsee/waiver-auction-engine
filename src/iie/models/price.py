"""The price model: what it costs to win a waiver claim.

FAAB waivers are a first-price sealed-bid auction, so the quantity that matters
is not what a player is worth but what it takes to outbid everyone else. The
engine should bid to beat the competition, not to pay full value.

The target is therefore the **highest losing bid** in an auction that had a
winner -- the amount that actually had to be exceeded. Winning bids are the wrong
target: in an uncontested claim the winner may have bid 40 when 0 would have
done, so their bid measures their enthusiasm rather than the price.

Measured on the corpus, that target is 78% zeros and sharply skewed above them:
a median of 0, a 90th percentile of 0.06 of budget and a 99th of 0.30. A single
regressor handles that badly, so this is a hurdle model -- one classifier for
whether anyone else bids at all, one regressor for how much given that they do.
The split is not just statistical convenience; it is the shape of the decision a
manager makes.

Two known imperfections in the target, neither fixable from this data alone.
About a quarter of contested auctions are exact ties broken by waiver priority
rather than price, and in about a tenth the winner bid *less* than a loser --
almost certainly bids that could not be honoured because the bidder lacked the
remaining budget. Both mean the observed price is a slightly noisy version of the
price that was really required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

log = logging.getLogger(__name__)

AUCTION_KEYS = ("league_id", "season", "week", "player_id")

# Everything here is knowable before the claim is made. Deliberately excluded:
# the number of bidders, which is not observable at decision time and is close to
# the target itself.
FEATURE_COLUMNS = (
    "week",
    "asof_week",
    "points_l1",
    "points_l3",
    "snap_pct_l1",
    "snap_pct_l3",
    "snap_pct_delta",
    "targets_l1",
    "targets_l3",
    "carries_l3",
    "receptions_l3",
    "played_l3",
    "total_rosters",
    "reception_points",
    "waiver_budget",
    "is_defense",
    # Opportunity shocks. depth_promotion and demand_share are here because they
    # move price hard while barely moving realized value -- a just-promoted starter
    # costs 66% more and returns 26% less -- so a price model that cannot see them
    # is blind to most of what the market is actually reacting to.
    "depth_rank",
    "depth_size",
    "depth_promotion",
    "depth_is_starter",
    "demand_share",
    "injury_severity",
)

# The Phase 1 additions, named so they can be ablated as a family. They are what
# distinguishes "the market reacts to news" from "the market reacts to box scores".
OPPORTUNITY_COLUMNS = (
    "depth_rank",
    "depth_size",
    "depth_promotion",
    "depth_is_starter",
    "demand_share",
    "injury_severity",
)

CATEGORICAL_COLUMNS = ("position",)


def build_auction_table(claims: pd.DataFrame) -> pd.DataFrame:
    """Collapse bids into auctions, with the price that had to be beaten.

    Auctions where nothing was won are dropped: a lone failed claim usually means
    a roster or budget problem rather than a lost contest, and it says nothing
    about what winning would have cost.
    """
    frame = claims.copy()
    frame["losing_bid"] = frame["bid_pct"].where(frame["won"] == 0)

    aggregations = {
        "n_bidders": ("bid_pct", "size"),
        "n_winners": ("won", "sum"),
        "price_to_beat": ("losing_bid", "max"),
        "winning_bid": ("bid_pct", "max"),
    }
    carried = [c for c in (*FEATURE_COLUMNS, *CATEGORICAL_COLUMNS, "franchise_id") if c in frame]
    for column in carried:
        aggregations[column] = (column, "first")

    auctions = frame.groupby(list(AUCTION_KEYS), as_index=False).agg(**aggregations)
    contested = auctions["n_winners"] >= 1
    log.info(
        "auctions: %d with a winner, %d dropped as unwon",
        contested.sum(),
        (~contested).sum(),
    )

    auctions = auctions[contested].copy()
    # No losing bid means nobody else wanted the player: the price to beat is zero.
    auctions["price_to_beat"] = auctions["price_to_beat"].fillna(0.0)
    auctions["contested"] = (auctions["price_to_beat"] > 0).astype(int)
    return auctions.reset_index(drop=True)


def _design_matrix(
    frame: pd.DataFrame,
    vocabulary: dict[str, list[str]] | None = None,
    features: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Feature matrix. Gradient boosting takes the NaNs natively, which matters
    because defenses legitimately have no player features at all.

    ``vocabulary`` pins the categorical levels recorded at fit time. Without it,
    ``.astype("category")`` derives codes from whatever frame it is handed, so a
    split missing a rare position renumbers every position above it and the model
    is scored on integers it never saw. A single-row frame -- the actual inference
    call -- is the degenerate case: every position becomes code 0.

    ``features`` narrows the column set, which exists so a feature family can be
    ablated through the same code path that trains the real model. Measuring what
    a feature contributes by editing the module constant measures a different
    program than the one that ships.
    """
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise KeyError(
            f"declared features absent from the frame: {missing}. Add them upstream or "
            f"remove them from FEATURE_COLUMNS -- silently training on fewer features "
            f"than declared is how a feature goes missing unnoticed."
        )

    matrix = frame[list(features)].apply(pd.to_numeric, errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            continue
        levels = None if vocabulary is None else vocabulary.get(column)
        if levels is None:
            matrix[column] = frame[column].astype("category").cat.codes.replace(-1, np.nan)
            continue
        # Explicit lookup rather than a CategoricalDtype cast: an unseen level maps
        # to NaN, which the boosted trees handle natively, instead of shifting the
        # codes of every level around it. Using "string" keeps nulls as nulls.
        lookup = {level: code for code, level in enumerate(levels)}
        matrix[column] = frame[column].astype("string").map(lookup).astype("Float64")
    return matrix


def _fit_vocabulary(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Categorical levels observed at fit time, to be reused at predict time."""
    return {
        column: sorted(frame[column].dropna().astype(str).unique())
        for column in CATEGORICAL_COLUMNS
        if column in frame.columns
    }


class ComparablesPrice:
    """Baseline: what leagues have historically paid in similar situations.

    Deliberately simple and interpretable -- a positional and week-bucket summary
    that any learned model has to beat to justify itself.

    ``statistic`` defaults to the mean rather than the median on purpose. Four
    auctions in five are uncontested, so a median baseline predicts zero
    everywhere and is indistinguishable from never bidding. That scores well on
    absolute error and is useless as a bid.
    """

    def __init__(self, week_bucket: int = 4, statistic: str = "mean") -> None:
        self.week_bucket = week_bucket
        self.statistic = statistic
        self._table: pd.Series | None = None
        self._fallback: float = 0.0

    def _key(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": frame.get("position", pd.Series(index=frame.index, dtype=object)),
                "week_group": (frame["week"] // self.week_bucket).astype(int),
            }
        )

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> ComparablesPrice:
        key = self._key(frame)
        grouped = target.groupby([key["position"], key["week_group"]])
        self._table = grouped.median() if self.statistic == "median" else grouped.mean()
        self._fallback = float(target.median() if self.statistic == "median" else target.mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        key = self._key(frame)
        index = pd.MultiIndex.from_arrays([key["position"], key["week_group"]])
        return self._table.reindex(index).fillna(self._fallback).to_numpy()


@dataclass(frozen=True)
class BidSchedule:
    """Per-row price quantile curves, evaluated at any confidence for free.

    The curves do not depend on the confidence asked for, so sweeping a range of
    operating points -- which is the only fair way to compare bidding policies --
    should cost one set of model predictions, not one per point.
    """

    probability: np.ndarray
    grid: np.ndarray
    curves: np.ndarray

    def at(self, confidence: float) -> np.ndarray:
        """Bid amounts that win about ``confidence`` of auctions."""
        with np.errstate(divide="ignore", invalid="ignore"):
            needed = np.where(self.probability > 0, 1 - (1 - confidence) / self.probability, -1.0)

        wanted = np.clip(needed, self.grid[0], self.grid[-1])
        upper = np.clip(np.searchsorted(self.grid, wanted), 1, len(self.grid) - 1)
        low, high = self.grid[upper - 1], self.grid[upper]
        weight = np.where(high > low, (wanted - low) / (high - low), 0.0)

        rows = np.arange(len(self.probability))
        out = self.curves[rows, upper - 1] * (1 - weight) + self.curves[rows, upper] * weight
        # A non-positive requirement means the odds of facing nobody already cover
        # the target confidence, so a minimal bid suffices.
        return np.where(needed > 0, out, 0.0)


@dataclass
class HurdlePrice:
    """Classifier for whether anyone else bids, regressor for how much.

    Predicted price is ``P(contested) * E[price | contested]``, which is the
    expected amount needed. A manager who wants to win with some confidence
    should look at the upper tail of that instead, which is what ``predict_quantile``
    is for.
    """

    seed: int = 0
    # Conditional price quantiles are fitted on a grid and interpolated between,
    # so no distributional shape has to be assumed for the tail.
    quantile_levels: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
    # Narrowing this is how a feature family is ablated; the fitted model carries
    # its own column list so predict cannot silently use a different one.
    features: tuple[str, ...] = FEATURE_COLUMNS
    classifier: HistGradientBoostingClassifier | None = field(default=None, repr=False)
    regressor: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    _quantile_models: list = field(default_factory=list, repr=False)
    # Categorical levels seen at fit time, replayed at predict time so codes mean
    # the same thing in both. Without this the model is scored on integers it was
    # never fitted on.
    _vocabulary: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> HurdlePrice:
        self._vocabulary = _fit_vocabulary(frame)
        matrix = _design_matrix(frame, self._vocabulary, self.features)
        contested = (target > 0).astype(int)

        self.classifier = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, random_state=self.seed
        ).fit(matrix, contested)

        # Fitted only where a contest happened; the zeros are the classifier's job.
        hit = contested.to_numpy() == 1
        self.regressor = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, random_state=self.seed
        ).fit(matrix[hit], target[hit])

        self._quantile_models = [
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=level,
                max_iter=200,
                learning_rate=0.06,
                random_state=self.seed,
            ).fit(matrix[hit], target[hit])
            for level in self.quantile_levels
        ]
        return self

    def predict_parts(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = _design_matrix(frame, self._vocabulary, self.features)
        probability = self.classifier.predict_proba(matrix)[:, 1]
        amount = np.clip(self.regressor.predict(matrix), 0.0, None)
        return probability, amount

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability, amount = self.predict_parts(frame)
        return probability * amount

    def predict_to_win(self, frame: pd.DataFrame, confidence: float = 0.8) -> np.ndarray:
        """What to bid to win with roughly ``confidence`` probability.

        The expected price is the wrong number to bid -- it loses most contested
        auctions it enters. Winning at a stated rate means bidding a quantile of
        the price distribution, and that distribution is a mixture: zero when
        nobody else bids, and the conditional price when somebody does.

        For a target ``q``, the quantile needed of the *conditional* distribution
        is ``1 - (1 - q) / p`` where ``p`` is the chance of a contest. When that
        is negative, ``q`` is already covered by the odds of facing nobody and a
        minimal bid suffices. The conditional quantiles come from real quantile
        regressions fitted on contested auctions, interpolated across the grid,
        rather than from an assumed distributional shape.
        """
        return self.bid_schedule(frame).at(confidence)

    def bid_schedule(self, frame: pd.DataFrame) -> BidSchedule:
        """Run the models once, so a confidence sweep costs array arithmetic."""
        matrix = _design_matrix(frame, self._vocabulary, self.features)
        return BidSchedule(
            probability=self.classifier.predict_proba(matrix)[:, 1],
            grid=np.asarray(self.quantile_levels),
            curves=np.column_stack(
                [np.clip(model.predict(matrix), 0.0, None) for model in self._quantile_models]
            ),
        )


def simulate_policy(
    auctions: pd.DataFrame, bids: np.ndarray, *, price_col: str = "price_to_beat"
) -> dict[str, float]:
    """Score a bidding policy the way a manager experiences it.

    A first-price auction is won by exceeding the best competing bid and paid at
    the amount bid, so the two numbers that matter are how often the policy wins
    and what each win costs. Absolute error against the price does not capture
    either: a bid one cent short loses everything, and a bid far above wins but
    overpays.

    An uncontested auction is won by *any* bid, zero included -- there is no rival
    to beat, and Sleeper awards it on waiver priority. Scoring those with a strict
    ``>`` silently discards every claim a policy wins for free, which is most of
    them: four auctions in five are uncontested, and a model that correctly bids
    nothing on them would be recorded as losing all of them.
    """
    bids = np.asarray(bids, dtype=float)
    price = auctions[price_col].to_numpy(dtype=float)
    contested = price > 0

    # Contested auctions need a strictly higher bid: an equal bid is a tie, broken
    # on waiver priority rather than price, and counting ties as losses keeps the
    # estimate conservative. Uncontested auctions have no rival to outbid.
    won = np.where(contested, bids > price, True)
    spend = np.where(won, bids, 0.0)
    wins = int(won.sum())

    # Pooling uncontested auctions makes every policy look good, so the contested
    # subset is reported separately: that is where a price model earns its keep.
    contested_wins = int((won & contested).sum())

    return {
        "win_rate": float(won.mean()),
        "contested_win_rate": float(won[contested].mean()) if contested.any() else float("nan"),
        "mean_bid": float(bids.mean()),
        "spend_per_auction": float(spend.mean()),
        "spend_per_win": float(spend.sum() / wins) if wins else float("nan"),
        "overpay_per_win": float((spend[won] - price[won]).sum() / wins) if wins else float("nan"),
        "contested_spend_per_win": (
            float(spend[contested].sum() / contested_wins) if contested_wins else float("nan")
        ),
    }


DEFAULT_CONFIDENCES = np.round(np.arange(0.50, 0.99, 0.005), 4)


def policy_frontier(
    auctions: pd.DataFrame, schedule: BidSchedule, confidences: np.ndarray | None = None
) -> pd.DataFrame:
    """Win rate and spend at every operating point, which is the real comparison.

    A single matched point is not enough, because this curve is not smooth. The
    needed conditional quantile is ``1 - (1-q)/p``, and as ``q`` rises it crosses
    the coarse grid the quantile regressions were fitted on, so the interpolated
    bid jumps. Measured on the corpus: between two adjacent confidences the mean bid
    moved 36% and the reported spend reduction moved from 42.9% to 22.2%, and one
    step further the contested win rate jumped eight points at once. Two runs over
    near-identical data landed on opposite sides of a step and reported 17.3% and
    46.9% for the same policy.
    """
    grid = DEFAULT_CONFIDENCES if confidences is None else np.asarray(confidences)
    rows = [{"confidence": float(c), **simulate_policy(auctions, schedule.at(c))} for c in grid]
    return pd.DataFrame(rows).sort_values("contested_win_rate").reset_index(drop=True)


def spend_at_win_rate(
    frontier: pd.DataFrame,
    target: float,
    *,
    win_col: str = "contested_win_rate",
    spend_col: str = "mean_bid",
) -> dict[str, float]:
    """Spend interpolated onto an exact win rate, with the local step size.

    Snapping to the nearest grid confidence lands two arms at different win rates
    and then differences them, which charges the gap to the model. Interpolating
    onto a common win rate is the minimum fix; ``step`` reports how far apart the
    bracketing points were, because a wide bracket means the number is fragile
    however it was obtained.
    """
    usable = frontier.dropna(subset=[win_col, spend_col]).sort_values(win_col)
    wins = usable[win_col].to_numpy(dtype=float)
    spend = usable[spend_col].to_numpy(dtype=float)
    if len(wins) == 0:
        return {"spend": float("nan"), "win_rate": float(target), "step": float("nan")}

    value = float(np.interp(target, wins, spend))
    index = int(np.searchsorted(wins, target))
    low = max(index - 1, 0)
    high = min(index, len(wins) - 1)
    # How much the spend varies across the bracket the answer was drawn from.
    step = abs(spend[high] - spend[low])
    inside = wins[0] <= target <= wins[-1]
    return {
        "spend": value,
        "win_rate": float(target),
        "step": step,
        "extrapolated": float(not inside),
    }


def evaluate(predictions: np.ndarray, truth: pd.Series) -> dict[str, float]:
    """Error and ranking quality.

    Both are reported because they answer different questions: absolute error is
    what a bid recommendation lives or dies by, while rank correlation says
    whether the model knows which players are expensive at all.
    """
    predictions = np.asarray(predictions, dtype=float)
    values = truth.to_numpy(dtype=float)
    return {
        "mae": float(np.mean(np.abs(predictions - values))),
        "rmse": float(np.sqrt(np.mean((predictions - values) ** 2))),
        "spearman": float(pd.Series(predictions).corr(pd.Series(values), method="spearman")),
    }
