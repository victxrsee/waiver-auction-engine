"""The value model: what a waiver claim is worth, in points.

The price model answers "what will this cost". This answers "is it worth that".
Without both there is no decision -- the engine can price a claim and cannot rank
one -- so this is the half that turns a price calculator into an engine.

Three choices here are deliberate and each has a reason the obvious alternative
does not.

**Trained on the player panel, not on the claim table.** Outcomes exist for every
player-week, whether or not anyone bid. Training on claims would condition on
someone having wanted the player, which is the selection the engine exists to
second-guess: the backup nobody claimed and should have is absent from that sample
by construction. The panel also has far more signal per row -- 107,226 player-weeks
against a claim table whose distinct players are a small fraction of that.

**Predicts scoring components, not league points.** A reception is worth 0 points
in one league and 1 in another, so a model fitted on league points learns one
league's rules. Instead this predicts standard points and receptions separately,
and :meth:`ValueModel.predict_points` combines them with whatever
``reception_points`` a league uses -- exactly as ``claims._apply_league_scoring``
does for the realised outcome. One fit serves every league.

**Conditions on the horizon rather than hiding it.** ``add_forward`` sums the six
weeks after each row, so late-season rows cover fewer weeks: average future games
fall from 3.14 at week 12 to 1.05 at week 16 and 0 at week 18, and the mean target
falls with them. The temptation is to drop those weeks as contaminated. That would
be wrong, because the truncation is not an artifact of the measurement -- it is the
quantity being measured. A player claimed in week 16 really does deliver only two
more weeks, and an engine that cannot see that will overpay for him every time.
``weeks_remaining`` is deterministic and known when the bid is placed, so it is a
legitimate feature; what would be illegitimate is training across the boundary
*without* it, which lets the model read "late in the season" as "bad player".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hurdle import DEFAULT_QUANTILES, Hurdle

log = logging.getLogger(__name__)

# The last week add_forward can see. Horizon truncation is measured against it.
FINAL_WEEK = 18

# Everything knowable at the moment a claim is placed. No forward-looking column
# and nothing derived from the claim itself -- the panel does not know about bids.
FEATURE_COLUMNS = (
    "week",
    "weeks_remaining",
    "fantasy_points_l1",
    "fantasy_points_l3",
    "receptions_l1",
    "receptions_l3",
    "targets_l1",
    "targets_l3",
    "carries_l3",
    "snap_pct_l1",
    "snap_pct_l3",
    "snap_pct_delta",
    "played_l3",
)

CATEGORICAL_COLUMNS = ("position",)

# The two components every league's scoring is built from. Kept separate so one fit
# serves all of them; see the module docstring.
COMPONENTS = ("fantasy_points", "receptions")


def add_horizon(
    panel: pd.DataFrame, *, final_week: int = FINAL_WEEK, horizon: int = 6
) -> pd.DataFrame:
    """Attach ``weeks_remaining``: how many weeks the forward window actually covers.

    ``min(final_week - week, horizon)`` and not ``final_week - week``. The latter is
    a strictly decreasing function of ``week``, which is already a feature, so a
    tree ensemble gains nothing from it: the two were measured at correlation
    -1.000 and an arm given both scored bit-identically to an arm given only one.
    It was also simply wrong as described, reporting 17 weeks of window at week 1
    where the window is 6 -- false on 61% of rows.

    Clipping at ``horizon`` is what makes it informative: the value is flat across
    weeks 1-12, where the window is full, and falls only over the last few weeks,
    where it genuinely truncates. That shape is *not* a monotone transform of
    ``week``, so it tells the model something ``week`` cannot.

    Deterministic from the calendar either way, so it is known when a bid is placed
    and is not lookahead.
    """
    out = panel.copy()
    out["weeks_remaining"] = (final_week - out["week"]).clip(lower=0, upper=horizon)
    return out


@dataclass
class ValueModel:
    """Expected scoring components over the remaining horizon.

    One :class:`~iie.models.hurdle.Hurdle` per component. The target is ~80% zeros
    -- most available players never produce -- which is the same shape as the price
    target and the reason the hurdle structure is shared rather than rewritten.
    """

    horizon: int = 6
    seed: int = 0
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    features: tuple[str, ...] = FEATURE_COLUMNS
    _hurdles: dict[str, Hurdle] = field(default_factory=dict, repr=False)

    def target_column(self, component: str) -> str:
        return f"{component}_next{self.horizon}"

    def fit(self, panel: pd.DataFrame) -> ValueModel:
        """Fit one hurdle per scoring component on the player panel."""
        self._hurdles = {}
        for component in COMPONENTS:
            column = self.target_column(component)
            if column not in panel.columns:
                raise KeyError(
                    f"{column} absent from the panel; call panel.add_forward("
                    f"horizon={self.horizon}) before fitting"
                )
            target = panel[column]
            log.info(
                "value: fitting %s on %d rows (%.1f%% zero)",
                component,
                len(panel),
                100.0 * float((target <= 0).mean()),
            )
            self._hurdles[component] = Hurdle(
                features=self.features,
                categorical=CATEGORICAL_COLUMNS,
                seed=self.seed,
                quantile_levels=self.quantile_levels,
            ).fit(panel, target)
        return self

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Expected value of each scoring component, one column per component."""
        if not self._hurdles:
            raise RuntimeError("ValueModel.predict_components called before fit")
        return pd.DataFrame(
            {c: h.predict(frame) for c, h in self._hurdles.items()}, index=frame.index
        )

    def predict_points(
        self, frame: pd.DataFrame, reception_points: pd.Series | float = 0.0
    ) -> np.ndarray:
        """Expected points in a league's own scoring.

        Mirrors ``claims._apply_league_scoring`` so the prediction and the realised
        outcome are expressed in the same units. ``reception_points`` may be a
        scalar or a per-row series, because a ranked claim list is built for one
        league at a time but a backtest spans many.
        """
        parts = self.predict_components(frame)
        rec = (
            reception_points
            if np.isscalar(reception_points)
            else pd.Series(reception_points, index=frame.index).to_numpy()
        )
        return parts["fantasy_points"].to_numpy() + rec * parts["receptions"].to_numpy()

    def predict_quantile(
        self, frame: pd.DataFrame, quantile: float, component: str = "fantasy_points"
    ) -> np.ndarray:
        """A quantile of one component's outcome, for the decision layer's downside.

        Same mixture as the price model's bid schedule, read the other way round: a
        fraction ``1 - p`` of the distribution sits at exactly zero, so the quantile
        wanted of the *conditional* distribution is ``1 - (1 - q) / p``. When that
        is non-positive the quantile falls inside the spike at zero, and the honest
        answer is zero -- that share of outcomes really is nothing at all.
        """
        probability, grid, curves = self._hurdles[component].quantile_curves(frame)
        with np.errstate(divide="ignore", invalid="ignore"):
            needed = np.where(probability > 0, 1 - (1 - quantile) / probability, -1.0)

        wanted = np.clip(needed, grid[0], grid[-1])
        upper = np.clip(np.searchsorted(grid, wanted), 1, len(grid) - 1)
        low, high = grid[upper - 1], grid[upper]
        weight = np.where(high > low, (wanted - low) / (high - low), 0.0)
        rows = np.arange(len(probability))
        interpolated = curves[rows, upper - 1] * (1 - weight) + curves[rows, upper] * weight
        return np.where(needed > 0, interpolated, 0.0)


class PositionalValue:
    """Baseline: mean outcome by position and week bucket.

    The same shape as ``ComparablesPrice``, and for the same reason -- a learned
    model has to beat something interpretable to justify itself. The mean rather
    than the median, because the target is ~80% zeros and a median baseline
    predicts zero everywhere.
    """

    def __init__(self, week_bucket: int = 4) -> None:
        self.week_bucket = week_bucket
        self._tables: dict[str, pd.Series] = {}
        self._fallbacks: dict[str, float] = {}
        self.horizon = 6

    def _key(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "position": frame.get("position", pd.Series(index=frame.index, dtype=object)),
                "week_group": (frame["week"] // self.week_bucket).astype(int),
            }
        )

    def fit(self, panel: pd.DataFrame) -> PositionalValue:
        key = self._key(panel)
        for component in COMPONENTS:
            target = panel[f"{component}_next{self.horizon}"]
            self._tables[component] = target.groupby(
                [key["position"], key["week_group"]], observed=True
            ).mean()
            self._fallbacks[component] = float(target.mean())
        return self

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        key = self._key(frame)
        index = pd.MultiIndex.from_arrays([key["position"], key["week_group"]])
        return pd.DataFrame(
            {
                component: table.reindex(index).fillna(self._fallbacks[component]).to_numpy()
                for component, table in self._tables.items()
            },
            index=frame.index,
        )

    def predict_points(
        self, frame: pd.DataFrame, reception_points: pd.Series | float = 0.0
    ) -> np.ndarray:
        parts = self.predict_components(frame)
        rec = (
            reception_points
            if np.isscalar(reception_points)
            else pd.Series(reception_points, index=frame.index).to_numpy()
        )
        return parts["fantasy_points"].to_numpy() + rec * parts["receptions"].to_numpy()


class TrailingProduction:
    """Baseline: recent production per week, scaled to the remaining window.

    **This, not the positional mean, is the bar a value model has to clear.** It
    fits nothing, reads two columns the model already has, and on the 2025 test
    season it ranks better *within every week* than the first version of
    ``ValueModel`` did -- 0.780 against 0.637 mean within-week Spearman, winning in
    all 17 scoreable weeks. A learned model that cannot beat it is not earning its
    complexity, however good its absolute error looks.

    Absolute error is the trap here, and CLAUDE.md invariant 6 names it: the target
    is ~80% zeros, so predicting zero everywhere scores MAE 5.19 while the
    positional mean scores 6.37. A baseline can lose to *nothing* on MAE and the
    comparison still look favourable.
    """

    def __init__(self, window: int = 3) -> None:
        self.window = window

    def fit(self, panel: pd.DataFrame) -> TrailingProduction:
        return self  # Nothing to fit; the signature exists so arms are swappable.

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        weeks = frame["weeks_remaining"].clip(lower=0).to_numpy()
        return pd.DataFrame(
            {
                component: frame[f"{component}_l{self.window}"].to_numpy() / self.window * weeks
                for component in COMPONENTS
            },
            index=frame.index,
        )

    def predict_points(
        self, frame: pd.DataFrame, reception_points: pd.Series | float = 0.0
    ) -> np.ndarray:
        parts = self.predict_components(frame)
        rec = (
            reception_points
            if np.isscalar(reception_points)
            else pd.Series(reception_points, index=frame.index).to_numpy()
        )
        return parts["fantasy_points"].to_numpy() + rec * parts["receptions"].to_numpy()


def evaluate_value(
    predictions: np.ndarray, truth: pd.Series, *, groups: pd.Series | None = None
) -> dict[str, float]:
    """Error and ranking quality for a value prediction.

    Spearman is the one that matters for a waiver engine. The decision is which
    claim to make, so getting the *order* right is worth more than getting the
    level right -- and a model can be well calibrated on average while ranking
    badly within a week, which is precisely when it would mislead.
    """
    predicted = np.asarray(predictions, dtype=float)
    actual = truth.to_numpy(dtype=float)
    out = {
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "spearman": float(pd.Series(predicted).corr(pd.Series(actual), method="spearman")),
    }
    if groups is not None:
        # Ranking within the set a manager actually chooses between, which is the
        # real task: a pooled correlation is flattered by easy cross-week contrasts.
        frame = pd.DataFrame({"p": predicted, "a": actual, "g": groups.to_numpy()})

        def rank_within(d: pd.DataFrame) -> float:
            # A constant column has no defined correlation, and the final week is
            # exactly that: no future weeks remain, so every outcome is zero. Such
            # a group carries no ranking information and is skipped rather than
            # counted as a zero correlation.
            if len(d) < 3 or d["a"].nunique() < 2 or d["p"].nunique() < 2:
                return np.nan
            return d["p"].corr(d["a"], method="spearman")

        within = frame.groupby("g").apply(rank_within, include_groups=False)
        out["spearman_within_group"] = float(within.mean())
        out["groups_ranked"] = float(within.notna().sum())
    return out
