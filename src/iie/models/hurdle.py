"""The hurdle machinery both models are built on.

Price and value have the same awkward shape: a target that is mostly zero with a
sharply skewed tail above it. The price to beat is 78% zeros because most waiver
claims are uncontested; the points a claim goes on to return are 80% zeros because
most available players never produce. A single regressor fits neither, because it
spends its capacity straddling a spike at zero and a long tail.

So both are hurdle models: one classifier for whether the outcome happens at all,
one regressor for how large it is given that it does, and a grid of quantile
regressions for the spread. The split is not just statistical convenience -- it is
the shape of the question. "Will anyone bid against me, and how much if they do"
and "will this player produce at all, and how much if he does" are two questions
each, not one.

What is *not* shared lives in the callers: the price model's bid schedule, which
inverts a mixture quantile to answer "what must I bid to win q of the time", and
the value model's component decomposition, which predicts scoring components so
one fit serves every league's scoring rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# Fitted on a grid and interpolated between, so no distributional shape has to be
# assumed for the tail.
DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def fit_vocabulary(frame: pd.DataFrame, categorical: tuple[str, ...]) -> dict[str, list[str]]:
    """Categorical levels observed at fit time, to be replayed at predict time."""
    return {
        column: sorted(frame[column].dropna().astype(str).unique())
        for column in categorical
        if column in frame.columns
    }


def design_matrix(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    vocabulary: dict[str, list[str]] | None = None,
    categorical: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Feature matrix. Gradient boosting takes NaNs natively, which matters because
    some rows legitimately have no player features at all (team defenses in the
    price model; players with no stat line in the value panel).

    Two things here are load-bearing, both from defects that shipped:

    * A declared-but-absent feature **raises**. Skipping it silently is how
      ``total_rosters`` came to be declared in the feature list and never joined,
      so production trained on fifteen features while every test used sixteen, with
      no error on either side.
    * ``vocabulary`` pins the categorical levels recorded at fit time. Without it,
      ``.astype("category")`` derives codes from whatever frame it is handed, so a
      split missing a rare level renumbers every level above it: position ``WR`` was
      code 17 at fit and 14 at predict. A single-row frame -- the actual inference
      call -- is the degenerate case, where every level becomes code 0.
    """
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise KeyError(
            f"declared features absent from the frame: {missing}. Add them upstream or "
            f"remove them from the model's feature list -- silently training on fewer "
            f"features than declared is how a feature goes missing unnoticed."
        )

    matrix = frame[list(features)].apply(pd.to_numeric, errors="coerce")

    for column in categorical:
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


@dataclass
class Hurdle:
    """One classifier for whether the outcome is nonzero, one regressor for its size.

    The expectation is ``P(nonzero) * E[size | nonzero]``. Callers wanting the tail
    rather than the mean use :meth:`quantile_curves`, which returns the conditional
    quantiles alongside the occurrence probability so the two can be combined
    however the question requires.
    """

    features: tuple[str, ...]
    categorical: tuple[str, ...] = ()
    seed: int = 0
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES
    max_iter: int = 200
    learning_rate: float = 0.06
    classifier: HistGradientBoostingClassifier | None = field(default=None, repr=False)
    regressor: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    _quantile_models: list = field(default_factory=list, repr=False)
    _vocabulary: dict[str, list[str]] = field(default_factory=dict, repr=False)

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return design_matrix(frame, self.features, self._vocabulary, self.categorical)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> Hurdle:
        self._vocabulary = fit_vocabulary(frame, self.categorical)
        matrix = self._matrix(frame)
        values = pd.Series(np.asarray(target, dtype=float), index=matrix.index)
        nonzero = (values > 0).astype(int)

        common = {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "random_state": self.seed,
        }
        self.classifier = HistGradientBoostingClassifier(**common).fit(matrix, nonzero)

        # Fitted only where the outcome happened; the zeros are the classifier's job.
        hit = nonzero.to_numpy() == 1
        self.regressor = HistGradientBoostingRegressor(**common).fit(matrix[hit], values[hit])
        self._quantile_models = [
            HistGradientBoostingRegressor(loss="quantile", quantile=level, **common).fit(
                matrix[hit], values[hit]
            )
            for level in self.quantile_levels
        ]
        return self

    def predict_parts(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(probability of a nonzero outcome, expected size given nonzero)``."""
        matrix = self._matrix(frame)
        probability = self.classifier.predict_proba(matrix)[:, 1]
        size = np.clip(self.regressor.predict(matrix), 0.0, None)
        return probability, size

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability, size = self.predict_parts(frame)
        return probability * size

    def quantile_curves(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(probability, quantile grid, per-row conditional quantiles)``.

        Run once, so sweeping operating points costs array arithmetic rather than a
        fresh set of model predictions per point.
        """
        matrix = self._matrix(frame)
        return (
            self.classifier.predict_proba(matrix)[:, 1],
            np.asarray(self.quantile_levels),
            np.column_stack(
                [np.clip(model.predict(matrix), 0.0, None) for model in self._quantile_models]
            ),
        )
