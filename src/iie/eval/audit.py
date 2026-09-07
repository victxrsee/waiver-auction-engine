"""Distribution checks on the claim table, aimed at one specific failure mode.

Every serious defect this project has produced was silent: the code ran, the model
fitted, the number looked plausible. `injury_severity` was uniformly 0.0 across an
entire training season. `demand_prior` averaged 8.6 in train and 278 in test.
Position `WR` was encoded 17 at fit and 14 at predict. None raised an error, and
reading the code did not reveal any of them.

What does reveal them is looking at the data *after* the code has run, split the
way the model will split it. All three of the above are visible in a single
``groupby(split).mean()``. This module is that check, made systematic and cheap
enough to run after every rebuild.

The checks are deliberately dumb and deterministic. Judgement about *why* a feature
shifted is a human's job; noticing that it did is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A feature whose mean moves by more than this between splits is reported. Set from
# the demand_prior defect, which moved 32x; the depth-chart schema change moved
# depth_is_starter by 1.7x and was also worth catching, so the bar is low.
SCALE_RATIO = 1.5
# Null-rate divergence in percentage points. The injuries schema drift took one
# split from 7% null to 100%.
NULL_SHIFT_PP = 10.0


@dataclass(frozen=True)
class Finding:
    feature: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code:9s} {self.feature:20s} {self.detail}"


def _describe(frame: pd.DataFrame, features: list[str], split_col: str) -> pd.DataFrame:
    """Mean and null rate per feature per split, as a tidy frame."""
    rows = []
    for split, part in frame.groupby(split_col, observed=True):
        for feature in features:
            values = pd.to_numeric(part[feature], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "split": str(split),
                    "n": len(values),
                    "null_pct": 100.0 * float(values.isna().mean()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "std": float(values.std()) if values.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _scale_finding(feature: str, means: pd.Series) -> Finding | None:
    """Flag a feature whose level differs between splits.

    Zero is handled separately from small: a feature that is exactly zero in one
    split and nonzero in another is not a scale shift, it is a join that failed,
    and it should be reported as such.
    """
    usable = means.dropna()
    if len(usable) < 2:
        return None

    lo, hi = float(usable.min()), float(usable.max())
    if np.isclose(lo, 0.0) and not np.isclose(hi, 0.0):
        where = usable.idxmin()
        return Finding(feature, "DEAD", f"exactly 0.0 in split {where!r}, {hi:.4g} elsewhere")

    if np.isclose(lo, 0.0):
        return None
    ratio = abs(hi / lo) if lo != 0 else float("inf")
    if ratio > SCALE_RATIO:
        pairs = ", ".join(f"{k}={v:.4g}" for k, v in usable.items())
        return Finding(feature, "SCALE", f"{ratio:.1f}x across splits ({pairs})")
    return None


def _audit_categorical(claims: pd.DataFrame, feature: str, split_col: str) -> list[Finding]:
    """Categoricals fail differently: by vocabulary, not by level.

    A mean is meaningless here, and coercing one to a number reports the column as
    100% null. What actually breaks a categorical is a level appearing at test time
    that was absent at fit time -- which is how `position` came to be encoded 17 in
    training and 14 in scoring, so the model read every skill position under a label
    it had never been fitted on.
    """
    splits = sorted(str(s) for s in claims[split_col].dropna().unique())
    if len(splits) < 2:
        return []

    levels = {
        split: set(claims.loc[claims[split_col].astype(str) == split, feature].dropna().astype(str))
        for split in splits
    }
    # Splits sort chronologically for both boundaries used here (2024 before 2025,
    # weeks<=9 before weeks>9), so the first is the one a model would train on.
    reference, *rest = splits
    unseen = sorted(set().union(*(levels[s] for s in rest)) - levels[reference])
    if unseen:
        shown = ", ".join(unseen[:6]) + (" ..." if len(unseen) > 6 else "")
        return [
            Finding(
                feature,
                "UNSEEN",
                f"{len(unseen)} level(s) absent from {reference!r} appear later: {shown}",
            )
        ]
    return []


def audit_features(
    claims: pd.DataFrame,
    features: tuple[str, ...],
    *,
    split_col: str = "season",
    categorical: tuple[str, ...] = (),
    ignore: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[Finding]]:
    """Per-split summary of every declared feature, plus anything anomalous.

    ``split_col`` should be whatever the model will train and test across, because
    a distribution shift only matters where the model has to generalise over it.
    ``ignore`` drops features that *define* the split -- auditing `week` across a
    week boundary reports the boundary itself, which is true and useless.
    """
    missing = [f for f in features if f not in claims.columns]
    findings = [
        Finding(f, "MISSING", "declared in FEATURE_COLUMNS but absent from the frame")
        for f in missing
    ]
    if split_col not in claims.columns:
        return pd.DataFrame(), findings

    for feature in categorical:
        if feature in claims.columns:
            findings += _audit_categorical(claims, feature, split_col)

    present = [
        f for f in features if f in claims.columns and f not in categorical and f not in ignore
    ]
    if not present:
        return pd.DataFrame(), findings

    summary = _describe(claims, present, split_col)

    for feature, part in summary.groupby("feature", observed=True):
        indexed = part.set_index("split")

        # Wholly absent in one split: the join produced nothing there at all.
        empty = [s for s, pct in indexed["null_pct"].items() if pct >= 100.0]
        if empty and len(empty) < len(indexed):
            findings.append(Finding(feature, "EMPTY", f"100% null in split(s) {empty}"))
            continue
        if len(empty) == len(indexed):
            findings.append(Finding(feature, "EMPTY", "100% null everywhere"))
            continue

        # Constant within a split teaches the model nothing and then varies at test.
        flat = [s for s, sd in indexed["std"].items() if pd.notna(sd) and np.isclose(sd, 0.0)]
        if flat:
            findings.append(Finding(feature, "CONSTANT", f"zero variance in split(s) {flat}"))

        nulls = indexed["null_pct"]
        if len(nulls.dropna()) >= 2 and (nulls.max() - nulls.min()) > NULL_SHIFT_PP:
            findings.append(
                Finding(
                    feature,
                    "NULLS",
                    f"coverage differs by {nulls.max() - nulls.min():.0f}pp "
                    f"({', '.join(f'{k}={v:.0f}%' for k, v in nulls.items())})",
                )
            )

        scale = _scale_finding(feature, indexed["mean"])
        if scale is not None:
            findings.append(scale)

    return summary, findings


def format_report(summary: pd.DataFrame, findings: list[Finding], split_col: str) -> str:
    """Human-readable report. The table is the evidence; the findings are the point."""
    lines: list[str] = []
    if not summary.empty:
        splits = sorted(summary["split"].unique())
        header = f"{'feature':22s}" + "".join(
            f"{f'{s} mean':>14s}{f'{s} null%':>12s}" for s in splits
        )
        lines += [header, "-" * len(header)]
        for feature, part in summary.groupby("feature", observed=True, sort=False):
            indexed = part.set_index("split")
            row = f"{feature:22s}"
            for split in splits:
                if split in indexed.index:
                    mean, null = indexed.loc[split, "mean"], indexed.loc[split, "null_pct"]
                    row += f"{mean:14.4g}{null:11.1f}%"
                else:
                    row += f"{'--':>14s}{'--':>12s}"
            lines.append(row)

    lines.append("")
    if findings:
        lines.append(f"{len(findings)} finding(s) — each has been a real defect in this repo:")
        lines += [f"  {f}" for f in findings]
        lines.append("")
        lines.append(
            "DEAD/EMPTY/CONSTANT usually mean a join or filter dropped a split entirely "
            "(check for a pd.concat before a filter). SCALE on a count-like feature "
            "usually means it measures the corpus rather than the player."
        )
    else:
        lines.append(f"no anomalies across {split_col}.")
    return "\n".join(lines)
