"""Tests for the feature audit.

Each test reconstructs a defect that actually shipped in this repo and asserts the
audit would have caught it. That is the whole justification for the module: all
three defects below trained, scored, and produced a plausible headline number, and
none of them was visible in the code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iie.eval.audit import audit_features, format_report

RNG = np.random.default_rng(0)


def frame(**columns) -> pd.DataFrame:
    """A two-split frame; every column not given is well-behaved."""
    n = 200
    base = {
        "season": ["2024"] * (n // 2) + ["2025"] * (n // 2),
        "points_l3": RNG.normal(10, 3, n),
    }
    return pd.DataFrame({**base, **columns})


def codes(findings) -> set[str]:
    return {f.code for f in findings}


def named(findings, feature) -> set[str]:
    return {f.code for f in findings if f.feature == feature}


def test_a_feature_dead_in_one_split_is_caught():
    """The injuries defect: `season_type` was filtered after a concat, so every 2024
    row was dropped and injury_severity came out uniformly 0.0 across the whole
    training season -- while looking perfectly healthy."""
    severity = np.concatenate([np.zeros(100), RNG.integers(0, 4, 100).astype(float)])
    _, findings = audit_features(frame(injury_severity=severity), ("injury_severity",))
    assert named(findings, "injury_severity") & {"DEAD", "CONSTANT"}


def test_a_feature_on_a_different_scale_between_splits_is_caught():
    """The demand defect: a raw count over our own corpus averaged 8.6 in the
    174-league training season and 278 in the 1,823-league test season."""
    demand = np.concatenate([RNG.poisson(8.6, 100), RNG.poisson(278, 100)]).astype(float)
    _, findings = audit_features(frame(demand_prior=demand), ("demand_prior",))
    assert "SCALE" in named(findings, "demand_prior")


def test_a_feature_missing_in_one_split_is_caught():
    """The depth-chart defect: concatenating two incompatible schemas produced NaN
    columns that downstream filters then dropped, losing one season's coverage."""
    rank = np.concatenate([np.full(100, np.nan), RNG.integers(1, 4, 100).astype(float)])
    _, findings = audit_features(frame(depth_rank=rank), ("depth_rank",))
    assert named(findings, "depth_rank") & {"EMPTY", "NULLS"}


def test_a_declared_feature_absent_from_the_frame_is_caught():
    """`total_rosters` was in FEATURE_COLUMNS and never joined, so production trained
    on 15 features while every test used 16, with no error on either side."""
    _, findings = audit_features(frame(), ("points_l3", "total_rosters"))
    assert named(findings, "total_rosters") == {"MISSING"}


def test_a_category_unseen_at_fit_time_is_caught():
    """Position `WR` was code 17 at fit and 14 at predict, because codes were derived
    per frame. A level present at test and absent at train is the signature."""
    position = ["WR"] * 100 + ["WR"] * 90 + ["LS"] * 10
    _, findings = audit_features(frame(position=position), ("position",), categorical=("position",))
    assert named(findings, "position") == {"UNSEEN"}


def test_a_shared_vocabulary_raises_nothing():
    position = ["WR", "RB"] * 50 + ["WR", "RB"] * 50
    _, findings = audit_features(frame(position=position), ("position",), categorical=("position",))
    assert not findings


# --- the checks must not cry wolf -------------------------------------------


def test_a_well_behaved_feature_raises_nothing():
    _, findings = audit_features(frame(), ("points_l3",))
    assert not findings, "a stable feature must not be reported"


def test_a_feature_that_is_legitimately_zero_everywhere_is_not_a_scale_finding():
    """All-zero in both splits is constant, not a scale shift -- reporting a ratio
    on it would be a division by zero dressed up as a finding."""
    _, findings = audit_features(frame(never_used=np.zeros(200)), ("never_used",))
    assert "SCALE" not in named(findings, "never_used")
    assert "CONSTANT" in named(findings, "never_used")


def test_features_defining_the_split_can_be_ignored():
    """Auditing `week` across a week boundary reports the boundary: true, useless."""
    week = np.concatenate([np.full(100, 5.0), np.full(100, 12.0)])
    _, noisy = audit_features(frame(week=week), ("week",))
    _, quiet = audit_features(frame(week=week), ("week",), ignore=("week",))
    assert "SCALE" in codes(noisy)
    assert not quiet


def test_a_single_split_reports_nothing_to_compare():
    single = pd.DataFrame({"season": ["2025"] * 20, "points_l3": RNG.normal(10, 3, 20)})
    _, findings = audit_features(single, ("points_l3",))
    assert not findings


def test_report_renders_with_and_without_findings():
    summary, findings = audit_features(frame(), ("points_l3",))
    assert "no anomalies" in format_report(summary, findings, "season")

    summary, findings = audit_features(frame(), ("points_l3", "absent"))
    assert "MISSING" in format_report(summary, findings, "season")
