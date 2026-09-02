"""Tests for the Sleeper -> nflverse player ID bridge.

The tiers exist because no single key spans both corpora, so the tests are mostly
about tier *ordering* and about refusing to guess when a match is ambiguous.
"""

from __future__ import annotations

import pandas as pd
import pytest

from iie.sources.playerids import PlayerIdBridge, _clean_sleeper_id, normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Marvin Harrison Jr.", "marvinharrison"),
        ("marvin harrison", "marvinharrison"),
        ("Ja'Marr Chase", "jamarrchase"),
        ("A.J. Brown", "ajbrown"),
        ("Michael Pittman Jr", "michaelpittman"),
        ("Amon-Ra St. Brown", "amonrastbrown"),
        ("Robert Griffin III", "robertgriffin"),
        (None, None),
        ("", None),
    ],
)
def test_name_normalization_folds_source_disagreements(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("6918.0", "6918"), ("6918", "6918"), (6918, "6918"), (None, None), ("", None), ("nan", None)],
)
def test_csv_numeric_ids_are_cleaned(raw, expected):
    """A numeric sleeper_id read from CSV arrives as '6918.0' and must not stay that way."""
    assert _clean_sleeper_id(raw) == expected


SLEEPER = {
    # Has its own gsis_id: tier 1.
    "1": {"full_name": "Direct Guy", "position": "WR", "gsis_id": "00-0000001"},
    # No gsis_id, but the crosswalk knows it: tier 2.
    "2": {"full_name": "Crosswalk Guy", "position": "RB", "gsis_id": None},
    # Neither: only a name match can save it. This is the rookie case.
    "3": {"full_name": "Rookie Guy Jr.", "position": "TE", "gsis_id": None},
    # Name is ambiguous in nflverse, so it must stay unresolved.
    "4": {"full_name": "Ambiguous Guy", "position": "WR", "gsis_id": None},
    # Not in any source at all.
    "5": {"full_name": "Ghost Guy", "position": "QB", "gsis_id": None},
}

ID_MAP = pd.DataFrame({"sleeper_id": ["2.0"], "gsis_id": ["00-0000002"]})

NFLVERSE = pd.DataFrame(
    {
        "gsis_id": ["00-0000003", "00-0000004", "00-0000044", "00-0000001"],
        "display_name": ["Rookie Guy", "Ambiguous Guy", "Ambiguous Guy", "Direct Guy"],
        "position": ["TE", "WR", "WR", "WR"],
    }
)


@pytest.fixture
def bridge():
    return PlayerIdBridge(SLEEPER, id_map=ID_MAP, nflverse_players=NFLVERSE)


def test_each_tier_resolves_and_is_labelled(bridge):
    assert (bridge.resolve("1").gsis_id, bridge.resolve("1").method) == ("00-0000001", "sleeper")
    assert (bridge.resolve("2").gsis_id, bridge.resolve("2").method) == ("00-0000002", "crosswalk")
    assert (bridge.resolve("3").gsis_id, bridge.resolve("3").method) == ("00-0000003", "name")


def test_suffix_mismatch_still_matches_by_name(bridge):
    """Sleeper says 'Rookie Guy Jr.', nflverse says 'Rookie Guy'."""
    assert bridge.resolve("3").gsis_id == "00-0000003"


def test_ambiguous_name_is_refused_not_guessed(bridge):
    """Two players share a name and position: a wrong join is worse than a null."""
    assert bridge.resolve("4") is None


def test_unknown_player_resolves_to_none(bridge):
    assert bridge.resolve("5") is None


def test_direct_id_wins_over_name_index():
    """Tier order must hold even when a lower tier would answer differently."""
    conflicting = pd.DataFrame(
        {"gsis_id": ["00-0009999"], "display_name": ["Direct Guy"], "position": ["WR"]}
    )
    bridge = PlayerIdBridge(SLEEPER, id_map=None, nflverse_players=conflicting)
    resolved = bridge.resolve("1")
    assert (resolved.gsis_id, resolved.method) == ("00-0000001", "sleeper")


def test_crosswalk_wins_over_name_index():
    conflicting = pd.DataFrame(
        {"gsis_id": ["00-0009999"], "display_name": ["Crosswalk Guy"], "position": ["RB"]}
    )
    bridge = PlayerIdBridge(SLEEPER, id_map=ID_MAP, nflverse_players=conflicting)
    assert bridge.resolve("2").method == "crosswalk"


def test_coverage_reports_the_tier_split(bridge):
    cov = bridge.coverage(SLEEPER.keys())
    assert cov == {
        "sleeper": 1,
        "crosswalk": 1,
        "name": 1,
        "resolved": 3,
        "unresolved": 2,
    }


def test_resolve_many_omits_failures(bridge):
    assert bridge.resolve_many(["1", "4", "5"]) == {"1": "00-0000001"}


def test_bridge_degrades_without_optional_sources():
    """With neither auxiliary source, tier 1 must still work."""
    bridge = PlayerIdBridge(SLEEPER)
    assert bridge.resolve("1").gsis_id == "00-0000001"
    assert bridge.resolve("2") is None
