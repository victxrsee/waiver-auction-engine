"""Tests for claim assembly.

The properties that matter: losing bids are kept, league scoring is applied per
league, and the outcome never contains a week the bidder had already seen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from iie.crawl.store import CorpusStore
from iie.features.asof import NflCalendar
from iie.features.claims import ClaimSpec, _league_context, _waiver_adds, build_claims

GAMES = pd.DataFrame(
    {
        "season": [2025] * 4,
        "week": [1, 2, 3, 4],
        "game_type": ["REG"] * 4,
        "gameday": ["2025-09-07", "2025-09-14", "2025-09-21", "2025-09-28"],
        "gametime": ["13:00"] * 4,
    }
)


def ms(text):
    return int(pd.Timestamp(text, tz="UTC").timestamp() * 1000)


class FakeCache:
    """Serves a fixed nflverse panel without touching the network."""

    def __init__(self, panel, players=None):
        self._panel = panel
        self._players = (
            players
            if players is not None
            else pd.DataFrame({"gsis_id": ["00-0000001"], "pfr_id": ["Play00"]})
        )

    def load(self, name, season=None, columns=None):
        if name == "schedules":
            return GAMES
        if name == "players":
            return self._players
        return pd.DataFrame()

    def load_seasons(self, name, seasons, columns=None):
        if name == "player_week":
            return self._panel
        return pd.DataFrame(columns=["pfr_player_id", "season", "week", "game_type", "offense_pct"])


class FakeBridge:
    def __init__(self, mapping, defenses=()):
        self.mapping = mapping
        self.defenses = set(defenses)

    def is_defense(self, sleeper_id):
        return str(sleeper_id) in self.defenses

    def resolve(self, sleeper_id):
        gsis = self.mapping.get(str(sleeper_id))
        return type("R", (), {"gsis_id": gsis, "method": "test"})() if gsis else None


PANEL = pd.DataFrame(
    {
        "player_id": ["00-0000001"] * 4,
        "season": [2025] * 4,
        "week": [1, 2, 3, 4],
        "season_type": ["REG"] * 4,
        "position": ["WR"] * 4,
        "fantasy_points": [10.0, 20.0, 30.0, 40.0],
        "receptions": [1.0, 2.0, 3.0, 4.0],
        "targets": [2.0, 4.0, 6.0, 8.0],
        "carries": [0.0] * 4,
        "receiving_yards": [10.0, 20.0, 30.0, 40.0],
    }
)


def _corpus(tmp_path, *, rec=0.5):
    """A league with one won and one lost bid on the same player."""
    with CorpusStore(tmp_path) as store:
        store.add_league(
            {
                "league_id": "L1",
                "season": "2025",
                "sport": "nfl",
                "status": "complete",
                "total_rosters": 12,
                "draft_id": "D1",
                "roster_positions": ["WR"],
                "scoring_settings": {"rec": rec},
                "settings": {"waiver_type": 2, "type": 0},
            }
        )
        # Claim made after week 2's games, so as-of week is 2.
        for txn_id, bid, status in [("t1", 30, "complete"), ("t2", 12, "failed")]:
            store.add_transactions(
                "L1",
                "2025",
                3,
                [
                    {
                        "transaction_id": txn_id,
                        "type": "waiver",
                        "status": status,
                        "created": ms("2025-09-17 12:00"),
                        "roster_ids": [1],
                        "adds": {"9001": 1},
                        "drops": None,
                        "settings": {"waiver_bid": bid},
                    }
                ],
            )
    return tmp_path


@pytest.fixture
def built(tmp_path):
    root = _corpus(tmp_path / "corpus")
    cache = FakeCache(PANEL)
    return build_claims(
        root,
        cache,
        FakeBridge({"9001": "00-0000001"}),
        NflCalendar(GAMES),
        spec=ClaimSpec(outcome_horizon=6),
    )


def test_losing_bids_are_kept(built):
    """Half the price signal is bids that did not win."""
    assert len(built) == 2
    assert set(built["won"]) == {0, 1}
    assert sorted(built["waiver_bid"]) == [12.0, 30.0]


def test_asof_week_comes_from_the_timestamp_not_the_label(built):
    """The transaction is labelled week 3 but was made after week 2's games."""
    assert built["week"].unique().tolist() == [3]
    assert built["asof_week"].unique().tolist() == [2]


def test_trailing_features_stop_at_the_asof_week(built):
    # Weeks 1 and 2 only: 10 + 20. Week 3's 30 was not yet playable.
    assert built["fantasy_points_l3"].iloc[0] == 30.0


def test_outcome_starts_after_the_asof_week(built):
    """points_gained must be weeks 3 and 4, never week 2."""
    # (30 + 40) standard + 0.5 * (3 + 4) receptions.
    assert built["points_gained"].iloc[0] == pytest.approx(73.5)


def test_league_scoring_is_applied_per_league(tmp_path):
    """The same player is a different asset under different scoring."""
    cache, bridge, cal = FakeCache(PANEL), FakeBridge({"9001": "00-0000001"}), NflCalendar(GAMES)
    ppr = build_claims(_corpus(tmp_path / "ppr", rec=1.0), cache, bridge, cal)
    std = build_claims(_corpus(tmp_path / "std", rec=0.0), cache, bridge, cal)
    assert ppr["points_gained"].iloc[0] > std["points_gained"].iloc[0]
    assert std["points_gained"].iloc[0] == pytest.approx(70.0)


def test_unresolvable_players_are_dropped_not_guessed(tmp_path):
    built = build_claims(
        _corpus(tmp_path / "c"), FakeCache(PANEL), FakeBridge({}), NflCalendar(GAMES)
    )
    assert built.empty


def test_league_context_defaults_when_settings_are_unparseable():
    leagues = pd.DataFrame(
        {
            "league_id": ["A", "B"],
            "scoring_settings_json": ["{}", "not json"],
            "settings_json": ["{}", "not json"],
        }
    )
    context = _league_context(leagues)
    assert context["reception_points"].tolist() == [0.0, 0.0]
    assert context["waiver_budget"].isna().all()


def test_bid_is_expressed_as_a_share_of_budget(tmp_path):
    """Raw bids are denominated in each league's own currency and cannot be pooled."""
    root = tmp_path / "corpus"
    with CorpusStore(root) as store:
        store.add_league(
            {
                "league_id": "L1",
                "season": "2025",
                "sport": "nfl",
                "status": "complete",
                "total_rosters": 12,
                "draft_id": "D1",
                "roster_positions": ["WR"],
                "scoring_settings": {"rec": 0.5},
                "settings": {"waiver_type": 2, "type": 0, "waiver_budget": 200},
            }
        )
        store.add_transactions(
            "L1",
            "2025",
            3,
            [
                {
                    "transaction_id": "t1",
                    "type": "waiver",
                    "status": "complete",
                    "created": ms("2025-09-17 12:00"),
                    "roster_ids": [1],
                    "adds": {"9001": 1},
                    "drops": None,
                    "settings": {"waiver_bid": 50},
                }
            ],
        )
    built = build_claims(
        root, FakeCache(PANEL), FakeBridge({"9001": "00-0000001"}), NflCalendar(GAMES)
    )
    assert built["bid_pct"].iloc[0] == pytest.approx(0.25)


def test_non_waiver_transactions_are_excluded():
    header = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "league_id": ["L"] * 2,
            "season": ["2025"] * 2,
            "week": [3, 3],
            "status": ["complete"] * 2,
            "created_ms": [1, 1],
            "roster_ids": [[1], [1]],
            "type": ["waiver", "free_agent"],
        }
    )
    players = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "player_id": ["p", "p"],
            "roster_id": [1, 1],
            "action": ["add", "add"],
            "waiver_bid": [5.0, None],
        }
    )
    assert _waiver_adds(header, players)["transaction_id"].tolist() == ["t1"]


def test_drops_are_not_claims():
    header = pd.DataFrame(
        {
            "transaction_id": ["t1"],
            "league_id": ["L"],
            "season": ["2025"],
            "week": [3],
            "status": ["complete"],
            "created_ms": [1],
            "roster_ids": [[1]],
            "type": ["waiver"],
        }
    )
    players = pd.DataFrame(
        {
            "transaction_id": ["t1", "t1"],
            "player_id": ["added", "dropped"],
            "roster_id": [1, 1],
            "action": ["add", "drop"],
            "waiver_bid": [5.0, None],
        }
    )
    assert _waiver_adds(header, players)["player_id"].tolist() == ["added"]


def test_unlabelled_seasons_are_dropped_not_coerced():
    """A league crawled before its season was known cannot be joined to a week."""
    header = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "league_id": ["L"] * 2,
            "season": ["2025", ""],
            "week": [3, 3],
            "status": ["complete"] * 2,
            "created_ms": [1, 1],
            "roster_ids": [[1], [1]],
            "type": ["waiver"] * 2,
        }
    )
    players = pd.DataFrame(
        {
            "transaction_id": ["t1", "t2"],
            "player_id": ["p", "p"],
            "roster_id": [1, 1],
            "action": ["add", "add"],
            "waiver_bid": [5.0, 6.0],
        }
    )
    assert _waiver_adds(header, players)["transaction_id"].tolist() == ["t1"]


def test_empty_corpus_returns_empty(tmp_path):
    with CorpusStore(tmp_path / "empty"):
        pass
    assert build_claims(
        tmp_path / "empty", FakeCache(PANEL), FakeBridge({}), NflCalendar(GAMES)
    ).empty
