"""Round-trip and normalization tests for the corpus store.

Payloads mirror real Sleeper shapes, including the awkward ones: a transaction
with no drops, a roster whose settings predate the waiver fields, and a user
record full of the identifying fields we deliberately discard.
"""

from __future__ import annotations

import json

import pytest

from iie.crawl.store import (
    SCHEMAS,
    CorpusStore,
    normalize_league_members,
    normalize_roster,
    normalize_transaction_players,
    read_table,
)

RAW_LEAGUE = {
    "league_id": "784462448236363776",
    "season": "2023",
    "sport": "nfl",
    "status": "complete",
    "name": "Some Private League Name",
    "total_rosters": 12,
    "previous_league_id": "651002222222222222",
    "draft_id": "784462448236363777",
    "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN"],
    "scoring_settings": {"rec": 0.5, "pass_td": 4.0},
    "settings": {"waiver_type": 2, "waiver_budget": 100},
}

RAW_USERS = [
    {
        "user_id": "1234",
        "display_name": "somebody",
        "avatar": "abc",
        "is_owner": True,
        "metadata": {"team_name": "Real Person's Team"},
    },
    {"user_id": "5678", "display_name": "another", "avatar": None},
]

RAW_ROSTERS = [
    {
        "roster_id": 1,
        "owner_id": "1234",
        "players": ["4034", "6794", "1466"],
        "starters": ["4034", "6794"],
        "settings": {
            "wins": 9,
            "losses": 5,
            "ties": 0,
            "fpts": 1543,
            "fpts_decimal": 46,
            "fpts_against": 1402,
            "fpts_against_decimal": 8,
            "waiver_budget_used": 63,
            "waiver_position": 4,
        },
    },
    # An older league: no waiver fields, no decimal component.
    {
        "roster_id": 2,
        "owner_id": "5678",
        "players": ["2307"],
        "starters": [],
        "settings": {"wins": 5, "losses": 9, "ties": 0, "fpts": 1201},
    },
]

RAW_TRANSACTIONS = [
    {
        "transaction_id": "tx-waiver-1",
        "type": "waiver",
        "status": "complete",
        "created": 1699000000000,
        "roster_ids": [1],
        "adds": {"6794": 1},
        "drops": {"1466": 1},
        "settings": {"waiver_bid": 27},
    },
    # A free-agent pickup: no bid, and drops is explicitly null.
    {
        "transaction_id": "tx-fa-2",
        "type": "free_agent",
        "status": "complete",
        "created": 1699100000000,
        "roster_ids": [2],
        "adds": {"2307": 2},
        "drops": None,
        "settings": {},
    },
]

RAW_PICKS = [
    {
        "pick_no": 1,
        "round": 1,
        "draft_slot": 1,
        "roster_id": 1,
        "player_id": "4034",
        "is_keeper": None,
    },
    {
        "pick_no": 2,
        "round": 1,
        "draft_slot": 2,
        "roster_id": 2,
        "player_id": "2307",
        "is_keeper": True,
    },
]


@pytest.fixture
def corpus(tmp_path):
    with CorpusStore(tmp_path, buffer_rows=2) as store:
        store.add_league(RAW_LEAGUE)
        store.add_league_users(RAW_LEAGUE["league_id"], RAW_USERS)
        store.add_rosters(RAW_LEAGUE["league_id"], RAW_ROSTERS)
        store.add_transactions(RAW_LEAGUE["league_id"], "2023", 10, RAW_TRANSACTIONS)
        store.add_draft_picks(RAW_LEAGUE["draft_id"], RAW_LEAGUE["league_id"], "2023", RAW_PICKS)
    return tmp_path


def test_identifying_fields_are_not_stored(corpus):
    """Display names and team names must not survive normalization."""
    rows = normalize_league_members("L", RAW_USERS)
    assert rows == [
        {"league_id": "L", "user_id": "1234", "is_owner": True},
        {"league_id": "L", "user_id": "5678", "is_owner": False},
    ]

    members = read_table(corpus, "league_members")
    assert set(members.columns) == {"league_id", "user_id", "is_owner"}
    blob = read_table(corpus, "leagues").to_json()
    assert "Some Private League Name" not in blob


def test_split_score_fields_are_recombined():
    assert normalize_roster("L", RAW_ROSTERS[0])["points_for"] == pytest.approx(1543.46)
    assert normalize_roster("L", RAW_ROSTERS[0])["points_against"] == pytest.approx(1402.08)


def test_missing_legacy_settings_become_null_not_errors():
    roster = normalize_roster("L", RAW_ROSTERS[1])
    assert roster["waiver_budget_used"] is None
    assert roster["waiver_position"] is None
    assert roster["points_for"] == pytest.approx(1201.0)


def test_transaction_players_explodes_adds_and_drops():
    rows = normalize_transaction_players("L", "2023", 10, RAW_TRANSACTIONS[0])
    assert sorted((r["player_id"], r["action"], r["waiver_bid"]) for r in rows) == [
        ("1466", "drop", None),
        ("6794", "add", 27),
    ]


def test_null_drops_map_is_tolerated():
    rows = normalize_transaction_players("L", "2023", 10, RAW_TRANSACTIONS[1])
    assert [(r["player_id"], r["action"], r["waiver_bid"]) for r in rows] == [("2307", "add", None)]


def test_roundtrip_preserves_rows_and_types(corpus):
    leagues = read_table(corpus, "leagues")
    assert len(leagues) == 1
    assert json.loads(leagues.loc[0, "scoring_settings_json"])["rec"] == 0.5
    assert list(leagues.loc[0, "roster_positions"])[:2] == ["QB", "RB"]

    roster_players = read_table(corpus, "roster_players")
    assert len(roster_players) == 4
    starters = roster_players[roster_players.is_starter].player_id.tolist()
    assert sorted(starters) == ["4034", "6794"]

    txn_players = read_table(corpus, "transaction_players")
    assert len(txn_players) == 3
    assert txn_players[txn_players.action == "add"].waiver_bid.max() == 27

    picks = read_table(corpus, "draft_picks")
    assert picks.is_keeper.tolist() == [False, True]


def test_recrawl_duplicates_collapse(tmp_path):
    """Writing the same league twice must not double it in the corpus."""
    for _ in range(2):
        with CorpusStore(tmp_path) as store:
            store.add_league(RAW_LEAGUE)
            store.add_rosters(RAW_LEAGUE["league_id"], RAW_ROSTERS)

    assert len(read_table(tmp_path, "leagues")) == 1
    assert len(read_table(tmp_path, "rosters")) == 2
    assert len(read_table(tmp_path, "leagues", dedupe=False)) == 2


def test_missing_table_returns_typed_empty_frame(tmp_path):
    frame = read_table(tmp_path, "transactions")
    assert frame.empty
    assert list(frame.columns) == [f.name for f in SCHEMAS["transactions"]]


def test_unknown_table_name_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        read_table(tmp_path, "not_a_table")


def test_buffer_flushes_across_multiple_part_files(corpus):
    """buffer_rows=2 must produce several parts that still read as one dataset."""
    parts = list((corpus / "roster_players").glob("*.parquet"))
    assert len(parts) > 1
    assert not list((corpus / "roster_players").glob("*.tmp"))
