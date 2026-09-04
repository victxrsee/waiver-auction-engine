"""Tests for the Sleeper client's TTL choices.

The client is thin, so what is worth pinning is which endpoints are treated as
immutable and which are not — getting that wrong either serves stale data or
re-fetches a 14MB payload on every run.
"""

from __future__ import annotations

from iie.sources.sleeper import PLAYERS_TTL, SleeperClient


class RecordingCache:
    def __init__(self, payload=None):
        self.payload = payload
        self.calls = []

    def get_json(self, path, ttl=None):
        self.calls.append((path, ttl))
        return self.payload


def test_player_list_is_cached_for_a_day():
    """Sleeper asks callers to fetch this at most daily; it is ~14MB."""
    cache = RecordingCache({"6918": {"full_name": "Salvon Ahmed"}})
    client = SleeperClient(cache)
    assert client.get_players() == {"6918": {"full_name": "Salvon Ahmed"}}
    assert cache.calls == [("players/nfl", PLAYERS_TTL)]
    assert PLAYERS_TTL == 24 * 3600


def test_missing_player_list_is_an_empty_mapping_not_none():
    """Callers index into this directly; None would fail far from the cause."""
    assert SleeperClient(RecordingCache(None)).get_players() == {}


def test_finished_season_endpoints_never_expire():
    cache = RecordingCache([])
    client = SleeperClient(cache)
    client.get_league_transactions("L1", 4)
    client.get_draft_picks("D1")
    assert all(ttl is None for _, ttl in cache.calls), "completed seasons are immutable"


def test_state_is_not_cached_indefinitely():
    """Current week moves; caching it forever would freeze the engine's clock."""
    cache = RecordingCache({})
    SleeperClient(cache).get_state()
    assert cache.calls[0][1] is not None
