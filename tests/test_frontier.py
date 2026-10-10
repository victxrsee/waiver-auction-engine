"""Tests for crawl state persistence.

The frontier's job is to make a long crawl interruptible, so the tests are mostly
about idempotence and about state surviving a reopen.
"""

from __future__ import annotations

import pytest

from iie.crawl.frontier import Frontier


@pytest.fixture
def frontier(tmp_path):
    with Frontier(tmp_path / "frontier.db") as f:
        yield f


def test_users_are_deduplicated(frontier):
    """The same person appears in many leagues; that is normal, not an error."""
    assert frontier.add_users(["a", "b"]) == 2
    assert frontier.add_users(["b", "c"]) == 1
    assert frontier.counts()["users_total"] == 3


def test_empty_and_null_ids_are_ignored(frontier):
    assert frontier.add_users([None, "", "a"]) == 1


def test_users_are_served_shallowest_first(frontier):
    frontier.add_users(["deep"], depth=3)
    frontier.add_users(["shallow"], depth=0)
    assert [r["user_id"] for r in frontier.next_users(2)] == ["shallow", "deep"]


def test_expanded_users_are_not_served_again(frontier):
    frontier.add_users(["a"])
    frontier.mark_user_expanded("a")
    assert frontier.next_users(5) == []
    assert frontier.counts()["users_expanded"] == 1


def test_readding_an_expanded_user_does_not_requeue_it(frontier):
    """Re-discovery must not undo work already done."""
    frontier.add_users(["a"])
    frontier.mark_user_expanded("a")
    frontier.add_users(["a"])
    assert frontier.next_users(5) == []


def test_league_insert_is_idempotent(frontier):
    assert frontier.add_league("L1", "2025") is True
    assert frontier.add_league("L1", "2025") is False
    assert frontier.counts()["leagues_total"] == 1


def test_league_state_transitions_and_reasons(frontier):
    frontier.add_league("L1", "2025", state="accepted", reason="accepted")
    frontier.set_league_state("L1", "fetched", "ok")
    assert frontier.counts()["leagues_fetched"] == 1
    assert frontier.next_leagues("accepted") == []


def test_rejection_reasons_are_tallied(frontier):
    for i, reason in enumerate(["not_faab", "not_faab", "too_small"]):
        frontier.add_league(f"L{i}", "2025", state="rejected", reason=reason)
    assert frontier.rejection_reasons() == {"not_faab": 2, "too_small": 1}


def test_bad_state_is_rejected(frontier):
    with pytest.raises(ValueError, match="bad league state"):
        frontier.add_league("L1", "2025", state="banana")
    with pytest.raises(ValueError, match="bad league state"):
        frontier.set_league_state("L1", "banana")


def test_state_survives_reopening(tmp_path):
    """The point of SQLite here: a killed crawl resumes where it stopped."""
    path = tmp_path / "frontier.db"
    with Frontier(path) as f:
        f.add_users(["a", "b"])
        f.mark_user_expanded("a")
        f.add_league("L1", "2025", state="accepted")

    with Frontier(path) as f:
        assert f.counts()["users_pending"] == 1
        assert [r["user_id"] for r in f.next_users(5)] == ["b"]
        assert [r["league_id"] for r in f.next_leagues("accepted")] == ["L1"]


def test_the_queue_can_be_filtered_to_a_season(tmp_path):
    """--seasons must govern what is fetched, not only what is screened.

    A crawl aimed at the live 2026 season discovered 1,975 of its leagues and
    accepted 196, then fetched none: the queue is first-in-first-out and 2,342
    accepted 2024/2025 leagues sat ahead of them.
    """
    with Frontier(tmp_path / "f.db") as frontier:
        frontier.add_league("old1", "2025", state="accepted")
        frontier.add_league("old2", "2025", state="accepted")
        frontier.add_league("new1", "2026", state="accepted")

        unfiltered = frontier.next_leagues("accepted", 5)
        assert [r["league_id"] for r in unfiltered] == ["old1", "old2", "new1"]

        filtered = frontier.next_leagues("accepted", 5, seasons=("2026",))
        assert [r["league_id"] for r in filtered] == ["new1"]


def test_a_league_of_unknown_season_is_never_filtered_out(tmp_path):
    """Season is backfilled at screening, but a league queued before that must not
    become permanently unreachable just because a filter is in force."""
    with Frontier(tmp_path / "f.db") as frontier:
        frontier.add_league("unknown", None, state="accepted")
        got = frontier.next_leagues("accepted", 5, seasons=("2026",))
        assert [r["league_id"] for r in got] == ["unknown"]
