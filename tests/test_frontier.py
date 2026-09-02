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
