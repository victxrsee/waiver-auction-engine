"""Tests for nflverse asset fetching.

No network: a fake session stands in for GitHub so the caching, staleness and
season-validation logic is testable offline.
"""

from __future__ import annotations

import datetime as dt

import pytest

from iie.sources.nflverse import DATASETS, NflverseCache, current_season


class FakeResponse:
    def __init__(self, payload: bytes = b"data", status: int = 200) -> None:
        self.content = payload
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected status {self.status_code}")

    def iter_content(self, chunk_size=None):
        yield self.content


class FakeSession:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls: list[str] = []

    def get(self, url, **kwargs):
        self.calls.append(url)
        return FakeResponse(status=self.status)


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (dt.date(2026, 9, 1), 2026),  # in season
        (dt.date(2026, 3, 1), 2026),  # league year has flipped
        (dt.date(2027, 1, 20), 2026),  # playoffs still belong to 2026
        (dt.date(2027, 2, 8), 2026),  # so does the Super Bowl
    ],
)
def test_season_label_follows_the_league_year(today, expected):
    assert current_season(today) == expected


def test_asset_urls_match_the_published_naming():
    assert (
        DATASETS["snap_counts"]
        .url(2024)
        .endswith("/nflverse-data/releases/download/snap_counts/snap_counts_2024.parquet")
    )
    # stats_player_week lives under the stats_player tag, not a matching one.
    assert (
        DATASETS["player_week"]
        .url(2024)
        .endswith("/nflverse-data/releases/download/stats_player/stats_player_week_2024.parquet")
    )
    assert DATASETS["players"].url(None).endswith("/players/players.parquet")


def test_season_before_publication_fails_loudly(tmp_path):
    cache = NflverseCache(tmp_path, session=FakeSession())
    with pytest.raises(ValueError, match="from 2012"):
        cache.fetch("snap_counts", 2010)


def test_seasonal_dataset_requires_a_season(tmp_path):
    cache = NflverseCache(tmp_path, session=FakeSession())
    with pytest.raises(ValueError, match="season-partitioned"):
        cache.fetch("snap_counts")


def test_unknown_dataset_is_rejected(tmp_path):
    cache = NflverseCache(tmp_path, session=FakeSession())
    with pytest.raises(KeyError):
        cache.fetch("not_a_dataset", 2024)


def test_completed_season_is_never_refetched(tmp_path):
    session = FakeSession()
    cache = NflverseCache(tmp_path, session=session)
    first = cache.fetch("snap_counts", 2020)
    again = cache.fetch("snap_counts", 2020)
    assert first == again
    assert len(session.calls) == 1, "a finished season is immutable; one fetch only"


def test_live_season_refetches_once_stale(tmp_path, monkeypatch):
    import iie.sources.nflverse as mod

    session = FakeSession()
    cache = NflverseCache(tmp_path, session=session)
    live = current_season()
    cache.fetch("depth_charts", live)
    assert len(session.calls) == 1

    cache.fetch("depth_charts", live)
    assert len(session.calls) == 1, "still fresh"

    # Capture the real clock before patching, or the replacement calls itself.
    now = mod.time.time()
    monkeypatch.setattr(mod.time, "time", lambda: now + 10 * 3600)
    cache.fetch("depth_charts", live)
    assert len(session.calls) == 2, "past the TTL, nflverse may have republished"


def test_interrupted_download_leaves_no_usable_file(tmp_path):
    class ExplodingSession(FakeSession):
        def get(self, url, **kwargs):
            super().get(url, **kwargs)
            raise __import__("requests").RequestException("connection reset")

    cache = NflverseCache(tmp_path, session=ExplodingSession(), max_retries=1)
    with pytest.raises(RuntimeError, match="giving up"):
        cache.fetch("snap_counts", 2020)
    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.tmp"))


def test_missing_asset_reports_the_url(tmp_path):
    cache = NflverseCache(tmp_path, session=FakeSession(status=404))
    with pytest.raises(FileNotFoundError, match="no asset at"):
        cache.fetch("snap_counts", 2020)
