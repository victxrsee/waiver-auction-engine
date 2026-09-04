"""Sleeper read-only API client.

Sleeper's v1 API needs no auth for any of these endpoints, which is the property
that makes a league corpus obtainable at all. The user<->league traversal is
bidirectional:

    user -> /user/{id}/leagues/nfl/{season} -> leagues
    league -> /league/{id}/users            -> users

so the two together form a graph the crawler can walk.

Nothing here persists data. The client returns raw API shapes; normalization and
the decision about what is safe to store live in ``iie.crawl.store``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import DEFAULT_RATE_PER_SECOND, HttpCache, RateLimiter

BASE_URL = "https://api.sleeper.app/v1"

# A finished season never changes, so its responses are cached forever. Anything
# fetched for the in-progress season should pass an explicit ttl.
IMMUTABLE = None

# The full player list is ~14MB and Sleeper asks callers to fetch it at most once
# a day. It is also the only source of the cross-reference IDs the nflverse bridge
# needs, so it belongs in the client rather than being fetched by hand.
PLAYERS_TTL = 24 * 3600


class SleeperClient:
    def __init__(self, cache: HttpCache) -> None:
        self.cache = cache

    @classmethod
    def build(
        cls,
        cache_root: Path | str,
        *,
        per_second: float = DEFAULT_RATE_PER_SECOND,
    ) -> SleeperClient:
        return cls(HttpCache(cache_root, BASE_URL, rate_limiter=RateLimiter(per_second)))

    # --- entry points -----------------------------------------------------

    def get_state(self, sport: str = "nfl") -> dict[str, Any] | None:
        """Current week/season. Cheap; useful as a connectivity smoke test."""
        return self.cache.get_json(f"state/{sport}", ttl=3600)

    def get_user(self, username_or_id: str) -> dict[str, Any] | None:
        return self.cache.get_json(f"user/{username_or_id}", ttl=IMMUTABLE)

    def get_players(self, sport: str = "nfl") -> dict[str, dict[str, Any]]:
        """Every player Sleeper knows, keyed by its own player_id.

        Large and slow-moving, so it is cached for a day. Team defenses appear
        here too, keyed by team abbreviation rather than a player id.
        """
        got = self.cache.get_json(f"players/{sport}", ttl=PLAYERS_TTL)
        return got or {}

    # --- graph traversal --------------------------------------------------

    def get_user_leagues(
        self, user_id: str, season: str, sport: str = "nfl"
    ) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"user/{user_id}/leagues/{sport}/{season}", ttl=IMMUTABLE)
        return got or []

    def get_league_users(self, league_id: str) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"league/{league_id}/users", ttl=IMMUTABLE)
        return got or []

    # --- league contents --------------------------------------------------

    def get_league(self, league_id: str) -> dict[str, Any] | None:
        return self.cache.get_json(f"league/{league_id}", ttl=IMMUTABLE)

    def get_league_rosters(self, league_id: str) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"league/{league_id}/rosters", ttl=IMMUTABLE)
        return got or []

    def get_league_transactions(self, league_id: str, week: int) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"league/{league_id}/transactions/{week}", ttl=IMMUTABLE)
        return got or []

    def get_league_drafts(self, league_id: str) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"league/{league_id}/drafts", ttl=IMMUTABLE)
        return got or []

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        return self.cache.get_json(f"draft/{draft_id}", ttl=IMMUTABLE)

    def get_draft_picks(self, draft_id: str) -> list[dict[str, Any]]:
        got = self.cache.get_json(f"draft/{draft_id}/picks", ttl=IMMUTABLE)
        return got or []
