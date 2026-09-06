"""Two-phase corpus crawler over the Sleeper league graph.

The graph is walked user -> leagues -> users. What makes it affordable is that
``/user/{id}/leagues/{sport}/{season}`` returns *complete* league objects, settings
included, so screening a league costs zero extra requests. Only leagues that
survive screening pay the ~17-request deep fetch. Rejecting a league is therefore
almost free, and filtering is worth far more than raw speed: skipping one
non-FAAB league saves twenty requests, while going faster only saves time on
requests that should not have been made.

Three steps run interleaved, cheapest first:

1. *expand* a pending user -- one request per season, screening included.
2. *screen* a league discovered through ``previous_league_id`` -- one request, the
   only case where metadata is not already in hand.
3. *fetch* an accepted league -- rosters, then an activity gate, then
   transactions and drafts.

The activity gate sits deliberately between rosters and transactions. Rosters
carry ``waiver_budget_used``, so one request reveals whether a league ran a real
FAAB market before fourteen more are spent discovering that it did not.

Expansion runs through accepted leagues only, by default. That biases the walk
toward FAAB-dense neighbourhoods, which is the bias we want -- FAAB players
cluster with other FAAB players -- and it keeps the crawl from spending its
budget wandering through leagues it will never use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..sources.sleeper import SleeperClient
from .frontier import Frontier
from .store import CorpusStore

log = logging.getLogger(__name__)

# Sleeper waiver_type 2 is FAAB; 0 and 1 are rolling and reverse-standings, and
# produce no bid data at all. league type 0 is redraft; 1 keeper, 2 dynasty.
WAIVER_TYPE_FAAB = 2
LEAGUE_TYPE_REDRAFT = 0


@dataclass(frozen=True)
class CrawlPolicy:
    """What the crawl will accept and how much it is allowed to spend."""

    seasons: tuple[str, ...] = ("2024", "2025")
    max_depth: int = 4
    target_leagues: int = 500
    max_requests: int = 40_000

    min_rosters: int = 8
    # A league that spent almost nothing ran no meaningful FAAB market, so its
    # transactions are not worth fourteen requests.
    min_budget_used: int = 50
    # Weeks 15-17 are fantasy playoffs: little waiver activity, ~18% of the cost.
    transaction_weeks: tuple[int, ...] = tuple(range(1, 15))

    expand_through_rejected: bool = False
    # Keep enough accepted leagues queued that fetching never starves.
    low_water_mark: int = 20
    # ...but not *only* when it would starve. Discovering solely below the low
    # water mark makes the walk depth-first: the queue stays full, so the crawler
    # keeps fetching leagues from the neighbourhood it is already inside and never
    # expands a new user. Measured after 1,906 leagues: 139 users expanded against
    # 20,590 pending, and one manager accounting for 30% of all claims. Forcing a
    # user expansion every N fetches costs ~2 requests against the ~18 a league
    # costs, and buys the sample breadth the estimates depend on.
    expand_every: int = 8
    require_complete: bool = True

    def screen(self, raw: dict[str, Any]) -> tuple[bool, str]:
        """Decide a league from its metadata alone. Pure, so it is testable."""
        settings = raw.get("settings") or {}
        season = str(raw.get("season") or "")

        if season not in self.seasons:
            return False, "season_out_of_range"
        if self.require_complete and raw.get("status") != "complete":
            return False, "season_incomplete"
        if settings.get("waiver_type") != WAIVER_TYPE_FAAB:
            return False, "not_faab"
        if settings.get("type") != LEAGUE_TYPE_REDRAFT:
            return False, "not_redraft"
        if (raw.get("total_rosters") or 0) < self.min_rosters:
            return False, "too_small"
        return True, "accepted"


@dataclass
class CrawlReport:
    """What a run did, so a disappointing corpus can be diagnosed."""

    requests: int = 0
    users_expanded: int = 0
    leagues_screened: int = 0
    leagues_fetched: int = 0
    leagues_rejected: int = 0
    stopped_because: str = "unfinished"
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    frontier_counts: dict[str, int] = field(default_factory=dict)


class Crawler:
    def __init__(
        self,
        client: SleeperClient,
        store: CorpusStore,
        frontier: Frontier,
        policy: CrawlPolicy | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.frontier = frontier
        self.policy = policy or CrawlPolicy()
        self.report = CrawlReport()
        self._start_requests = self._requests_used()
        # Fetch count at the last forced expansion, so one cadence hit fires once.
        self._last_expansion = -1

    def _requests_used(self) -> int:
        """Network fetches only. Cache hits are free and must not spend budget."""
        return self.client.cache.stats["fetches"]

    @property
    def spent(self) -> int:
        return self._requests_used() - self._start_requests

    # --- seeding ----------------------------------------------------------

    def seed_username(self, username: str) -> str | None:
        """Resolve a username to a user_id and queue it. Returns the id."""
        user = self.client.get_user(username)
        if not user or not user.get("user_id"):
            log.error("no such Sleeper user: %s", username)
            return None
        user_id = str(user["user_id"])
        self.frontier.add_users([user_id], depth=0)
        log.info("seeded %s -> %s", username, user_id)
        return user_id

    def seed_user_ids(self, user_ids: list[str]) -> int:
        return self.frontier.add_users(user_ids, depth=0)

    # --- steps ------------------------------------------------------------

    def _expand_user(self, user_id: str, depth: int) -> None:
        """List a user's leagues for every season and screen them in place."""
        for season in self.policy.seasons:
            for raw in self.client.get_user_leagues(user_id, season):
                league_id = raw.get("league_id")
                if not league_id or self.frontier.knows_league(league_id):
                    continue

                accepted, reason = self.policy.screen(raw)
                self.frontier.add_league(
                    league_id,
                    str(raw.get("season") or season),
                    state="accepted" if accepted else "rejected",
                    reason=reason,
                    depth=depth,
                )
                self.report.leagues_screened += 1
                if accepted:
                    self.store.add_league(raw)
                    # A league chains backward a season at a time, and history is
                    # what the realized-outcome model trains on.
                    previous = raw.get("previous_league_id")
                    if previous and previous != "0" and not self.frontier.knows_league(previous):
                        self.frontier.add_league(previous, None, state="discovered", depth=depth)
                else:
                    self.report.leagues_rejected += 1

        self.frontier.mark_user_expanded(user_id)
        self.report.users_expanded += 1

    def _screen_discovered(self, league_id: str, depth: int) -> None:
        """Screen a league reached via previous_league_id, which needs a fetch."""
        raw = self.client.get_league(league_id)
        self.report.leagues_screened += 1
        if not raw:
            self.frontier.set_league_state(league_id, "rejected", "not_found")
            self.report.leagues_rejected += 1
            return

        accepted, reason = self.policy.screen(raw)
        if not accepted:
            self.frontier.set_league_state(league_id, "rejected", reason)
            self.report.leagues_rejected += 1
            return

        self.store.add_league(raw)
        # The season was unknown when this league was queued from a
        # previous_league_id; record it now or its rows cannot be joined.
        self.frontier.set_league_state(league_id, "accepted", reason, season=raw.get("season"))
        previous = raw.get("previous_league_id")
        if previous and previous != "0" and not self.frontier.knows_league(previous):
            self.frontier.add_league(previous, None, state="discovered", depth=depth)

    def _fetch_league(self, league_id: str, season: str | None, depth: int) -> None:
        """Deep fetch, with the activity gate before the expensive part."""
        rosters = self.client.get_league_rosters(league_id)
        if not rosters:
            self.frontier.set_league_state(league_id, "rejected", "no_rosters")
            self.report.leagues_rejected += 1
            return

        spent = sum(((r.get("settings") or {}).get("waiver_budget_used") or 0) for r in rosters)
        if spent < self.policy.min_budget_used:
            self.frontier.set_league_state(league_id, "rejected", "inactive_faab")
            self.report.leagues_rejected += 1
            return

        self.store.add_rosters(league_id, rosters)

        if not season:
            # Should not happen now that discovery backfills it, but an unlabelled
            # season must stay null rather than become "", which parses as neither
            # a season nor a missing value downstream.
            log.warning("league %s has no season; its rows will not be joinable", league_id)
            season = None
        for week in self.policy.transaction_weeks:
            transactions = self.client.get_league_transactions(league_id, week)
            if transactions:
                self.store.add_transactions(league_id, season, week, transactions)

        for draft in self.client.get_league_drafts(league_id):
            draft_id = draft.get("draft_id")
            if draft_id:
                picks = self.client.get_draft_picks(draft_id)
                if picks:
                    self.store.add_draft_picks(draft_id, league_id, season, picks)

        # Membership last: it both records the edge and feeds the next hop.
        users = self.client.get_league_users(league_id)
        self.store.add_league_users(league_id, users)
        if depth < self.policy.max_depth:
            self.frontier.add_users(
                [u.get("user_id") for u in users if u.get("user_id")], depth=depth + 1
            )

        self.frontier.set_league_state(league_id, "fetched", "ok")
        self.report.leagues_fetched += 1

    # --- driver -----------------------------------------------------------

    def run(self) -> CrawlReport:
        """Interleave the three steps until a stop condition trips."""
        while True:
            if self.report.leagues_fetched >= self.policy.target_leagues:
                self.report.stopped_because = "target_reached"
                break
            if self.spent >= self.policy.max_requests:
                self.report.stopped_because = "request_budget_exhausted"
                break

            queued = self.frontier.counts()["leagues_accepted"]
            starving = queued < self.policy.low_water_mark
            # Breadth on a fixed cadence, so a full queue cannot starve discovery.
            due = self.policy.expand_every > 0 and (
                self.report.leagues_fetched
                and self.report.leagues_fetched % self.policy.expand_every == 0
                and self.report.leagues_fetched != self._last_expansion
            )
            if (starving or due) and self._discover_once():
                self._last_expansion = self.report.leagues_fetched
                continue
            if self._fetch_once():
                continue
            # Nothing queued and nothing left to discover.
            if not self._discover_once():
                self.report.stopped_because = "frontier_exhausted"
                break

        self.report.requests = self.spent
        self.report.rejection_reasons = self.frontier.rejection_reasons()
        self.report.frontier_counts = self.frontier.counts()
        return self.report

    def _discover_once(self) -> bool:
        """One cheap step: expand a user, else screen a discovered league."""
        users = self.frontier.next_users(1)
        if users:
            self._expand_user(users[0]["user_id"], users[0]["depth"])
            self._log_progress()
            return True

        discovered = self.frontier.next_leagues("discovered", 1)
        if discovered:
            row = discovered[0]
            self._screen_discovered(row["league_id"], row["depth"])
            return True
        return False

    def _fetch_once(self) -> bool:
        accepted = self.frontier.next_leagues("accepted", 1)
        if not accepted:
            return False
        row = accepted[0]
        self._fetch_league(row["league_id"], row["season"], row["depth"])
        self._log_progress()
        return True

    def _log_progress(self) -> None:
        if self.report.leagues_fetched and self.report.leagues_fetched % 25 == 0:
            log.info(
                "fetched %d/%d leagues | %d requests | %d rejected",
                self.report.leagues_fetched,
                self.policy.target_leagues,
                self.spent,
                self.report.leagues_rejected,
            )
