"""Tests for the two-phase crawler.

A fake Sleeper client stands in for the network and counts requests, so the
tests can assert the thing that actually matters about this crawler: that it
does not spend the expensive fetch on leagues it is going to reject.
"""

from __future__ import annotations

import pytest

from iie.crawl.crawler import Crawler, CrawlPolicy
from iie.crawl.frontier import Frontier
from iie.crawl.store import CorpusStore, read_table


def league(
    league_id,
    season="2025",
    *,
    waiver_type=2,
    ltype=0,
    rosters=12,
    status="complete",
    previous=None,
):
    return {
        "league_id": league_id,
        "season": season,
        "sport": "nfl",
        "status": status,
        "total_rosters": rosters,
        "previous_league_id": previous,
        "draft_id": f"D{league_id}",
        "roster_positions": ["QB", "RB", "WR"],
        "scoring_settings": {"rec": 0.5},
        "settings": {"waiver_type": waiver_type, "type": ltype, "waiver_budget": 100},
    }


class FakeCache:
    def __init__(self):
        self.stats = {"hits": 0, "fetches": 0, "misses_cached": 0}


class FakeClient:
    """Counts every call so request accounting can be asserted."""

    def __init__(
        self, user_leagues=None, leagues=None, rosters=None, transactions=None, members=None
    ):
        self.cache = FakeCache()
        self.user_leagues = user_leagues or {}
        self.leagues = leagues or {}
        self.rosters = rosters or {}
        self.transactions = transactions or {}
        self.members = members or {}
        self.calls = []

    def _hit(self, kind):
        self.calls.append(kind)
        self.cache.stats["fetches"] += 1

    def get_user(self, name):
        self._hit("user")
        return {"user_id": f"uid-{name}"}

    def get_user_leagues(self, user_id, season, sport="nfl"):
        self._hit("user_leagues")
        return self.user_leagues.get((user_id, season), [])

    def get_league(self, league_id):
        self._hit("league")
        return self.leagues.get(league_id)

    def get_league_rosters(self, league_id):
        self._hit("rosters")
        return self.rosters.get(league_id, [])

    def get_league_transactions(self, league_id, week):
        self._hit("transactions")
        return self.transactions.get((league_id, week), [])

    def get_league_drafts(self, league_id):
        self._hit("drafts")
        return []

    def get_draft_picks(self, draft_id):
        self._hit("picks")
        return []

    def get_league_users(self, league_id):
        self._hit("league_users")
        return self.members.get(league_id, [])


ACTIVE_ROSTERS = [
    {
        "roster_id": 1,
        "owner_id": "u1",
        "players": ["p1"],
        "starters": ["p1"],
        "settings": {"waiver_budget_used": 90, "wins": 7, "fpts": 1400},
    },
    {
        "roster_id": 2,
        "owner_id": "u2",
        "players": ["p2"],
        "starters": [],
        "settings": {"waiver_budget_used": 80, "wins": 5, "fpts": 1200},
    },
]

DEAD_ROSTERS = [
    {
        "roster_id": 1,
        "owner_id": "u1",
        "players": ["p1"],
        "starters": [],
        "settings": {"waiver_budget_used": 0, "wins": 7, "fpts": 1400},
    },
]


@pytest.fixture
def wiring(tmp_path):
    def build(client, policy=None):
        store = CorpusStore(tmp_path / "corpus")
        frontier = Frontier(tmp_path / "f.db")
        crawler = Crawler(client, store, frontier, policy or CrawlPolicy(seasons=("2025",)))
        return crawler, store, frontier

    return build


@pytest.mark.parametrize(
    ("raw", "accepted", "reason"),
    [
        (league("A"), True, "accepted"),
        (league("A", waiver_type=0), False, "not_faab"),
        (league("A", waiver_type=1), False, "not_faab"),
        (league("A", ltype=2), False, "not_redraft"),
        (league("A", ltype=1), False, "not_redraft"),
        (league("A", rosters=4), False, "too_small"),
        (league("A", status="in_season"), False, "season_incomplete"),
        (league("A", season="2019"), False, "season_out_of_range"),
    ],
)
def test_screening_rules(raw, accepted, reason):
    assert CrawlPolicy(seasons=("2024", "2025")).screen(raw) == (accepted, reason)


def test_rejected_leagues_never_cost_a_deep_fetch(wiring):
    """The whole economics of the crawler: a reject must stay cheap."""
    client = FakeClient(
        user_leagues={
            ("uid-me", "2025"): [
                league("GOOD"),
                league("ROLLING", waiver_type=0),
                league("DYNASTY", ltype=2),
            ]
        },
        rosters={"GOOD": ACTIVE_ROSTERS},
        members={"GOOD": [{"user_id": "u1", "display_name": "x"}]},
    )
    crawler, store, _frontier = wiring(client)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()

    assert report.leagues_fetched == 1
    assert report.rejection_reasons == {"not_faab": 1, "not_redraft": 1}
    # Screening all three cost nothing beyond the season listing itself.
    assert client.calls.count("league") == 0, "screening must not refetch metadata"
    # Only the accepted league saw rosters or transactions.
    assert client.calls.count("rosters") == 1
    assert client.calls.count("transactions") == len(CrawlPolicy().transaction_weeks)


def test_activity_gate_runs_before_transactions(wiring):
    """A dead FAAB league costs one request, not fifteen."""
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("DEAD")]},
        rosters={"DEAD": DEAD_ROSTERS},
    )
    crawler, store, _frontier = wiring(client)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()

    assert report.leagues_fetched == 0
    assert report.rejection_reasons == {"inactive_faab": 1}
    assert client.calls.count("rosters") == 1
    assert client.calls.count("transactions") == 0, "gate must precede the expensive part"


def test_accepted_league_is_fully_stored(wiring, tmp_path):
    txn = {
        "transaction_id": "t1",
        "type": "waiver",
        "status": "complete",
        "created": 1699000000000,
        "roster_ids": [1],
        "adds": {"p9": 1},
        "drops": None,
        "settings": {"waiver_bid": 21},
    }
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("GOOD")]},
        rosters={"GOOD": ACTIVE_ROSTERS},
        transactions={("GOOD", 3): [txn]},
        members={"GOOD": [{"user_id": "u1"}, {"user_id": "u2"}]},
    )
    crawler, store, _frontier = wiring(client)
    crawler.seed_username("me")
    crawler.run()
    store.close()

    corpus = tmp_path / "corpus"
    assert len(read_table(corpus, "leagues")) == 1
    assert len(read_table(corpus, "rosters")) == 2
    assert len(read_table(corpus, "league_members")) == 2
    bids = read_table(corpus, "transaction_players")
    assert bids.loc[0, "waiver_bid"] == 21
    assert bids.loc[0, "player_id"] == "p9"


def test_graph_expands_through_league_membership(wiring):
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("GOOD")]},
        rosters={"GOOD": ACTIVE_ROSTERS},
        members={"GOOD": [{"user_id": "u1"}, {"user_id": "u2"}]},
    )
    crawler, store, frontier = wiring(client)
    crawler.seed_username("me")
    crawler.run()
    store.close()
    # Seed plus the two leaguemates discovered through the fetched league.
    assert frontier.counts()["users_total"] == 3


def test_a_full_queue_does_not_starve_user_expansion(wiring):
    """Breadth must not depend on the fetch queue running dry.

    Discovering only below the low water mark makes the walk depth-first: the
    queue stays full, so the crawler keeps fetching leagues from the neighbourhood
    it is already in and never expands a new user. Measured on the real crawl that
    left 139 users expanded against 20,590 pending, with one manager supplying 30%
    of all claims -- a corpus far narrower than its league count suggests.
    """
    many = [league(f"L{i}") for i in range(12)]
    client = FakeClient(
        user_leagues={("uid-me", "2025"): many, ("u1", "2025"): []},
        rosters={f"L{i}": ACTIVE_ROSTERS for i in range(12)},
        # Every league carries the same leaguemate, discoverable only by expanding.
        members={f"L{i}": [{"user_id": "u1"}] for i in range(12)},
    )
    policy = CrawlPolicy(seasons=("2025",), expand_every=4, low_water_mark=1)
    crawler, store, _frontier = wiring(client, policy)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    # The queue never drops to the low water mark, so without the cadence the seed
    # would be the only user ever expanded.
    assert report.users_expanded > 1, "a full queue starved discovery entirely"


def test_expansion_cadence_can_be_disabled(wiring):
    """expand_every=0 restores the pure low-water-mark behaviour.

    The crawl stops on its target with leagues still queued, so the terminal
    "nothing left to fetch" expansion never fires and the cadence is the only
    thing that could have expanded a second user.
    """
    many = [league(f"L{i}") for i in range(6)]
    client = FakeClient(
        user_leagues={("uid-me", "2025"): many, ("u1", "2025"): []},
        rosters={f"L{i}": ACTIVE_ROSTERS for i in range(6)},
        members={f"L{i}": [{"user_id": "u1"}] for i in range(6)},
    )
    policy = CrawlPolicy(seasons=("2025",), expand_every=0, low_water_mark=0, target_leagues=4)
    crawler, store, _frontier = wiring(client, policy)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.stopped_because == "target_reached"
    assert report.users_expanded == 1


def test_previous_league_id_is_queued_and_screened(wiring):
    """History is the cheapest data available; it must be followed."""
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("GOOD", previous="OLD")]},
        leagues={"OLD": league("OLD", season="2024")},
        rosters={"GOOD": ACTIVE_ROSTERS, "OLD": ACTIVE_ROSTERS},
    )
    policy = CrawlPolicy(seasons=("2024", "2025"))
    crawler, store, _frontier = wiring(client, policy)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.leagues_fetched == 2, "the prior season should have been crawled too"


def test_zero_previous_league_id_is_not_followed(wiring):
    """Sleeper writes '0' for 'no previous league'."""
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("GOOD", previous="0")]},
        rosters={"GOOD": ACTIVE_ROSTERS},
    )
    crawler, store, _frontier = wiring(client)
    crawler.seed_username("me")
    crawler.run()
    store.close()
    assert client.calls.count("league") == 0


def test_one_league_failing_mid_fetch_does_not_end_the_crawl(wiring):
    """A league costs fourteen requests; a blip on any one used to kill the run.

    An unattended crawl died this way after 56 leagues, on a read timeout partway
    through one league's transactions.
    """

    class Flaky(FakeClient):
        def get_league_transactions(self, league_id, week):
            if league_id == "BAD":
                raise RuntimeError("read timed out")
            return super().get_league_transactions(league_id, week)

    client = Flaky(
        user_leagues={("uid-me", "2025"): [league("BAD"), league("GOOD")]},
        rosters={"BAD": ACTIVE_ROSTERS, "GOOD": ACTIVE_ROSTERS},
    )
    crawler, store, frontier = wiring(client, CrawlPolicy(seasons=("2025",)))
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.leagues_fetched == 1, "the healthy league must still be fetched"
    assert frontier.rejection_reasons().get("fetch_failed") == 1


def test_a_sustained_outage_stops_rather_than_burning_the_frontier(wiring):
    """Abandoning one league is resilience; abandoning every queued league during
    an outage would empty the frontier with nothing to show for it."""

    class Dead(FakeClient):
        def get_league_transactions(self, league_id, week):
            raise RuntimeError("connection refused")

    client = Dead(
        user_leagues={("uid-me", "2025"): [league(f"L{i}") for i in range(30)]},
        rosters={f"L{i}": ACTIVE_ROSTERS for i in range(30)},
    )
    policy = CrawlPolicy(seasons=("2025",), max_consecutive_failures=3)
    crawler, store, frontier = wiring(client, policy)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.stopped_because == "network_unstable"
    assert report.leagues_rejected == 3, "stops at the threshold, not after all 30"
    assert frontier.counts()["leagues_accepted"] > 0, "the rest stay queued for a resume"


def test_target_stops_the_crawl(wiring):
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("A"), league("B"), league("C")]},
        rosters={k: ACTIVE_ROSTERS for k in "ABC"},
    )
    crawler, store, _frontier = wiring(client, CrawlPolicy(seasons=("2025",), target_leagues=2))
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.leagues_fetched == 2
    assert report.stopped_because == "target_reached"


def test_request_budget_stops_the_crawl(wiring):
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("A"), league("B")]},
        rosters={k: ACTIVE_ROSTERS for k in "AB"},
    )
    crawler, store, _frontier = wiring(client, CrawlPolicy(seasons=("2025",), max_requests=3))
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.stopped_because == "request_budget_exhausted"


def test_exhausted_frontier_terminates(wiring):
    client = FakeClient(user_leagues={("uid-me", "2025"): []})
    crawler, store, _frontier = wiring(client)
    crawler.seed_username("me")
    report = crawler.run()
    store.close()
    assert report.stopped_because == "frontier_exhausted"
    assert report.leagues_fetched == 0


def test_unknown_username_does_not_seed(wiring):
    class NoUser(FakeClient):
        def get_user(self, name):
            self._hit("user")

    crawler, _store, frontier = wiring(NoUser())
    assert crawler.seed_username("ghost") is None
    assert frontier.counts()["users_total"] == 0


def test_crawl_resumes_without_refetching(tmp_path):
    """A killed crawl must pick up where it stopped."""
    client = FakeClient(
        user_leagues={("uid-me", "2025"): [league("A"), league("B")]},
        rosters={k: ACTIVE_ROSTERS for k in "AB"},
    )
    policy_one = CrawlPolicy(seasons=("2025",), target_leagues=1)
    store = CorpusStore(tmp_path / "corpus")
    frontier = Frontier(tmp_path / "f.db")
    crawler = Crawler(client, store, frontier, policy_one)
    crawler.seed_username("me")
    crawler.run()
    store.close()
    frontier.close()

    store = CorpusStore(tmp_path / "corpus")
    frontier = Frontier(tmp_path / "f.db")
    resumed = Crawler(client, store, frontier, CrawlPolicy(seasons=("2025",), target_leagues=2))
    report = resumed.run()
    store.close()

    assert report.leagues_fetched == 1, "only the league left over should be fetched"
    assert client.calls.count("user_leagues") == 1, "the seed user was already expanded"
