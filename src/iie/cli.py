"""Command line entry point.

Every stage of this project was originally driven by ad-hoc scripts, which meant
no published number could be reproduced by anyone who had not watched it happen.
These commands are the same code paths with their arguments named.

The stages are separable on purpose. Crawling is slow and network-bound, building
claims is fast and pure, and evaluation is fast and repeated constantly, so a
change to the model should not re-crawl anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("iie")

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_SEASONS = ("2024", "2025")


def _paths(root: Path) -> dict[str, Path]:
    return {
        "httpcache": root / "httpcache",
        "nflverse": root / "nflverse",
        "idmap": root / "idmap",
        "corpus": root / "corpus",
        "frontier": root / "frontier.db",
        "claims": root / "claims.parquet",
    }


def _seasons(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


# --- commands ---------------------------------------------------------------


def cmd_crawl(args: argparse.Namespace) -> int:
    from .crawl.crawler import Crawler, CrawlPolicy
    from .crawl.frontier import Frontier
    from .crawl.store import CorpusStore
    from .sources.sleeper import SleeperClient

    paths = _paths(args.data)
    client = SleeperClient.build(paths["httpcache"], per_second=args.rate)
    policy = CrawlPolicy(
        seasons=_seasons(args.seasons),
        target_leagues=args.target,
        max_requests=args.max_requests,
    )

    with Frontier(paths["frontier"]) as frontier, CorpusStore(paths["corpus"]) as store:
        crawler = Crawler(client, store, frontier, policy)
        if args.seed and not crawler.seed_username(args.seed):
            return 1
        report = crawler.run()

    log.info("stopped: %s after %d requests", report.stopped_because, report.requests)
    log.info(
        "fetched %d leagues, rejected %d (%s)",
        report.leagues_fetched,
        report.leagues_rejected,
        report.rejection_reasons,
    )
    return 0


def cmd_claims(args: argparse.Namespace) -> int:
    from .crawl.store import read_table
    from .eval.clustered import attach_clusters
    from .features.asof import NflCalendar
    from .features.claims import ClaimSpec, build_claims
    from .sources.nflverse import NflverseCache
    from .sources.playerids import build_bridge
    from .sources.sleeper import SleeperClient

    paths = _paths(args.data)
    client = SleeperClient.build(paths["httpcache"])
    nflverse = NflverseCache(paths["nflverse"])

    bridge = build_bridge(client, paths["idmap"], nflverse)
    calendar = NflCalendar.from_cache(nflverse)
    spec = ClaimSpec(outcome_horizon=args.horizon, include_defense=not args.no_defense)

    claims = build_claims(paths["corpus"], nflverse, bridge, calendar, spec=spec)
    if claims.empty:
        log.error("no claims built; is the corpus populated?")
        return 1

    claims = attach_clusters(claims, read_table(paths["corpus"], "leagues"))
    out = args.out or paths["claims"]
    claims.to_parquet(out)
    log.info("wrote %d claims to %s", len(claims), out)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from .eval.clustered import block_bootstrap, season_split
    from .models.price import (
        FEATURE_COLUMNS,
        OPPORTUNITY_COLUMNS,
        ComparablesPrice,
        HurdlePrice,
        build_auction_table,
        simulate_policy,
    )

    paths = _paths(args.data)
    source = args.claims or paths["claims"]
    if not Path(source).exists():
        log.error("no claim table at %s; run `iie claims` first", source)
        return 1

    auctions = build_auction_table(pd.read_parquet(source))
    train, test = season_split(auctions, _seasons(args.train), _seasons(args.test))
    log.info(
        "train %d auctions (%d franchises) | test %d (%d)",
        len(train),
        train.franchise_id.nunique(),
        len(test),
        test.franchise_id.nunique(),
    )

    target = train.price_to_beat
    baseline = ComparablesPrice().fit(train, target)
    base_bids = baseline.predict(test)
    base = simulate_policy(test, base_bids)

    def matched_arm(features: tuple[str, ...]) -> tuple[float, dict, np.ndarray]:
        """Fit a hurdle on ``features`` and find its bid at the baseline's win rate.

        Policies are only comparable at a matched win rate: bidding more always
        wins more and costs more, so one axis has to be held fixed. Matching on
        *confidence* instead would compare two different operating points.
        """
        schedule = HurdlePrice(features=features).fit(train, target).bid_schedule(test)
        best = None
        for confidence in np.arange(0.60, 0.99, 0.005):
            result = simulate_policy(test, schedule.at(confidence))
            gap = abs(result["contested_win_rate"] - base["contested_win_rate"])
            if best is None or gap < best[0]:
                best = (gap, confidence, result)
        _, confidence, result = best
        return confidence, result, schedule.at(confidence)

    without = tuple(c for c in FEATURE_COLUMNS if c not in OPPORTUNITY_COLUMNS)
    arms = {"hurdle (box score only)": without} if args.ablate else {}
    arms["hurdle (full)"] = FEATURE_COLUMNS

    fitted = {name: matched_arm(features) for name, features in arms.items()}

    print(f"\n{'policy':28s} {'contested win%':>15s} {'mean bid':>10s} {'vs baseline':>12s}")
    print(
        f"{'comparables (positional)':28s} {100 * base['contested_win_rate']:15.1f} "
        f"{base['mean_bid']:10.4f} {'--':>12s}"
    )
    for name, (confidence, result, _) in fitted.items():
        print(
            f"{f'{name} q={confidence:.3f}':28s} {100 * result['contested_win_rate']:15.1f} "
            f"{result['mean_bid']:10.4f} "
            f"{100 * (1 - result['mean_bid'] / base['mean_bid']):11.1f}%"
        )

    scored = test.copy()
    scored["baseline"] = base_bids
    for name, (_, _, bids) in fitted.items():
        scored[name] = bids

    def spend_gap(left: str, right: str):
        def statistic(frame: pd.DataFrame) -> float:
            return (
                simulate_policy(frame, frame[left].to_numpy())["mean_bid"]
                - simulate_policy(frame, frame[right].to_numpy())["mean_bid"]
            )

        return block_bootstrap(scored, statistic, draws=args.draws)

    print(
        f"\nbudget saved per auction vs baseline (franchise-clustered): "
        f"{spend_gap('baseline', 'hurdle (full)')}"
    )
    if args.ablate:
        # The Phase 1 question: do the opportunity features buy anything the box
        # score did not already say? Positive means the full model bids less for
        # the same wins. The interval is what decides it, not the point estimate.
        print(
            f"opportunity features, extra saving per auction: "
            f"{spend_gap('hurdle (box score only)', 'hurdle (full)')}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .crawl.frontier import Frontier
    from .crawl.store import SCHEMAS, read_table

    paths = _paths(args.data)
    if paths["frontier"].exists():
        with Frontier(paths["frontier"]) as frontier:
            counts = frontier.counts()
            print("frontier:", {k: v for k, v in counts.items() if v})
            print("rejections:", frontier.rejection_reasons())
    else:
        print("frontier: not created")

    print("\ncorpus:")
    for name in SCHEMAS:
        print(f"  {name:22s} {len(read_table(paths['corpus'], name)):8d} rows")

    claims = args.claims or paths["claims"]
    print(f"\nclaims: {'present' if Path(claims).exists() else 'not built'} ({claims})")
    return 0


# --- wiring -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iie", description="Fantasy waiver decision engine.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_ROOT, help="data root")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl = subparsers.add_parser("crawl", help="extend the league corpus")
    crawl.add_argument("--seed", help="Sleeper username to seed from (omit to resume)")
    crawl.add_argument("--target", type=int, default=500, help="leagues to fetch")
    crawl.add_argument("--seasons", default=",".join(DEFAULT_SEASONS))
    crawl.add_argument("--rate", type=float, default=12.0, help="requests per second")
    crawl.add_argument("--max-requests", type=int, default=60_000)
    crawl.set_defaults(func=cmd_crawl)

    claims = subparsers.add_parser("claims", help="build the claim table from the corpus")
    claims.add_argument("--out", type=Path)
    claims.add_argument("--horizon", type=int, default=6, help="outcome weeks after a claim")
    claims.add_argument("--no-defense", action="store_true", help="exclude team defenses")
    claims.set_defaults(func=cmd_claims)

    evaluate = subparsers.add_parser("evaluate", help="train and score the price model")
    evaluate.add_argument("--claims", type=Path)
    evaluate.add_argument("--train", default="2024")
    evaluate.add_argument("--test", default="2025")
    evaluate.add_argument("--draws", type=int, default=300, help="bootstrap resamples")
    evaluate.add_argument(
        "--ablate",
        action="store_true",
        help="also fit without the opportunity features, to price their contribution",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    status = subparsers.add_parser("status", help="show corpus and frontier state")
    status.add_argument("--claims", type=Path)
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
