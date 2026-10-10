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
        # An in-progress season has no complete leagues, so the completeness gate
        # rejects every one of them. Prices are observable the moment a waiver
        # clears, so a live season is usable for P even though its forward outcomes
        # (and therefore V) are truncated.
        require_complete=not args.include_incomplete,
        expand_every=args.expand_every,
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
    import pandas as pd

    from .eval.clustered import block_bootstrap, season_split
    from .models.price import (
        FEATURE_COLUMNS,
        OPPORTUNITY_COLUMNS,
        ComparablesPrice,
        HurdlePrice,
        build_auction_table,
        policy_frontier,
        simulate_policy,
        spend_at_win_rate,
    )

    paths = _paths(args.data)
    source = args.claims or paths["claims"]
    if not Path(source).exists():
        log.error("no claim table at %s; run `iie claims` first", source)
        return 1

    auctions = build_auction_table(pd.read_parquet(source))
    if args.exclude_owner:
        # One account holding 877 of 3,256 leagues ran a different market: 27% of
        # auctions contested against 16%, three times the price, a flat $100 budget.
        # Pooled, the positional baseline has to average both regimes and gets both
        # wrong, which inflates every model's apparent margin past what it achieves
        # in either population alone. The headline therefore excludes it, and this
        # flag is how that figure is reproduced.
        from .crawl.store import read_table

        owners = read_table(_paths(args.data)["corpus"], "rosters")[["league_id", "owner_id"]]
        drop = set(owners.loc[owners["owner_id"] == args.exclude_owner, "league_id"])
        before = len(auctions)
        auctions = auctions[~auctions["league_id"].isin(drop)]
        log.info(
            "excluded owner %s: %d leagues, %d auctions",
            args.exclude_owner,
            len(drop),
            before - len(auctions),
        )
        if not drop:
            log.warning("owner %s holds no leagues in this corpus", args.exclude_owner)
    if args.protocol == "week":
        # The protocol CLAUDE.md mandates, and the one the CLI could not previously
        # run: train on the early weeks of a season and predict its later ones. It
        # keeps train-on-the-past without spanning a season's schema change, and is
        # closer to deployment -- you bid in week 10 knowing weeks 1-9.
        one = auctions[auctions["season"] == args.season]
        train = one[one["week"] <= args.cut]
        test = one[one["week"] > args.cut]
    else:
        train, test = season_split(auctions, _seasons(args.train), _seasons(args.test))
    if train.empty or test.empty:
        log.error("empty split: check --protocol/--season/--cut against the corpus")
        return 1
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

    # The shipping model is FEATURE_COLUMNS, which no longer carries the
    # opportunity family. --ablate adds it back rather than taking it away, so the
    # finding that it hurts stays re-checkable as the corpus grows.
    arms = {"hurdle (price model)": FEATURE_COLUMNS}
    if args.ablate:
        arms["hurdle (+ opportunity)"] = FEATURE_COLUMNS + OPPORTUNITY_COLUMNS

    # Every arm is read off its own frontier at one common win rate -- the
    # baseline's. Snapping each arm to its nearest grid confidence instead landed
    # them at *different* win rates (27.37% against 27.90%) and then differenced
    # them, charging the gap in operating point to the model.
    goal = base["contested_win_rate"]
    fitted = {}
    for name, features in arms.items():
        schedule = HurdlePrice(features=features).fit(train, target).bid_schedule(test)
        curve = policy_frontier(test, schedule)
        fitted[name] = (curve, spend_at_win_rate(curve, goal), schedule)

    print(f"\nall policies read at a contested win rate of {100 * goal:.2f}%\n")
    print(f"{'policy':28s} {'mean bid':>10s} {'vs baseline':>12s} {'bracket':>10s}")
    print(f"{'comparables (positional)':28s} {base['mean_bid']:10.4f} {'--':>12s} {'--':>10s}")
    for name, (_, point, _) in fitted.items():
        flag = " *" if point["step"] > 0.1 * base["mean_bid"] else ""
        print(
            f"{name:28s} {point['spend']:10.4f} "
            f"{100 * (1 - point['spend'] / base['mean_bid']):11.1f}% "
            f"{point['step']:10.4f}{flag}"
        )

    # The bracket is how far the spend moves between the two grid points the
    # interpolation sits between. A wide one means the figure is fragile no matter
    # how it was derived, so it is printed rather than hidden behind the estimate.
    if any(p["step"] > 0.1 * base["mean_bid"] for _, p, _ in fitted.values()):
        print(
            "\n* the frontier steps by more than 10% of the baseline bid here; "
            "treat the figure as indicative and read the frontier with --frontier"
        )
    if any(p.get("extrapolated") for _, p, _ in fitted.values()):
        print("\nWARNING: a policy never reaches that win rate; its spend is extrapolated")

    if args.frontier:
        for name, (curve, _, _) in fitted.items():
            print(f"\n{name} frontier:")
            print(f"{'q':>7s} {'contested win%':>15s} {'mean bid':>10s}")
            for row in curve.itertuples():
                print(
                    f"{row.confidence:7.3f} {100 * row.contested_win_rate:15.2f} "
                    f"{row.mean_bid:10.4f}"
                )

    # Clustered intervals are taken on bids realised at the common win rate, so the
    # bootstrap resamples the same policies the table above reports.
    scored = test.copy()
    scored["baseline"] = base_bids
    for name, (curve, point, schedule) in fitted.items():
        nearest = curve.iloc[(curve["contested_win_rate"] - goal).abs().argmin()]
        scored[name] = schedule.at(float(nearest["confidence"]))

    def spend_gap(left: str, right: str):
        def statistic(frame: pd.DataFrame) -> float:
            return (
                simulate_policy(frame, frame[left].to_numpy())["mean_bid"]
                - simulate_policy(frame, frame[right].to_numpy())["mean_bid"]
            )

        return block_bootstrap(scored, statistic, draws=args.draws)

    print(
        f"\nbudget saved per auction vs baseline (franchise-clustered): "
        f"{spend_gap('baseline', 'hurdle (price model)')}"
    )
    if args.ablate:
        # Does adding the opportunity family back buy anything? Positive would mean
        # it bids less for the same wins. Measured 2026-09-07 it is negative, which
        # is why it is not in the model. The interval decides, not the point.
        print(
            f"opportunity features, extra saving per auction: "
            f"{spend_gap('hurdle (price model)', 'hurdle (+ opportunity)')}"
        )
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    """Train and score the value model on the nflverse player panel.

    Deliberately independent of the crawled corpus: outcomes exist for every
    player-week whether or not anyone bid, so training here avoids conditioning on
    somebody having already wanted the player -- which is the selection the engine
    exists to second-guess.
    """
    from .features.panel import add_forward, add_trailing, build_player_week, on_full_week_grid
    from .models.value import PositionalValue, ValueModel, add_horizon, evaluate_value
    from .sources.nflverse import NflverseCache

    cache = NflverseCache(_paths(args.data)["nflverse"])
    seasons = [int(s) for s in _seasons(args.seasons)]
    panel = build_player_week(cache, seasons)
    if panel.empty:
        log.error("no player-week data for %s", seasons)
        return 1
    panel = add_horizon(add_forward(add_trailing(on_full_week_grid(panel)), args.horizon))

    train = panel[panel["season"].isin([int(s) for s in _seasons(args.train)])]
    test = panel[panel["season"].isin([int(s) for s in _seasons(args.test)])]
    if train.empty or test.empty:
        log.error("empty split: --train %s --test %s against %s", args.train, args.test, seasons)
        return 1
    log.info("train %d player-weeks | test %d", len(train), len(test))

    truth = (
        test[f"fantasy_points_next{args.horizon}"]
        + args.reception_points * test[f"receptions_next{args.horizon}"]
    )
    arms = {
        "positional baseline": PositionalValue(),
        "value model": ValueModel(horizon=args.horizon),
    }
    print(f"\n{'model':22s} {'MAE':>8s} {'RMSE':>8s} {'Spearman':>10s} {'within-week':>12s}")
    for name, model in arms.items():
        predicted = model.fit(train).predict_points(test, args.reception_points)
        got = evaluate_value(predicted, truth, groups=test["week"])
        print(
            f"{name:22s} {got['mae']:8.2f} {got['rmse']:8.2f} "
            f"{got['spearman']:10.3f} {got['spearman_within_group']:12.3f}"
        )
    # Within-week ranking is the number that matters: the decision is which claim to
    # make from the set available this week, so order inside a week beats level.
    print("\nwithin-week Spearman is the decision-relevant figure; pooled is flattered")
    print("by contrasts between early and late weeks that no manager ever faces.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    import pandas as pd

    from .eval.audit import audit_features, format_report
    from .models.price import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, OPPORTUNITY_COLUMNS

    paths = _paths(args.data)
    source = args.claims or paths["claims"]
    if not Path(source).exists():
        log.error("no claim table at %s; run `iie claims` first", source)
        return 1

    claims = pd.read_parquet(source)
    # Auditing a feature across the boundary it defines reports the boundary.
    ignore = ("week", "asof_week") if args.split == "week" else ()
    if args.split == "week":
        # The evaluation protocol splits 2025 on week, so that is the boundary the
        # model must generalise over; auditing across seasons would miss a shift
        # that happens inside one.
        claims = claims[claims["season"] == args.season].copy()
        claims["_split"] = ["weeks<=9" if w <= 9 else "weeks>9" for w in claims["week"]]
        split_col = "_split"
    else:
        split_col = "season"

    summary, findings = audit_features(
        claims,
        # Opportunity columns are audited although the price model does not use
        # them: they feed the mispricing analysis and are expected in the value
        # model, so a join defect in one still matters.
        FEATURE_COLUMNS + OPPORTUNITY_COLUMNS + CATEGORICAL_COLUMNS,
        split_col=split_col,
        categorical=CATEGORICAL_COLUMNS,
        ignore=ignore,
    )
    print(f"\n{len(claims)} claims, split by {args.split}\n")
    print(format_report(summary, findings, split_col))
    # Nonzero exit so this can gate a rebuild in a script or a hook.
    return 1 if findings else 0


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
    crawl.add_argument(
        "--include-incomplete",
        action="store_true",
        help="accept leagues whose season is still running (required for a live season)",
    )
    crawl.add_argument(
        "--expand-every",
        type=int,
        default=8,
        help="expand a new user every N fetches; lower favours discovery over depth",
    )
    crawl.set_defaults(func=cmd_crawl)

    claims = subparsers.add_parser("claims", help="build the claim table from the corpus")
    claims.add_argument("--out", type=Path)
    claims.add_argument("--horizon", type=int, default=6, help="outcome weeks after a claim")
    claims.add_argument("--no-defense", action="store_true", help="exclude team defenses")
    claims.set_defaults(func=cmd_claims)

    evaluate = subparsers.add_parser("evaluate", help="train and score the price model")
    evaluate.add_argument("--claims", type=Path)
    evaluate.add_argument(
        "--exclude-owner",
        help="drop every league containing this Sleeper owner_id (see cmd_evaluate)",
    )
    evaluate.add_argument(
        "--protocol",
        choices=("week", "season"),
        default="week",
        help="'week' trains on early weeks of one season (the mandated protocol)",
    )
    evaluate.add_argument("--season", default="2025", help="season to split when --protocol=week")
    evaluate.add_argument("--cut", type=int, default=9, help="last training week")
    evaluate.add_argument("--train", default="2024")
    evaluate.add_argument("--test", default="2025")
    evaluate.add_argument("--draws", type=int, default=300, help="bootstrap resamples")
    evaluate.add_argument(
        "--frontier", action="store_true", help="print the full win-rate/spend curve per arm"
    )
    evaluate.add_argument(
        "--ablate",
        action="store_true",
        help="also fit without the opportunity features, to price their contribution",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    value = subparsers.add_parser("value", help="train and score the value model V")
    value.add_argument("--seasons", default="2023,2024,2025", help="panel seasons to build")
    value.add_argument("--train", default="2023,2024")
    value.add_argument("--test", default="2025")
    value.add_argument("--horizon", type=int, default=6, help="outcome weeks after each row")
    value.add_argument(
        "--reception-points", type=float, default=0.5, help="league scoring to report in"
    )
    value.set_defaults(func=cmd_value)

    audit = subparsers.add_parser("audit", help="check features for silent join defects")
    audit.add_argument("--claims", type=Path)
    audit.add_argument(
        "--split",
        choices=("season", "week"),
        default="week",
        help="boundary to audit across; 'week' matches the evaluation protocol",
    )
    audit.add_argument("--season", default="2025", help="season to split on when --split=week")
    audit.set_defaults(func=cmd_audit)

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
