# waiver-auction-engine

Pricing FAAB waiver claims in fantasy football as what they actually are: **first-price sealed-bid auctions**.

The engine learns what it costs to win a waiver claim from a corpus of 500 real
league-seasons, then recommends a bid. It is a working price model with a
reproducible pipeline behind it; the value half — whether a player is *worth* the
price — is not built yet.

---

## The framing

A FAAB waiver is a sealed-bid auction: every manager submits one hidden bid, the
highest wins, and the winner pays their own bid. Two quantities matter, and they
are not the same:

- **P — the price.** What it takes to outbid everyone else.
- **V — the value.** What the player is worth to your lineup.

You should bid to beat **P**, not to pay **V**. This repository implements P.

The non-obvious consequence is that **the winning bid is the wrong prediction
target.** In an uncontested auction someone may bid 40 when 0 would have won, so a
winning bid measures the winner's enthusiasm rather than the market price. The
right target is the **highest losing bid** — the amount that actually had to be
exceeded.

That is only observable because Sleeper reports *failed* waiver claims with their
bid amounts attached. Most auction data is censored to the winner; here roughly
half the observations are losing bids, so the competing distribution is visible
directly.

## Results

Trained on 2024, tested on 2025, with franchise-clustered intervals throughout.

Bidding policies cannot be compared head-to-head — bidding more always wins more
and costs more — so they are compared on a **cost-effectiveness frontier**, at a
matched win rate:

| policy | contested win rate | mean bid (share of budget) |
| --- | --- | --- |
| comparables baseline (positional mean) | 26.7% | 0.0246 |
| hurdle model | 26.9% | **0.0191** |

**22.5% less budget spent for the same win rate.** Franchise-clustered saving of
`+0.0055 [+0.0040, +0.0067]` of budget per auction. Supporting signal: Spearman
0.305 `[0.288, 0.322]`, contested AUC 0.741.

The baseline is self-built (mean price by position and week bucket) — not a
commercial product and not human managers.

## Three problems worth reading the code for

**Entity resolution across disjoint ID spaces** — [`sources/playerids.py`](src/iie/sources/playerids.py)

Sleeper and nflverse share no key, and the obvious bridge is a dead end: Sleeper
populates `sportradar_id` on 98.6% of players, but nflverse's player table has no
such column. Three tiers — Sleeper's own `gsis_id` (20.4%), a community crosswalk
(91.3%), then normalized name and position (97%). The name tier **refuses
ambiguous matches** rather than guessing, because a wrong join silently
misattributes one player's production to another. Precision was measured without
labels by stripping known IDs and checking whether the matcher reproduced them:
99.7% on fantasy positions.

**Point-in-time correctness** — [`features/asof.py`](src/iie/features/asof.py)

Sleeper labels each transaction with a fantasy week, but that week is a
Wednesday-to-Wednesday window that *straddles* the games it names — week 5 of 2025
runs Oct 1–8 while the games run Oct 2–6. Claims land throughout it, so two
transactions sharing a label can rest on completely different information. A
week-label join would have leaked post-game results into 47% of observations. The
as-of boundary is resolved from each transaction's timestamp against the real NFL
schedule instead, with the uncertainty deliberately biased toward knowing *less*.

**Cluster-robust inference** — [`eval/clustered.py`](src/iie/eval/clustered.py)

Claims within a league share managers, scoring rules and a player pool, so they
are not independent observations. Clusters are *franchises*, not league-seasons —
a league chains across years through `previous_league_id`, resolved with
union-find, collapsing 497 leagues into 428 clusters. Treating rows as independent
made standard errors **2.6x too narrow**.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

iie status                          # corpus and crawl state
iie crawl --seed <sleeper-username> # build/extend the corpus (~26 min per 500 leagues)
iie claims                          # build the modeling table (~7s)
iie evaluate                        # train, test, report with clustered intervals (~20s)
```

Everything is cached on disk under `data/` (gitignored). The HTTP cache makes
re-runs effectively free, and the crawler resumes from SQLite frontier state after
an interruption.

## Architecture

```
sources/    Sleeper client, nflverse ingestion, player ID bridge
crawl/      SQLite frontier, two-phase crawler, Parquet corpus store
features/   as-of boundary, player-week panel, claim table
eval/       franchise clustering, block bootstrap, leak-proof splits
models/     hurdle price model, bidding policy simulation
```

The crawler is two-phase because the league graph explodes and roughly half of
what it finds is unusable. The endpoint listing a user's leagues returns complete
league objects, so **screening costs zero extra requests** and only survivors pay
the ~18-request deep fetch. Fetch-then-filter over the same graph would have cost
~75,000 requests; the crawl spent 9,144.

## Data

500 league-seasons → 497 leagues → **428 franchises**, 2024–25, FAAB redraft only.
212,767 transactions → **142,102 bid observations**.

Only public, unauthenticated Sleeper endpoints are used. Identifying fields —
display names, avatars, team names, league names — are **dropped at ingestion**;
only the `user_id` needed as a cross-league join key is retained. No crawled data
is committed to this repository.

## Status and limitations

Honest about what this is: a personal project with a working price model, not a
deployed product.

- **The value model is not built.** The engine prices claims but cannot yet say
  which are worth buying.
- **Training is thin** — 2024 only, 174 franchises, tested on a different season.
- **Prices are conditional on a contest occurring.** A deployed engine faces the
  whole free-agent pool, most of whom nobody claims; that selection gap is open.
- **Team defenses are unpriceable** — 12% of waiver activity. They key on team
  abbreviation and need team-level stats rather than a player join.
- **Sampling is snowballed from one seed**, so it oversamples high-degree users.
  Measured: one manager's network is 30% of observations, which moves the headline
  correlation by 0.005. Fine for modeling, not for population-level claims.
- **The price target is noisy.** A quarter of contested auctions are exact ties
  broken by waiver priority, and in ~10% the winner bid less than a loser —
  almost certainly bids that could not be honoured for lack of budget.

## Tests

168 tests, no network access required.

```bash
.venv/bin/python -m pytest tests/ -q
```

They cover invariants (features never drawn from after the as-of boundary) and
**cost properties** — asserting that a rejected league triggers exactly one roster
call and zero transaction calls, which is the economic claim the crawler is built
on.
