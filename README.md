# waiver-auction-engine

Pricing FAAB waiver claims in fantasy football as what they actually are: **first-price sealed-bid auctions**.

The engine learns what it costs to win a waiver claim from a corpus of 2,280 real
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

### The engine against the managers who were there

Fitted on 2025 weeks 1-9 and replayed over weeks 10+, on 203,495 claims across 2,142
franchises. Policies cannot be compared at a single point -- bidding more always wins
more and costs more -- so the engine is placed on the managers' own spend first:

| | budget spent | points acquired |
| --- | --- | --- |
| managers (actual) | 1.476 | 1,785 |
| **engine, at the same spend** | **1.476** | **2,447** |

**+662 points per franchise, +37%**, franchise-clustered at `+704 [+679, +727]` at the
nearest swept operating point. Budget is in whole-season units, so 1.476 means a
franchise spent about 1.5 times its starting FAAB across the back half (budgets
replenish in some leagues). Reproduce with:

```bash
iie backtest --exclude-owner 479157902301458432
```

**Two limitations are structural, and the number means nothing without them.** The
candidate set is *claims somebody actually made*, so the engine never sees the player
nobody wanted and should have — that needs free-agent pool reconstruction, which is
not built. And the rest of the market is held fixed, which is the right counterfactual
for one franchise of twelve and the wrong one for a policy every manager adopted,
because then the prices themselves would move.

**Not the headline: efficiency.** The engine reaches 1,310 points per unit of budget
against the managers' 1,210, and an earlier version of this backtest reported *262x*
on that measure. That figure is meaningless rather than impressive: it is a ratio
whose denominator the policy chooses, and bidding zero everywhere wins the 78.6% of
claims nobody contests for nothing at all. Compare at matched spend.

### The price model

| policy | mean bid (share of budget) | vs baseline |
| --- | --- | --- |
| comparables baseline (positional mean) | 0.0149 | — |
| hurdle price model | **0.0078** | **47.5% less** |

Trained on 2025 weeks 1-9, tested on weeks 10+, at a matched contested win rate.
Franchise-clustered saving `+0.0072 [+0.0068, +0.0074]` of budget per auction.

### The value model

Mean realised points of each week's top five claims from the waiver pool, trained
2023-24 and tested on 2025:

| model | top-5 points |
| --- | --- |
| positional mean | 7.8 |
| trailing production | 23.5 |
| **value model** | **27.5** |
| perfect foresight | 63.9 |
| pool average | 1.5 |

Judged on top-k and not on rank correlation, for a reason worth knowing: trailing
production leaves **91%** of the waiver pool tied at zero and 88% of the pool really
does return zero, so matched ties score it 0.780 against the model's 0.656 while it
picks visibly worse players. A manager claims a handful of players; they do not order
the wire.

The baselines are self-built — not a commercial product and not human managers,
except in the backtest above, which is against the managers directly.

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
~75,000 requests; the crawl spent ~25,000.

## Data

2,280 league-seasons → 1,997 leagues → **1,928 franchises**, 2024–25, FAAB redraft
only. 882,767 transactions → **744,026 bid observations**.

Only public, unauthenticated Sleeper endpoints are used. Identifying fields —
display names, avatars, team names, league names — are **dropped at ingestion**;
only the `user_id` needed as a cross-league join key is retained. No crawled data
is committed to this repository.

## Status and limitations

Honest about what this is: a personal project with a working price model, not a
deployed product.

- **The value model is not built.** The engine prices claims but cannot yet say
  which are worth buying.
- **The corpus is 91% one season** — 1,824 leagues in 2025 against 175 in 2024, and
  no 2024 leagues remain unfetched. Cross-season generalization is therefore
  untested, and the 2024 depth-chart schema is not directly comparable to 2025's.
- **Prices are conditional on a contest occurring.** A deployed engine faces the
  whole free-agent pool, most of whom nobody claims; that selection gap is open.
- **Team defenses are unpriceable** — 12% of waiver activity. They key on team
  abbreviation and need team-level stats rather than a player join.
- **Sampling is snowballed from one seed**, so it oversamples high-degree users.
  Measured: one manager's network is 30% of observations, which moves the headline
  correlation by 0.005. Fine for modeling, not for population-level claims. The
  crawler made this worse than it needed to be — it only expanded to new users when
  its fetch queue ran dry, which never happened, so 2,280 leagues came from ~140
  people. It now expands on a fixed cadence.
- **The price target is noisy.** A quarter of contested auctions are exact ties
  broken by waiver priority, and in ~10% the winner bid less than a loser —
  almost certainly bids that could not be honoured for lack of budget.

## Tests

290 tests, no network access required.

```bash
.venv/bin/python -m pytest tests/ -q
```

They cover invariants (features never drawn from after the as-of boundary) and
**cost properties** — asserting that a rejected league triggers exactly one roster
call and zero transaction calls, which is the economic claim the crawler is built
on.
