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

Trained on 2025 weeks 1–9 and tested on weeks 10+ — strictly past to future, with
franchise-clustered intervals throughout.

> **The headline is withdrawn pending re-derivation (2026-09-07).** Three independent
> defects were found in one day, each sufficient on its own to invalidate the number
> that stood here. They are documented below because how they were found is more
> useful than the figure they replaced.

**1. The depth-chart features read the future.** Each daily snapshot was labelled with
the week that had *completed* when it was published, and a claim then joined the last
snapshot carrying its `asof_week`. That label names a ~7-day bucket running from the
end of week *W*'s games to the end of week *W+1*'s, and the claim sits at the near end
of it. Measured across 707,398 joined rows: the snapshot a claim reads is published a
median **5.0 days after that claim**, and is later than the claim **97.7%** of the
time. Correcting it as-of drops `spearman(depth_rank, bid_pct)` from −0.061 to −0.025.
The no-lookahead guard cannot see this — it compares week labels, which match by
construction.

**2. The corpus is 27% one account.** A single Sleeper user belongs to 877 of 3,256
leagues, and those leagues are structurally unlike the rest: 27% of auctions contested
against 16%, mean price 0.038 against 0.014, and a standardised $100 budget against a
median of $131. They are public mass-entry leagues, not leagues among friends. Any
statistic over the pooled corpus is a blend of two markets in a ratio set by one
crawler decision.

**3. The comparison metric is unstable at the operating point.** Policies are compared
at a matched contested win rate, swept over a confidence grid. That curve is not
smooth: between two adjacent grid points the mean bid moves 36% and the reported
spend reduction moves from 42.9% to 22.2%, and one step further the win rate jumps
8 points at once. Two runs over nearly identical data matched on opposite sides of a
step and reported 17.3% and 46.9% for the same policy.

The pooled corpus also produces a spend reduction *larger than either population
alone* — 47% pooled against 17% and 11% — which no better model can do. The
positional-mean baseline degrades when averaging two price regimes, so that gain
measures the baseline getting worse.

What survives: the pipeline, the auction framing, the target choice, and the
defect-finding apparatus that produced this list. The number does not.

**On the split.** An earlier version trained on 2024 and tested on 2025, which
looks like the more conventional choice and is in fact the weaker one here: the
corpus is 91% 2025, so that split fits 22k auctions and scores 369k, across a
season where nflverse changed the depth-chart schema outright. Measured under it,
the opportunity features appear to *hurt* — entirely an artifact of the mismatch.
Training on early weeks and predicting later ones keeps the train-on-the-past
discipline and is closer to deployment anyway: you bid in week 10 knowing weeks
1–9, never knowing only last season.

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

202 tests, no network access required.

```bash
.venv/bin/python -m pytest tests/ -q
```

They cover invariants (features never drawn from after the as-of boundary) and
**cost properties** — asserting that a rejected league triggers exactly one roster
call and zero transaction calls, which is the economic claim the crawler is built
on.
