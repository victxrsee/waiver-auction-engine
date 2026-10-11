# iie — waiver auction engine

Prices FAAB fantasy-football waiver claims as **first-price sealed-bid auctions**,
learned from a crawled corpus of real Sleeper leagues.

## The two quantities, and why they are separate

- **P — price.** What it takes to outbid everyone else. *This is what is built.*
- **V — value.** What the player is worth to your lineup. *Not built yet.*

You bid to beat P, not to pay V. Any suggestion that conflates them is wrong.

**The prediction target is the highest *losing* bid** (`price_to_beat`), not the
winning bid. In an uncontested auction the winner may bid 40 when 0 would have won,
so a winning bid measures enthusiasm, not price. This is observable only because
Sleeper reports *failed* claims with their amounts attached.

The target is **78% zeros**, which is why the model is a hurdle: a classifier for
whether anyone else bids, then quantile regressions for how much given that they do.

**The opportunity features are built but deliberately not in the price model.**
Depth-chart movement, injury status and cross-league demand live in
`OPPORTUNITY_COLUMNS`, not `FEATURE_COLUMNS`: adding them costs
`-0.0040 [-0.0042, -0.0039]` of budget per auction. They are carried through the
claim table and audited, because they are the mispricing thesis and are expected in
V. `iie evaluate --ablate` adds them back so the verdict stays re-checkable. Do not
quietly move them into `FEATURE_COLUMNS` — a test pins this.

## Non-negotiable invariants

Violating any of these has caused a real, silent defect in this repo before.

1. **As-of discipline.** Features join at `asof_week` — the last week whose games had
   finished when the claim was made — and *never* later. Sleeper's transaction `week`
   is a Wed→Wed window straddling games; joining on it leaks in 47% of claims.
   `features/asof.py:check_no_lookahead` guards this and must compare the *panel*
   side of a merge, not a column to itself.
2. **Normalise each season before `pd.concat`, never after.** nflverse changes schemas
   between seasons. A filter applied to the concatenated frame silently means "drop
   every season lacking this column" — it has erased a whole training season twice
   (`depth_charts` 2024, then `injuries` 2024).
3. **No feature may be a raw count over our own corpus.** Such a feature measures how
   many leagues the crawler reached, not the player. `demand_prior` averaged 8.6 in a
   174-league season and 278 in an 1,823-league one; it is now `demand_share`, a
   fraction. Normalise by something that scales with the corpus.
4. **Intervals are franchise-clustered.** Naive standard errors are **2.6× too narrow**
   here — one manager appears in many leagues. Use `eval/clustered.py:block_bootstrap`;
   franchises are chained across seasons via `previous_league_id` union-find.
   **The statistic must not group by the cluster column.** Resampling draws whole
   clusters, so one drawn three times appears as three copies of its rows; regrouping
   on the cluster id merges them into a single group at triple the total. That
   produced an interval not containing its own estimate (+704 against [+1071, +1155]).
   `sum() / nunique()` is wrong the same way. Pre-aggregate to one row per cluster.
5. **`data/` is never committed.** The corpus holds 4,267 real users' `user_id`s. It is
   gitignored; verify with `git ls-tree -r --name-only origin/main | grep ^data/`
   before any push that touches ignore rules. Identifying fields (display names,
   avatars, team and league names) are dropped at ingestion.
6. **Bidding policies are compared at a matched win rate**, never at matched
   confidence and never on MAE. Bidding more always wins more and costs more, so one
   axis must be pinned. MAE rewards never bidding, because the target is 78% zeros.
7. **A claim's value is its gain over replacement, never its raw projection.** A
   manager who claims a 40-point player gains 40 minus what they would have started
   instead. Treating the alternative as zero made *all 50,337* of one week's
   candidates clear a positive surplus -- a statement about units, not a discovery.
   Replacement is the Nth-best *available* player at that position (rank, not
   quantile: the available pool's 90th percentile is 0.0 points for most
   position-weeks, so a quantile barely subtracts anything).
8. **Value models are judged on top-k realised points, not on rank correlation.**
   A manager claims a handful of players, they do not order the wire. Most of a
   waiver pool returns nothing, so a predictor that leaves the undifferentiated mass
   tied is scored on *matched* ties and rates well on Spearman while picking badly:
   trailing production left 91% of the 2025 pool in one tie, scored 0.780 against
   the fitted model's 0.656, and its top-5 claims returned 23.5 points against 27.5.
   Always print the oracle and the pool average beside a top-k figure, or the number
   means nothing.

## Evaluation protocol

Train on **2025 weeks 1–9**, test on **weeks 10+**. Strictly past→future, and closer
to deployment than a season split: you bid in week 10 knowing weeks 1–9, never
knowing only last season. `iie evaluate` runs this by default.

**Exclude owner `479157902301458432` from any headline.** That one account holds 877
of 3,256 leagues and they are a different market (27% of auctions contested against
16%, three times the price, a flat $100 budget). Pooled, the positional baseline must
average both regimes and gets both wrong, inflating the model's apparent margin past
what it achieves in either population alone. `iie evaluate --exclude-owner <id>`.

**Policies are read off the frontier, never at a single matched grid point.** That
curve steps: two adjacent confidences once moved the headline from 42.9% to 22.2%.
`policy_frontier` plus `spend_at_win_rate` interpolate to a common win rate and
report the bracket; a wide bracket means the figure is fragile.

Do **not** use train-2024 / test-2025 as evidence about a feature. The corpus is 85%
one season, so that split fits ~22k auctions and scores ~369k across a season where
nflverse changed the depth-chart schema outright. Measured under it, features that
genuinely help appear to hurt.

## Layout

```
sources/    cache.py (HTTP + TTL), sleeper.py, nflverse.py, playerids.py (3-tier bridge)
crawl/      crawler.py (2-phase walk), frontier.py (SQLite state), store.py (Parquet)
features/   asof.py (week boundary), panel.py (player-week grid),
            opportunity.py (depth/injury/demand), claims.py (assembles the claim table)
models/     hurdle.py (shared classifier+regressor+quantiles), price.py (P: bid
            schedule, policy simulation), value.py (V: components, top-k),
            decide.py (exchange rate, replacement level, surplus-maximising bids)
eval/       clustered.py (block bootstrap, group k-fold, franchise chaining)
cli.py      crawl | claims | evaluate | value | decide | backtest | audit | status
```

## Commands

```bash
.venv/bin/python -m pytest tests/ -q     # 290 tests, no network needed
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests

.venv/bin/iie status                     # corpus + frontier state
.venv/bin/iie crawl --target N           # extend corpus (--target is per-run, not cumulative)
.venv/bin/iie claims                     # rebuild the claim table (~25s)
.venv/bin/iie audit                      # feature distribution checks — run after every claims rebuild
.venv/bin/iie evaluate --ablate          # both arms + clustered interval (~2 min)
```

Always use `.venv/bin/`, not bare `python`.

## Style

Comments explain *why*, especially where a line encodes a defect that has already
happened — those comments are load-bearing and should not be trimmed. Match the
density of the surrounding file. Tests are named as sentences describing the
behaviour, and their docstrings say what breaks if the behaviour regresses.

## Known gaps

- **The decision layer is a first cut.** `iie decide` produces a ranked claim list
  with bids, but the exchange rate is market-implied (so it inherits the market's
  overpayment as its reference level) and replacement level is calibrated only
  roughly -- it moves mean claim value from 55.3 to 42.4 points, which still leaves
  nearly every real claim clearing a positive surplus. There is also no budget
  constraint across a season: claims are priced one week at a time with no cost
  charged for spending early.
- **V beats its baselines on top-k but loses on rank correlation**, because the
  baseline wins that metric through ties (see invariant 7). Taken at face value the
  Spearman column says the model is not worth its complexity; it is, on the metric
  that matches the decision.
- **`depth_rank` is not comparable across schemas** — legacy `depth_team` has three
  levels, modern `pos_rank` runs 1–15.
- **`_regular_season` prefers `season_type` on the column *existing*, not being
  *populated*** — the exact shape of the bug it was written to fix.
- **Team defenses (12% of claims) are unpriceable** — they key on team abbreviation
  and need team-level stats, not a player join.
- **Prices are conditional on a contest occurring**; a deployed engine faces the whole
  free-agent pool, most of whom nobody claims.
- **Sampling is snowballed** from one seed and oversamples high-degree users.
