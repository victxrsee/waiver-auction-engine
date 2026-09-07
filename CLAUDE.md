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
5. **`data/` is never committed.** The corpus holds 4,267 real users' `user_id`s. It is
   gitignored; verify with `git ls-tree -r --name-only origin/main | grep ^data/`
   before any push that touches ignore rules. Identifying fields (display names,
   avatars, team and league names) are dropped at ingestion.
6. **Bidding policies are compared at a matched win rate**, never at matched
   confidence and never on MAE. Bidding more always wins more and costs more, so one
   axis must be pinned. MAE rewards never bidding, because the target is 78% zeros.

## Evaluation protocol

Train on **2025 weeks 1–9**, test on **weeks 10+**. Strictly past→future, and closer
to deployment than a season split: you bid in week 10 knowing weeks 1–9, never
knowing only last season.

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
models/     price.py (hurdle + baseline + policy simulation)
eval/       clustered.py (block bootstrap, group k-fold, franchise chaining)
cli.py      crawl | claims | evaluate | audit | status
```

## Commands

```bash
.venv/bin/python -m pytest tests/ -q     # 202 tests, no network needed
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

- **V is not built**, so the engine prices claims but cannot rank them.
- **Team defenses (12% of claims) are unpriceable** — they key on team abbreviation
  and need team-level stats, not a player join.
- **Prices are conditional on a contest occurring**; a deployed engine faces the whole
  free-agent pool, most of whom nobody claims.
- **Sampling is snowballed** from one seed and oversamples high-degree users.
