# Tennis Polymarket Betting Bot

## What This Is

Automated tennis value betting system on Polymarket. Compares AI-generated match win probabilities against Polymarket implied odds. Bets when the gap exceeds a threshold (currently 5%, to be revisited with real data).

## Why This Should Work

- Tennis has real predictive features (Elo ratings, H2H, surface, fatigue, recent form)
- Polymarket tennis markets are thin ($2K-$25K volume per match) = less efficient pricing
- We're classifying/filtering (find mispriced matches) not predicting direction in efficient markets
- Counterparties are retail, not HFT algorithms

## Architecture

```
[Data Layer]
  Jeff Sackmann CSV + Elo ratings (historical) --> Model Training
  SportRadar / Matchstat API (live, pluggable) --> Live Player Stats
  Polymarket Gamma API (live) --> Current Implied Odds + Odds Snapshots

[Prediction Layer]
  Surface-adjusted Elo (backbone) + supplementary features --> XGBoost/LightGBM + calibration layer --> Win Probability per player

[Execution Layer]
  Compare model prob vs Polymarket price --> Bet when edge > threshold
  py-clob-client --> Order placement on Polymarket CLOB
  Early exit logic --> Sell position if odds shift favorably
  Telegram bot --> Monitoring & alerts
```

## Current Status

### Phase 0: Data Collection - COMPLETE (moved to VPS)
- **Scraper running on Hetzner VPS** (`ssh root@65.21.178.90`), every 5 minutes via cron
- GitHub Actions scraper/paper-trade workflows DISABLED (VPS handles both)
- VPS auto-commits+pushes to `dev` every 30 min via deploy scripts in `/root/tennis-pm/deploy/`
- Scrapes both ATP (series_id: 10365) and WTA (series_id: 10366) moneyline markets only
- Paginates through all results (Gamma API defaults to 20 per page, ~130 active markets typical)
- Stores events, markets, and point-in-time odds snapshots in SQLite (`data/tennis_pm.db`)
- Entry point: `python run_scraper.py` (once), `python run_scraper.py --loop` (continuous), `python run_scraper.py --backfill` (closed events)
- Logs append to `data/scraper.log`

### Phase 1: Model Training - COMPLETE
- Sackmann data downloaded to `data/sackmann/` (ATP + WTA, 2000-2024, including Challengers)
- Custom Elo computation in `model/elo.py` (overall + surface-specific, dynamic K-factor)
- 23 difference-based features in `model/features.py` (Elo, ranking, serve stats, fatigue, H2H, form)
- XGBoost + isotonic calibration selected as best model (ECE=0.009 on 2024 test set)
- All models ~65% accuracy, AUC ~0.715 — well-calibrated across full probability range
- Feature importance: elo_diff (33%), surface_elo_diff (21%), rank_diff (6%), fatigue_14d (4%)
- Models saved to `data/models/`; calibration plots in `data/evaluation/`
- Run: `python -m model.train`

### Phase 2: Paper Trading - ACTIVE
- **SportRadar integration** (`scraper/sportradar.py`): live rankings, match results, serve stats
  - API key in `.env` (SPORTRADAR_API_KEY), 30-day trial started 2026-03-17
  - Player mapping: SR IDs -> Sackmann IDs -> PM names (data/sr_player_map.json)
  - 182/237 active PM players linked through full chain
- **Elo backfilled** to March 2026 (`model/backfill_elo.py`): 21K matches from SR daily schedules
  - Cached in `data/sr_daily_cache/`, state saved in `data/elo_state.json`
  - Re-run `python -m model.backfill_elo` to update (cached days are skipped)
- **Verified player name map** (`data/player_map.json`): 296/311 PM players matched
  - Runtime uses ONLY verified mappings — no fuzzy logic, no guessing
  - Re-generate: `python -m model.name_match generate`
- **Live predictor** (`model/predictor.py`): scans PM markets with current Elo + SR live data
  - Filters: $500 min volume, skips resolved markets (price < 2% or > 98%)
  - Edge window: 5-10% (edges >10% are model error — paper trading confirmed inverse correlation)
  - Classifies each signal as **contrarian** (Elo disagrees with PM favorite) or **reinforcing**
  - Paper trade log: `data/paper_trades.json`
  - VPS runs predictor every 30 min with live SR data (form, fatigue, serve stats)
  - Run: `python -m model.predictor`

### Phase 2.5: Contrarian Strategy Validation - ACTIVE (started 2026-03-28)
- **Hypothesis:** PM tennis prices are noisy (57% prediction accuracy vs Elo's 67%). When they disagree, Elo is right 65% of the time. Betting the PM underdog when Elo says undervalued = the edge.
- **Backtest result (492 resolved markets):**
  - Contrarian bets, 5-20% gap: 61 bets, 67% win, +49% ROI
  - ATP contrarian, 5-20% gap: 49 bets, 71% win, +57% ROI
  - Reinforcing bets (same gap): 173 bets, 52% win, -4% ROI
- **Live paper trading (started 2026-03-28):**
  - Contrarian 5-20%: 3/6 wins (50%) — below backtest but tiny sample
  - Contrarian >20%: 0/3 wins — confirmed bad, now capped out
  - Need 50+ resolved contrarian trades before evaluating
  - ~13 contrarian trades pending resolution as of 2026-03-29
- **Decision point:** If contrarian signal holds at 50+ bets with >55% win rate and positive ROI, proceed to Phase 3 with small bankroll on ATP contrarian bets only
- **Still needed:**
  - Matchstat fallback before SR trial expires (~April 16)

## Key Design Decisions

These were discussed and agreed upon before building. Do not change without discussion.

1. **XGBoost/LightGBM + calibration layer** instead of sklearn RandomForest/GradientBoosting. RF is poorly calibrated out of the box. Use isotonic regression or Platt scaling on top. Logistic regression as baseline comparison.

2. **Sackmann Elo ratings as backbone feature.** Surface-specific Elo encodes ranking gap, H2H, and surface form in one number. Supplementary features (fatigue, serve stats, recent form) add on top, but Elo does the heavy lifting.

3. **Backtest against actual Polymarket odds, not bookmaker closing lines.** Bookmaker closing odds are the most efficient prices in sports — almost impossible to beat. Polymarket is far less efficient. That's where the edge is. The Phase 0 scraper is building this dataset.

4. **Fuzzy player name matching system.** Polymarket, Sackmann, and SportRadar all use different name formats. Needs a dedicated matching layer with manual override support.

5. **SQLite for all storage** (not flat CSV files). Single-file, zero-config, queryable. Schema is in `scraper/db.py`.

6. **Pluggable data sources.** SportRadar has a 30-day free trial that will expire. The data layer must allow swapping SportRadar for Matchstat or other sources without rewiring everything.

7. **Odds snapshots over time, not single readings.** The scraper captures odds every 30 minutes. Odds movement is signal — sudden moves suggest injury news, weather, or insider info.

8. **5-10% edge window** to consider a bet. Paper trading (345 trades) showed higher edges = worse performance (inverted signal). 5-10% was the only zone near breakeven. Edges >10% mean the model is wrong, not the market.

## Data Sources

### Historical Match Data (Training) - FREE
- **Jeff Sackmann/tennis_atp**: https://github.com/JeffSackmann/tennis_atp
- ATP matches from 1968, match stats from 1991, Challengers from 2008
- **Does NOT include pre-computed Elo** — we compute our own in `model/elo.py` (overall + surface-specific)
- CSV format: player rankings, H2H, serve stats, break points, all integer totals
- License: CC BY-NC-SA 4.0 (non-commercial — note: using for profit betting is a gray area)
- Also has WTA: https://github.com/JeffSackmann/tennis_wta

### Polymarket Odds - FREE, NO AUTH
- Gamma API (read-only, no auth): `https://gamma-api.polymarket.com`
- ATP series_id: `10365`, WTA series_id: `10366`
- Events: `GET /events?series_id=10365&active=true&closed=false`
- Each event has `markets` array; we filter to `sportsMarketType: "moneyline"` only
- Other market types exist (tennis_first_set_winner, tennis_first_set_totals, etc.) — we skip these
- **IMPORTANT**: `outcomePrices` comes as a JSON-encoded string, not a native list. Must `json.loads()` before indexing.
- CLOB API for order placement (needs wallet + API creds)
- Docs: https://docs.polymarket.com/quickstart/fetching-data

### Gamma API Response Structure (key fields)
```
Event level:  id, title, slug, gameId, startDate, endDate, active, closed
              eventMetadata.league (tournament name)
Market level: id, outcomes[], outcomePrices[], clobTokenIds[]
              bestBid, bestAsk, lastTradePrice, spread
              volume, liquidityClob, sportsMarketType
              umaResolutionStatus ("proposed" | "resolved")
```

### Live Tennis Data - FREE TO START (pluggable)
- **SportRadar**: 30-day free trial, same data as production, lower rate limits
  - v3 Tennis API, covers ATP + WTA + Challengers + 20K UTR events
  - Endpoints: seasons, schedules, competitor profiles, H2H, match summaries
  - Docs: https://developer.sportradar.com/tennis/reference/overview
  - WARNING: trial expires — must have fallback ready
- **Matchstat.com**: Free tier available, tennis-specific, includes pre-match odds
  - H2H stats, player profiles, historical data back to 1990
  - RapidAPI: https://rapidapi.com/user/jjrm365-kIFr3Nx_odV
- **Fallback**: API-Tennis (14-day free trial), Goalserve ($150/mo)

## Key Features for Model

**Backbone (from Sackmann Elo data):**
- **Surface-adjusted Elo** — overall Elo + surface-specific Elo (clay/hard/grass). This single feature encodes ranking, H2H history, and surface affinity.

**Supplementary features (from Sackmann match CSVs):**
1. **H2H record** - historical win/loss between the two players (direct matchup history beyond what Elo captures)
2. **Recent form** - win rate in last 10-20 matches
3. **Fatigue** - days since last match, matches played in last 7/14 days
4. **Tournament round** - early rounds have more upsets
5. **Serve stats** - ace rate, 1st serve %, break points saved (from 2008+ for Challengers)
6. **Age/experience** - career matches played, years on tour

## Polymarket CLOB Execution

### py-clob-client Setup
- Install: `pip install py-clob-client`
- Requires: Polygon wallet private key + Polymarket API credentials
- API creds obtained via: CLOB API `/auth/api-key` endpoint (sign with wallet)
- Docs: https://docs.polymarket.com

### CRITICAL py-clob-client Gotchas
These are hard-won lessons from a previous Polymarket project. Do not skip.

1. **Order book sorting is inconsistent.** When fetching order book, `bids` may be sorted ascending or descending depending on the endpoint/version. ALWAYS verify: best bid = highest price, best ask = lowest price. Sort explicitly, never assume `bids[0]` or `bids[-1]` is the best bid.

2. **Token ID mapping.** Each Polymarket market has TWO tokens (e.g., "Player A wins" and "Player B wins"). The `outcomePrices` array maps 1:1 with the `outcomes` array. Double-check you're buying the right token.

3. **Order amounts are in USDC (6 decimals).** Polymarket uses USDC on Polygon. Amounts are raw integers (1000000 = 1 USDC). Be careful with decimal handling.

4. **Approval flow.** Before placing orders, the wallet must approve USDC spending on the Polymarket CTF Exchange contract. This is a one-time on-chain transaction per wallet.

5. **Order types.** Use limit orders (GTC - Good Till Cancelled). Market orders don't exist on CLOB; you place a limit order at current best price. For tennis matches, we want to get fills at specific prices that represent our value threshold, not chase the market.

6. **Rate limits.** The CLOB API has rate limits. Don't poll aggressively. For tennis (matches happen over hours), checking every 30-60 seconds is plenty.

### Early Exit (EE) Logic
If we buy a position and the market moves in our favor (our player's odds increase), we can sell before match resolution to lock in profit:
- Buy "Player A" at $0.55 (model says 65% chance)
- Market moves to $0.70 during the match
- Sell at $0.70, profit $0.15 per share without waiting for resolution
- Threshold: sell if current price >= buy price + min_profit (e.g., $0.10)

### Position Sizing
- Kelly criterion or flat sizing (1-2% of bankroll per bet)
- Scale by edge size: bigger model-vs-market gap = larger position
- Maximum position per match: never more than 5% of bankroll
- Start with flat sizing; switch to Kelly only after 100+ calibrated bets

## Phase Plan

### Phase 0: Data Collection - ACTIVE
- Polymarket scraper running every 30 min via Windows Task Scheduler
- Stores moneyline odds snapshots in SQLite
- Also supports `--backfill` to fetch closed/resolved events with outcomes
- **Goal:** accumulate enough Polymarket odds history to backtest against real PM prices

### Phase 1: Backtest Model (local)
- Download Sackmann CSVs + Elo ratings
- Engineer features with surface-adjusted Elo as backbone
- Train XGBoost/LightGBM classifier with calibration layer
- Backtest against actual Polymarket odds (from Phase 0 data)
- KEY METRIC: calibration (when model says 65%, does the player win ~65%?)
- MUST use proper train/test split. Never evaluate on training data.
- Compare against logistic regression baseline

### Phase 2: Paper Trading (local)
- Connect live tennis data (SportRadar/Matchstat, pluggable)
- Build fuzzy player name matching (Polymarket names <-> Sackmann/SportRadar IDs)
- Match upcoming Polymarket events to player data
- Generate predictions, compare to Polymarket odds
- Log everything, no real bets
- Measure: how often does model find edge? what's the theoretical ROI?

### Phase 3: Live (needs VPS)
- Deploy to Hetzner VPS (Helsinki)
- Small bankroll ($50-100)
- Strict bet sizing (Kelly criterion or flat 1-2%)
- 7-DAY NO-DEPLOY RULE after any parameter change
- Minimum 100 bets before evaluating

## Git Workflow

### Branches
- **`main`** — stable, tested code only. Merged to at phase milestones, tagged with versions.
- **`dev`** — daily working branch. All Claude sessions and auto-pushes happen here.
- **`feature/*`** — branch off `dev` for bigger changes (e.g., `feature/model-training`), merge back to `dev` via PR when done.

### Flow
1. Work happens on `dev` (or a feature branch off `dev`)
2. Scraper auto-commits+pushes to `dev` every 30 min
3. When a phase milestone is complete: merge `dev` → `main`, tag it (v0.1.0, v0.2.0, etc.)
4. `TennisPM.bat` always checks out `dev` before launching Claude

### Tags
- `v0.1.0` — Phase 0 scraper complete

## SQLite Schema

Database: `data/tennis_pm.db` (defined in `scraper/db.py`)

**events** — one row per Polymarket event (match)
- id, title, slug, series (atp/wta), league, game_id, start_date, end_date, created_at, active, closed

**markets** — one row per moneyline market
- id, event_id, question, outcome_a, outcome_b, token_id_a, token_id_b, condition_id, resolution_status, winner, closed_time

**snapshots** — one row per odds reading (every 30 min for active markets)
- id, market_id, scraped_at, price_a, price_b, best_bid, best_ask, last_trade_price, spread, volume, liquidity

## File Structure

```
tennis-pm/
├── CLAUDE.md              # This file — project context for every session
├── requirements.txt       # Dependencies (Phase 0 uses stdlib only)
├── run_scraper.py         # Entry point: scrape once, --loop, or --backfill
├── scrape_task.bat        # Windows Task Scheduler target
├── config/
│   └── settings.py        # Gamma API URLs, series IDs, intervals, thresholds
├── scraper/
│   ├── db.py              # SQLite schema, upsert/insert functions
│   └── pm_scraper.py      # Gamma API fetch + scrape logic
├── data/
│   ├── tennis_pm.db       # SQLite database (auto-created)
│   ├── scraper.log        # Appended by scheduled task
│   └── sackmann/          # (Phase 1) Downloaded Sackmann CSVs + Elo
├── model/                 # (Phase 1) To be built
│   ├── train.py           # Model training
│   ├── features.py        # Feature engineering
│   └── evaluate.py        # Backtesting & calibration
├── bot/                   # (Phase 2-3) To be built
│   ├── main.py            # Main event loop
│   ├── predictor.py       # Live predictions
│   ├── executor.py        # Polymarket order placement
│   ├── telegram_bot.py    # Monitoring & alerts
│   └── early_exit.py      # EE logic
├── config/
│   └── settings.py        # All configurable parameters
├── changelog.md           # (Phase 3) Every parameter change with reasoning
└── deploy/
    └── deploy.sh          # (Phase 3) VPS deployment
```

## Environment

- Python 3.13 (on dev machine), 3.11+ minimum
- Dev machine: Windows 10 Pro, Git Bash shell
- Launch shortcut: `C:\Users\Rn5ho\Desktop\TennisPM.bat` (opens Claude Code with --dangerously-skip-permissions)
- Scheduled task: `TennisPM_Scraper` (Windows Task Scheduler, every 30 min, interactive mode)
- Deployment target: Hetzner VPS (Helsinki) for Phase 3
- Phase 0 deps: Python stdlib only (no pip install needed)
- Phase 1+ deps: xgboost, lightgbm, pandas, scikit-learn, aiohttp
- Phase 3 deps: py-clob-client, python-telegram-bot
