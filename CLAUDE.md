# Tennis Polymarket Betting Bot

## What This Is

Automated tennis value betting system on Polymarket. Compares AI-generated match win probabilities against Polymarket implied odds. Bets when the gap exceeds a threshold.

## Why This Should Work

- Tennis has real predictive features (rankings, H2H, surface, fatigue, recent form)
- Polymarket tennis markets are thin ($2K-$25K volume per match) = less efficient pricing
- We're classifying/filtering (find mispriced matches) not predicting direction in efficient markets
- Counterparties are retail, not HFT algorithms

## Architecture

```
[Data Layer]
  Jeff Sackmann CSV (historical) --> Model Training
  SportRadar / Matchstat API (live) --> Live Player Stats
  Polymarket Gamma API (live) --> Current Implied Odds

[Prediction Layer]
  Feature Engineering --> ML Model --> Win Probability per player

[Execution Layer]
  Compare model prob vs Polymarket price --> Bet when edge > threshold
  py-clob-client --> Order placement on Polymarket CLOB
  Early exit logic --> Sell position if odds shift favorably
  Telegram bot --> Monitoring & alerts
```

## Data Sources

### Historical Match Data (Training) - FREE
- **Jeff Sackmann/tennis_atp**: https://github.com/JeffSackmann/tennis_atp
- ATP matches from 1968, match stats from 1991, Challengers from 2008
- CSV format: player rankings, H2H, serve stats, break points, all integer totals
- License: CC BY-NC-SA 4.0 (non-commercial)
- Also has WTA: https://github.com/JeffSackmann/tennis_wta

### Polymarket Odds - FREE, NO AUTH
- Gamma API (read-only, no auth): `https://gamma-api.polymarket.com`
- Sports endpoint: `GET /sports` (lists all leagues)
- ATP series_id: `10365`, WTA series_id: `10366`
- Events: `GET /events?series_id=10365&active=true&closed=false`
- Each event has `outcomePrices` array = implied probabilities
- CLOB API for order placement (needs wallet + API creds)
- Docs: https://docs.polymarket.com/quickstart/fetching-data

### Live Tennis Data - FREE TO START
- **SportRadar**: 30-day free trial, same data as production, lower rate limits
  - v3 Tennis API, covers ATP + WTA + Challengers + 20K UTR events
  - Endpoints: seasons, schedules, competitor profiles, H2H, match summaries
  - Docs: https://developer.sportradar.com/tennis/reference/overview
- **Matchstat.com**: Free tier available, tennis-specific, includes pre-match odds
  - H2H stats, player profiles, historical data back to 1990
  - RapidAPI: https://rapidapi.com/user/jjrm365-kIFr3Nx_odV
- **Fallback**: API-Tennis (14-day free trial), Goalserve ($150/mo)

## Key Features for Model

Priority features to engineer from Sackmann data:

1. **Ranking gap** - absolute and relative difference in ATP ranking
2. **H2H record** - historical win/loss between the two players
3. **Surface form** - win rate on clay/hard/grass in last 6-12 months
4. **Recent form** - win rate in last 10-20 matches
5. **Fatigue** - days since last match, matches played in last 7/14 days
6. **Tournament round** - early rounds have more upsets
7. **Serve stats** - ace rate, 1st serve %, break points saved (from 2008+ for Challengers)
8. **Age/experience** - career matches played, years on tour

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

### Phase 0: Data Collection (NOW - runs locally, no server needed)
- Start scraping Polymarket ATP/WTA events daily via gamma API
- Store: event_id, match details, player names, implied odds, timestamps, resolution
- This builds the backtest dataset we don't have yet
- Can run as a cron job, GitHub Action, or simple script on any machine

### Phase 1: Backtest Model (Week 1-2, local)
- Download Sackmann CSVs
- Engineer features
- Train classifier (GradientBoosting or RandomForest)
- Backtest against traditional bookmaker closing odds as proxy for Polymarket
- KEY METRIC: calibration (when model says 65%, does the player win ~65%?)
- MUST use proper train/test split. Never evaluate on training data.

### Phase 2: Paper Trading (Week 3-4, local)
- Connect live SportRadar/Matchstat data
- Match upcoming Polymarket events to player data
- Generate predictions, compare to Polymarket odds
- Log everything, no real bets
- Measure: how often does model find edge? what's the theoretical ROI?

### Phase 3: Live (Week 5+, needs VPS)
- Deploy to Hetzner VPS
- Small bankroll ($50-100)
- Strict bet sizing (Kelly criterion or flat 1-2%)
- 7-DAY NO-DEPLOY RULE after any parameter change
- Minimum 100 bets before evaluating


## File Structure

```
tennis-pm/
├── CLAUDE.md              # This file
├── data/
│   ├── sackmann/          # Git submodule or downloaded CSVs
│   ├── polymarket/        # Scraped PM odds history
│   └── features/          # Engineered feature sets
├── model/
│   ├── train.py           # Model training
│   ├── features.py        # Feature engineering
│   └── evaluate.py        # Backtesting & calibration
├── scraper/
│   ├── pm_scraper.py      # Polymarket gamma API scraper
│   └── sportradar.py      # Live data fetcher
├── bot/
│   ├── main.py            # Main loop
│   ├── predictor.py       # Live predictions
│   ├── executor.py        # Polymarket order placement
│   ├── telegram_bot.py    # Monitoring
│   └── early_exit.py      # EE logic
├── config/
│   └── settings.py        # All configurable parameters
├── changelog.md           # Every parameter change with reasoning
└── deploy/
    └── deploy.sh          # VPS deployment
```

## Environment

- Python 3.11+
- Deployment target: Hetzner VPS (Helsinki)
- Key dependencies: py-clob-client, pandas, scikit-learn, aiohttp, python-telegram-bot
- Dev: runs locally for Phase 0-2, server only needed for Phase 3
