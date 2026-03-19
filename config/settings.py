"""Configuration for Tennis-PM."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "tennis_pm.db"

# Polymarket APIs
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
ATP_SERIES_ID = "10365"
WTA_SERIES_ID = "10366"

# Scraper settings
SCRAPE_INTERVAL_MINUTES = 10  # how often to snapshot odds
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 1  # delay between API calls to be polite

# Betting defaults (revisit with real data)
MIN_EDGE_THRESHOLD = 0.05  # 5% minimum edge to consider a bet
MIN_VOLUME = 500           # skip markets with less than $500 volume (price is noise)

# Paper trading
PAPER_TRADES_PATH = PROJECT_ROOT / "data" / "paper_trades.json"

# --- Phase 1: Model Training ---

# Sackmann data
SACKMANN_ATP_DIR = PROJECT_ROOT / "data" / "sackmann" / "tennis_atp"
SACKMANN_WTA_DIR = PROJECT_ROOT / "data" / "sackmann" / "tennis_wta"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"

# Elo settings
ELO_START_RATING = 1500
ELO_K_INITIAL = 40       # first 20 matches
ELO_K_MEDIUM = 20        # 20-50 matches
ELO_K_STABLE = 10        # 50+ matches
ELO_SURFACE_K_MULTIPLIER = 1.5

# Training data splits (years, inclusive)
ELO_BURN_IN_YEARS = (2000, 2014)
TRAIN_YEARS = (2015, 2022)
VAL_YEARS = (2023, 2023)
TEST_YEARS = (2024, 2024)
MATCH_YEARS = (2000, 2024)

# Feature settings
ROLLING_WINDOW = 30       # matches for rolling serve stats
RECENT_FORM_WINDOW = 20   # matches for win rate
