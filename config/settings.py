"""Configuration for Tennis-PM."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "tennis_pm.db"

# Polymarket Gamma API
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
ATP_SERIES_ID = "10365"
WTA_SERIES_ID = "10366"

# Scraper settings
SCRAPE_INTERVAL_MINUTES = 30  # how often to snapshot odds
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_DELAY_SECONDS = 1  # delay between API calls to be polite

# Betting defaults (revisit with real data)
MIN_EDGE_THRESHOLD = 0.05  # 5% minimum edge to consider a bet
