#!/usr/bin/env python3
"""Entry point for the Polymarket tennis scraper.

Usage:
    python run_scraper.py              # scrape once
    python run_scraper.py --loop       # scrape every SCRAPE_INTERVAL_MINUTES
    python run_scraper.py --backfill   # fetch closed events for historical data
"""

import argparse
import logging
import time
import sys

from config.settings import SCRAPE_INTERVAL_MINUTES, DB_PATH
from scraper.pm_scraper import scrape_once, backfill_closed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_scraper")


def main():
    parser = argparse.ArgumentParser(description="Polymarket tennis odds scraper")
    parser.add_argument("--loop", action="store_true",
                        help=f"Run continuously every {SCRAPE_INTERVAL_MINUTES} min")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch closed/resolved events for historical data")
    args = parser.parse_args()

    logger.info("Database: %s", DB_PATH)

    if args.backfill:
        logger.info("Backfilling closed events...")
        stats = backfill_closed()
        logger.info("Backfill done: %s", stats)
        return

    if args.loop:
        logger.info("Starting scraper loop (every %d min)...", SCRAPE_INTERVAL_MINUTES)
        while True:
            try:
                stats = scrape_once()
                logger.info("Scrape done: %s", stats)
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Scrape cycle failed")
            time.sleep(SCRAPE_INTERVAL_MINUTES * 60)
    else:
        stats = scrape_once()
        logger.info("Scrape done: %s", stats)
        _print_summary()


def _print_summary():
    """Print a quick summary of what's in the database."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    events = conn.execute("SELECT COUNT(*) as n FROM events").fetchone()["n"]
    markets = conn.execute("SELECT COUNT(*) as n FROM markets").fetchone()["n"]
    snapshots = conn.execute("SELECT COUNT(*) as n FROM snapshots").fetchone()["n"]

    print(f"\n--- Database Summary ---")
    print(f"Events:    {events}")
    print(f"Markets:   {markets}")
    print(f"Snapshots: {snapshots}")

    # Show active events
    rows = conn.execute("""
        SELECT e.series, e.title, m.outcome_a, m.outcome_b,
               s.price_a, s.price_b, s.volume, s.liquidity
        FROM events e
        JOIN markets m ON m.event_id = e.id
        LEFT JOIN snapshots s ON s.market_id = m.id
            AND s.scraped_at = (SELECT MAX(scraped_at) FROM snapshots WHERE market_id = m.id)
        WHERE e.active = 1 AND e.closed = 0
        ORDER BY e.series, s.volume DESC
    """).fetchall()

    if rows:
        print(f"\nActive matches ({len(rows)}):")
        for r in rows:
            series = r["series"].upper()
            p_a = r["price_a"]
            p_b = r["price_b"]
            if p_a is not None:
                odds_str = f"{r['outcome_a']} {p_a:.3f} / {r['outcome_b']} {p_b:.3f}"
            else:
                odds_str = "no snapshot"
            vol = r["volume"] or 0
            print(f"  [{series}] {r['title']}")
            print(f"         {odds_str}  (vol: ${vol:,.0f})")

    conn.close()


if __name__ == "__main__":
    main()
