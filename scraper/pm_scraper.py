"""Polymarket Gamma API scraper for tennis events."""

import logging
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json

from config.settings import (
    GAMMA_API_BASE,
    ATP_SERIES_ID,
    WTA_SERIES_ID,
    REQUEST_TIMEOUT_SECONDS,
    REQUEST_DELAY_SECONDS,
)
from scraper.db import get_connection, upsert_event, upsert_market, insert_snapshot

logger = logging.getLogger(__name__)

SERIES = {
    "atp": ATP_SERIES_ID,
    "wta": WTA_SERIES_ID,
}


def _fetch_json(url: str) -> list | dict:
    """Fetch JSON from a URL using stdlib only."""
    req = Request(url, headers={"User-Agent": "Tennis-PM-Scraper/0.1"})
    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        logger.error("HTTP %d fetching %s", e.code, url)
        raise
    except URLError as e:
        logger.error("URL error fetching %s: %s", url, e.reason)
        raise


def fetch_events(series_id: str, closed: bool = False) -> list[dict]:
    """Fetch tennis events from Gamma API, paginating through all results.

    Args:
        series_id: ATP or WTA series ID.
        closed: If True, also fetch closed/resolved events (for backfill).
    """
    PAGE_SIZE = 100
    base_params = f"series_id={series_id}&limit={PAGE_SIZE}"
    if not closed:
        base_params += "&active=true&closed=false"

    all_events = []
    offset = 0

    while True:
        url = f"{GAMMA_API_BASE}/events?{base_params}&offset={offset}"
        logger.info("Fetching %s", url)
        data = _fetch_json(url)

        if isinstance(data, dict):
            data = data.get("data", data.get("events", [data]))
        batch = data if isinstance(data, list) else [data]

        # Empty response or placeholder with no real data = done
        if not batch or not batch[0].get("id"):
            break

        all_events.extend(batch)

        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY_SECONDS)

    return all_events


def scrape_once() -> dict:
    """Run one scrape cycle: fetch all ATP + WTA events, store snapshots.

    Returns a summary dict with counts.
    """
    scraped_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    stats = {"events": 0, "markets": 0, "snapshots": 0, "errors": 0}

    for series_name, series_id in SERIES.items():
        try:
            events = fetch_events(series_id)
        except Exception:
            logger.exception("Failed to fetch %s events", series_name)
            stats["errors"] += 1
            continue

        logger.info("Found %d %s events", len(events), series_name.upper())

        for event in events:
            try:
                upsert_event(conn, event, series_name)
                stats["events"] += 1

                for market in event.get("markets", []):
                    # Only track moneyline markets (skip set winners, totals, etc.)
                    if market.get("sportsMarketType") != "moneyline":
                        continue

                    event_id = str(event["id"])
                    upsert_market(conn, market, event_id)
                    stats["markets"] += 1

                    # Only snapshot active, non-closed markets
                    if market.get("active") and not market.get("closed"):
                        insert_snapshot(conn, market, scraped_at)
                        stats["snapshots"] += 1

            except Exception:
                logger.exception("Error processing event %s", event.get("id"))
                stats["errors"] += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    conn.commit()
    conn.close()
    return stats


def backfill_closed() -> dict:
    """Fetch closed/resolved events to capture historical outcomes."""
    scraped_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    stats = {"events": 0, "markets": 0, "errors": 0}

    for series_name, series_id in SERIES.items():
        try:
            events = fetch_events(series_id, closed=True)
        except Exception:
            logger.exception("Failed to fetch closed %s events", series_name)
            stats["errors"] += 1
            continue

        logger.info("Found %d %s events (including closed)", len(events), series_name.upper())

        for event in events:
            try:
                upsert_event(conn, event, series_name)
                stats["events"] += 1

                for market in event.get("markets", []):
                    if market.get("sportsMarketType") != "moneyline":
                        continue
                    upsert_market(conn, market, str(event["id"]))
                    stats["markets"] += 1

            except Exception:
                logger.exception("Error processing event %s", event.get("id"))
                stats["errors"] += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    conn.commit()
    conn.close()
    return stats
