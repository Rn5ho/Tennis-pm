"""SQLite database layer for storing Polymarket tennis data."""

import json
import sqlite3
from pathlib import Path

from config.settings import DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a database connection, creating the DB and tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            slug TEXT NOT NULL,
            series TEXT NOT NULL,
            league TEXT,
            game_id INTEGER,
            start_date TEXT,
            end_date TEXT,
            created_at TEXT,
            active INTEGER DEFAULT 1,
            closed INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS markets (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            question TEXT,
            outcome_a TEXT NOT NULL,
            outcome_b TEXT NOT NULL,
            token_id_a TEXT,
            token_id_b TEXT,
            condition_id TEXT,
            resolution_status TEXT,
            winner TEXT,
            closed_time TEXT,
            FOREIGN KEY (event_id) REFERENCES events(id)
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            scraped_at TEXT NOT NULL,
            price_a REAL,
            price_b REAL,
            best_bid REAL,
            best_ask REAL,
            last_trade_price REAL,
            spread REAL,
            volume REAL,
            liquidity REAL,
            FOREIGN KEY (market_id) REFERENCES markets(id)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_market_time
            ON snapshots(market_id, scraped_at);

        CREATE INDEX IF NOT EXISTS idx_events_series
            ON events(series);

        CREATE INDEX IF NOT EXISTS idx_markets_event
            ON markets(event_id);
    """)
    conn.commit()


def _parse_list_field(val) -> list:
    """Parse a field that may be a list or a JSON-encoded string."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def upsert_event(conn: sqlite3.Connection, event: dict, series: str) -> None:
    """Insert or update an event."""
    metadata = event.get("eventMetadata") or {}
    conn.execute("""
        INSERT INTO events (id, title, slug, series, league, game_id,
                            start_date, end_date, created_at, active, closed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            active=excluded.active,
            closed=excluded.closed,
            end_date=excluded.end_date
    """, (
        str(event["id"]),
        event.get("title", ""),
        event.get("slug", ""),
        series,
        metadata.get("league"),
        event.get("gameId"),
        event.get("startDate"),
        event.get("endDate"),
        event.get("createdAt"),
        int(event.get("active", True)),
        int(event.get("closed", False)),
    ))


def upsert_market(conn: sqlite3.Connection, market: dict, event_id: str) -> None:
    """Insert or update a market."""
    outcomes = _parse_list_field(market.get("outcomes", []))
    token_ids = _parse_list_field(market.get("clobTokenIds", []))

    outcome_a = outcomes[0] if len(outcomes) > 0 else ""
    outcome_b = outcomes[1] if len(outcomes) > 1 else ""
    token_id_a = token_ids[0] if len(token_ids) > 0 else ""
    token_id_b = token_ids[1] if len(token_ids) > 1 else ""

    # Determine winner from resolution status and prices
    winner = None
    resolution = market.get("umaResolutionStatus", "")
    if resolution == "resolved":
        prices = _parse_list_field(market.get("outcomePrices", []))
        if len(prices) >= 2:
            try:
                if float(prices[0]) > 0.5:
                    winner = outcome_a
                else:
                    winner = outcome_b
            except (ValueError, TypeError):
                pass

    conn.execute("""
        INSERT INTO markets (id, event_id, question, outcome_a, outcome_b,
                             token_id_a, token_id_b, condition_id,
                             resolution_status, winner, closed_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            resolution_status=excluded.resolution_status,
            winner=COALESCE(excluded.winner, markets.winner),
            closed_time=COALESCE(excluded.closed_time, markets.closed_time)
    """, (
        str(market["id"]),
        str(event_id),
        market.get("question", ""),
        outcome_a,
        outcome_b,
        token_id_a,
        token_id_b,
        market.get("conditionId", ""),
        resolution,
        winner,
        market.get("closedTime"),
    ))


def insert_snapshot(conn: sqlite3.Connection, market: dict, scraped_at: str) -> None:
    """Insert a point-in-time odds snapshot for a market."""
    prices = _parse_list_field(market.get("outcomePrices", []))
    price_a = float(prices[0]) if len(prices) > 0 else None
    price_b = float(prices[1]) if len(prices) > 1 else None

    conn.execute("""
        INSERT INTO snapshots (market_id, scraped_at, price_a, price_b,
                               best_bid, best_ask, last_trade_price,
                               spread, volume, liquidity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(market["id"]),
        scraped_at,
        price_a,
        price_b,
        market.get("bestBid"),
        market.get("bestAsk"),
        market.get("lastTradePrice"),
        market.get("spread"),
        float(market.get("volume", 0) or 0),
        float(market.get("liquidityClob", 0) or market.get("liquidity", 0) or 0),
    ))
