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

        CREATE TABLE IF NOT EXISTS book_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            scraped_at TEXT NOT NULL,
            -- Token A (outcome A) order book summary
            best_bid_a REAL,
            best_bid_size_a REAL,
            best_ask_a REAL,
            best_ask_size_a REAL,
            bid_liquidity_a REAL,
            ask_liquidity_a REAL,
            n_bid_levels_a INTEGER,
            n_ask_levels_a INTEGER,
            -- Token B (outcome B) order book summary
            best_bid_b REAL,
            best_bid_size_b REAL,
            best_ask_b REAL,
            best_ask_size_b REAL,
            bid_liquidity_b REAL,
            ask_liquidity_b REAL,
            n_bid_levels_b INTEGER,
            n_ask_levels_b INTEGER,
            -- Top 10 levels each side as JSON for detailed analysis
            depth_a TEXT,
            depth_b TEXT,
            FOREIGN KEY (market_id) REFERENCES markets(id)
        );

        CREATE INDEX IF NOT EXISTS idx_book_snapshots_market_time
            ON book_snapshots(market_id, scraped_at);
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


def _summarize_book(book: dict) -> dict:
    """Extract summary metrics from a CLOB order book response."""
    raw_bids = book.get("bids", [])
    raw_asks = book.get("asks", [])

    # Parse and sort: bids descending by price, asks ascending
    bids = sorted(
        [{"price": float(b["price"]), "size": float(b["size"])} for b in raw_bids],
        key=lambda x: -x["price"],
    )
    asks = sorted(
        [{"price": float(a["price"]), "size": float(a["size"])} for a in raw_asks],
        key=lambda x: x["price"],
    )

    best_bid = bids[0]["price"] if bids else None
    best_bid_size = bids[0]["size"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    best_ask_size = asks[0]["size"] if asks else None

    # Total dollar liquidity (price * size summed across all levels)
    bid_liquidity = sum(b["price"] * b["size"] for b in bids)
    ask_liquidity = sum(a["price"] * a["size"] for a in asks)

    # Top 10 levels each side for detailed storage
    depth = {
        "bids": [[b["price"], b["size"]] for b in bids[:10]],
        "asks": [[a["price"], a["size"]] for a in asks[:10]],
    }

    return {
        "best_bid": best_bid,
        "best_bid_size": best_bid_size,
        "best_ask": best_ask,
        "best_ask_size": best_ask_size,
        "bid_liquidity": bid_liquidity,
        "ask_liquidity": ask_liquidity,
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
        "depth": json.dumps(depth),
    }


def insert_book_snapshot(
    conn: sqlite3.Connection,
    market_id: str,
    scraped_at: str,
    book_a: dict,
    book_b: dict,
) -> None:
    """Insert an order book snapshot for a market (both tokens)."""
    sa = _summarize_book(book_a)
    sb = _summarize_book(book_b)

    conn.execute("""
        INSERT INTO book_snapshots (
            market_id, scraped_at,
            best_bid_a, best_bid_size_a, best_ask_a, best_ask_size_a,
            bid_liquidity_a, ask_liquidity_a, n_bid_levels_a, n_ask_levels_a,
            best_bid_b, best_bid_size_b, best_ask_b, best_ask_size_b,
            bid_liquidity_b, ask_liquidity_b, n_bid_levels_b, n_ask_levels_b,
            depth_a, depth_b
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market_id, scraped_at,
        sa["best_bid"], sa["best_bid_size"], sa["best_ask"], sa["best_ask_size"],
        sa["bid_liquidity"], sa["ask_liquidity"], sa["n_bid_levels"], sa["n_ask_levels"],
        sb["best_bid"], sb["best_bid_size"], sb["best_ask"], sb["best_ask_size"],
        sb["bid_liquidity"], sb["ask_liquidity"], sb["n_bid_levels"], sb["n_ask_levels"],
        sa["depth"], sb["depth"],
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
