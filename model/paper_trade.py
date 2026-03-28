"""Paper trading system: resolve outcomes and track P&L.

Three commands:
    python -m model.paper_trade resolve   # fill in outcomes for resolved markets
    python -m model.paper_trade report    # show P&L summary
    python -m model.paper_trade run       # run predictor + resolve in one go
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import DB_PATH, PAPER_TRADES_PATH, MIN_EDGE_THRESHOLD

logger = logging.getLogger(__name__)

FLAT_BET_SIZE = 10.0  # $10 per paper bet for P&L tracking


def resolve_outcomes() -> int:
    """Check resolved PM markets and fill in paper trade outcomes.

    Returns number of newly resolved trades.
    """
    if not PAPER_TRADES_PATH.exists():
        print("No paper trades to resolve.")
        return 0

    with open(PAPER_TRADES_PATH) as f:
        trades = json.load(f)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    resolved = 0

    for trade in trades:
        if trade.get("outcome") is not None:
            continue  # already resolved

        # Find the market by matching title and players
        bet_on = trade["bet_on"]
        opponent = trade["opponent"]

        row = conn.execute("""
            SELECT m.winner, m.outcome_a, m.outcome_b
            FROM markets m
            JOIN events e ON e.id = m.event_id
            WHERE e.title = ?
            AND m.winner IS NOT NULL
            LIMIT 1
        """, (trade["title"],)).fetchone()

        if not row:
            continue  # not resolved yet

        winner = row["winner"]

        if winner == bet_on:
            trade["outcome"] = "win"
            # Profit = (1/pm_price - 1) * bet_size (buy at pm_price, get $1 if win)
            trade["pnl"] = round((1.0 / trade["pm_price"] - 1.0) * FLAT_BET_SIZE, 2)
        elif winner == opponent:
            trade["outcome"] = "loss"
            trade["pnl"] = round(-FLAT_BET_SIZE, 2)
        else:
            # Winner name doesn't match either player — might be name format issue
            trade["outcome"] = "unknown"
            trade["pnl"] = 0.0
            logger.warning(f"Winner '{winner}' doesn't match '{bet_on}' or '{opponent}' in {trade['title']}")

        trade["resolved_at"] = datetime.now(timezone.utc).isoformat()
        resolved += 1

    conn.close()

    with open(PAPER_TRADES_PATH, "w") as f:
        json.dump(trades, f, indent=2)

    print(f"Resolved {resolved} trades ({sum(1 for t in trades if t.get('outcome') is not None)} total resolved)")
    return resolved


def report() -> None:
    """Print P&L report from paper trades."""
    if not PAPER_TRADES_PATH.exists():
        print("No paper trades yet. Run: python -m model.predictor")
        return

    with open(PAPER_TRADES_PATH) as f:
        trades = json.load(f)

    total = len(trades)
    resolved = [t for t in trades if t.get("outcome") is not None]
    pending = total - len(resolved)
    wins = [t for t in resolved if t["outcome"] == "win"]
    losses = [t for t in resolved if t["outcome"] == "loss"]
    unknown = [t for t in resolved if t["outcome"] == "unknown"]

    total_pnl = sum(t.get("pnl", 0) for t in resolved)
    total_wagered = len([t for t in resolved if t["outcome"] in ("win", "loss")]) * FLAT_BET_SIZE

    print(f"\n{'='*60}")
    print(f"PAPER TRADING REPORT")
    print(f"{'='*60}")
    print(f"  Total signals logged: {total}")
    print(f"  Resolved:  {len(resolved)}")
    print(f"  Pending:   {pending}")
    print()

    if resolved:
        win_loss = len(wins) + len(losses)
        if win_loss > 0:
            win_rate = len(wins) / win_loss
            print(f"  Wins:    {len(wins)}")
            print(f"  Losses:  {len(losses)}")
            if unknown:
                print(f"  Unknown: {len(unknown)}")
            print(f"  Win rate: {win_rate:.1%}")
            print()
            print(f"  Total wagered: ${total_wagered:,.2f} (${FLAT_BET_SIZE:.0f} flat bets)")
            print(f"  Total P&L:     ${total_pnl:+,.2f}")
            if total_wagered > 0:
                roi = total_pnl / total_wagered
                print(f"  ROI:           {roi:+.1%}")
        print()

        # Breakdown by strategy type (contrarian vs reinforcing)
        wl = [t for t in resolved if t["outcome"] in ("win", "loss")]
        contrarian = [t for t in wl if t.get("contrarian")]
        reinforcing = [t for t in wl if not t.get("contrarian")]

        if contrarian or reinforcing:
            print(f"  --- By Strategy ---")
            for label, bucket in [("CONTRARIAN", contrarian), ("Reinforcing", reinforcing)]:
                if not bucket:
                    continue
                bw = sum(1 for t in bucket if t["outcome"] == "win")
                bl = sum(1 for t in bucket if t["outcome"] == "loss")
                bpnl = sum(t.get("pnl", 0) for t in bucket)
                bwag = len(bucket) * FLAT_BET_SIZE
                broi = bpnl / bwag if bwag > 0 else 0
                print(f"  {label:>12}: {bw}W/{bl}L  P&L: ${bpnl:+,.2f}  ROI: {broi:+.1%}")

        # Breakdown by edge bucket
        print(f"\n  --- By Edge Size ---")
        buckets = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
        for lo, hi in buckets:
            bucket = [t for t in wl if lo <= t.get("edge", 0) < hi]
            if bucket:
                bw = sum(1 for t in bucket if t["outcome"] == "win")
                bl = sum(1 for t in bucket if t["outcome"] == "loss")
                bpnl = sum(t.get("pnl", 0) for t in bucket)
                print(f"  {lo:.0%}-{hi:.0%}: {bw}W/{bl}L  P&L: ${bpnl:+,.2f}")

        # Show last 10 resolved trades
        print(f"\n  --- Recent Resolved Trades ---")
        recent = sorted([t for t in resolved if t["outcome"] in ("win", "loss")],
                       key=lambda t: t.get("resolved_at", ""), reverse=True)[:10]
        for t in recent:
            icon = "W" if t["outcome"] == "win" else "L"
            print(f"  [{icon}] {t['title']}")
            print(f"      Bet: {t['bet_on']} | Model: {t['model_prob']:.1%} | PM: {t['pm_price']:.1%} | Edge: {t['edge']:+.1%} | P&L: ${t['pnl']:+.2f}")
    else:
        print("  No trades resolved yet. Markets take time to close.")
        print("  Run the scraper backfill to pick up resolved markets:")
        print("    python run_scraper.py --backfill")

    print(f"{'='*60}")


def run():
    """Run predictor + resolve outcomes in one command."""
    from model.predictor import scan_markets

    print("=== Running predictor ===\n")
    edges = scan_markets()

    print("\n=== Resolving outcomes ===\n")
    # First, backfill closed events to get latest resolutions
    print("Backfilling closed events...")
    from scraper.pm_scraper import backfill_closed
    stats = backfill_closed()
    print(f"Backfill: {stats}")

    resolve_outcomes()

    print("\n=== Report ===")
    report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "resolve":
        resolve_outcomes()
    elif cmd == "report":
        report()
    elif cmd == "run":
        run()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m model.paper_trade [resolve|report|run]")
