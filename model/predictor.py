"""Live predictor: compare model probabilities against Polymarket prices.

Loads the trained model and player map, then for each active PM market:
1. Look up both players in the verified name map
2. Compute features using latest Sackmann data
3. Generate win probability
4. Compare against PM implied odds
5. Flag markets where edge exceeds threshold
"""

import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    DB_PATH, MODEL_DIR, MIN_EDGE_THRESHOLD,
    SACKMANN_ATP_DIR, SACKMANN_WTA_DIR,
    ROLLING_WINDOW, RECENT_FORM_WINDOW,
    ELO_START_RATING,
)
from model.name_match import load_player_map, lookup
from model.elo import compute_elo, SURFACE_MAP, expected_score
from model.features import FEATURE_NAMES, ROUND_NUM, LEVEL_NUM


def load_model(name: str = "best_model"):
    """Load a trained model."""
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m model.train` first.")
    return joblib.load(path)


def _load_player_stats() -> dict:
    """Load per-player stats from Sackmann data.

    Returns dict: {player_id: {
        'elo': float, 'surface_elo': {surface: float},
        'rank': float, 'rank_points': float,
        'age': float, 'height': float,
        'form': float, 'matches_7d': int, 'matches_14d': int,
        'days_rest': float, 'games_last': float,
        'ace_rate': float, '1st_pct': float, '1st_won': float, 'bp_saved': float,
        'h2h': {opponent_id: (wins, losses)},
    }}
    """
    # Load recent matches and compute Elo
    frames = []
    for tour, data_dir in [("atp", SACKMANN_ATP_DIR), ("wta", SACKMANN_WTA_DIR)]:
        data_dir = Path(data_dir)
        for year in range(2015, 2025):
            for pattern in [f"{tour}_matches_{year}.csv", f"{tour}_matches_qual_chall_{year}.csv"]:
                f = data_dir / pattern
                if f.exists():
                    df = pd.read_csv(f, low_memory=False)
                    df["tour"] = tour
                    frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
    matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    matches["tourney_date"] = matches["tourney_date"].astype(int)
    matches["winner_id"] = matches["winner_id"].astype(str)
    matches["loser_id"] = matches["loser_id"].astype(str)

    # Compute Elo on full history
    matches = compute_elo(matches)

    # Extract latest state per player
    stats = {}
    player_results = defaultdict(list)
    player_serve = defaultdict(list)
    player_dates = defaultdict(list)
    player_games = defaultdict(list)
    h2h = defaultdict(int)

    # Collect the final Elo and surface Elo from the last match
    elo_overall = {}
    elo_surface = defaultdict(dict)

    import re

    for row in matches.itertuples():
        wid, lid = row.winner_id, row.loser_id
        surf = SURFACE_MAP.get(getattr(row, "surface", "Hard"), "Hard")
        date = row.tourney_date

        # Update Elo snapshots (post-match, so these are the latest)
        elo_overall[wid] = row.winner_elo
        elo_overall[lid] = row.loser_elo
        elo_surface[wid][surf] = row.winner_surface_elo
        elo_surface[lid][surf] = row.loser_surface_elo

        # H2H
        score = getattr(row, "score", "")
        is_wo = not isinstance(score, str) or bool(re.search(r"\bW/O\b|\bDEF\b", str(score)))
        if not is_wo:
            h2h[(wid, lid)] += 1
            player_results[wid].append(1)
            player_results[lid].append(0)

            # Serve stats
            w_svpt = getattr(row, "w_svpt", None)
            if pd.notna(w_svpt) and w_svpt > 0:
                player_serve[wid].append({
                    "svpt": w_svpt,
                    "ace": getattr(row, "w_ace", 0) or 0,
                    "1stIn": getattr(row, "w_1stIn", 0) or 0,
                    "1stWon": getattr(row, "w_1stWon", 0) or 0,
                    "bpSaved": getattr(row, "w_bpSaved", 0) or 0,
                    "bpFaced": getattr(row, "w_bpFaced", 0) or 0,
                })
            l_svpt = getattr(row, "l_svpt", None)
            if pd.notna(l_svpt) and l_svpt > 0:
                player_serve[lid].append({
                    "svpt": l_svpt,
                    "ace": getattr(row, "l_ace", 0) or 0,
                    "1stIn": getattr(row, "l_1stIn", 0) or 0,
                    "1stWon": getattr(row, "l_1stWon", 0) or 0,
                    "bpSaved": getattr(row, "l_bpSaved", 0) or 0,
                    "bpFaced": getattr(row, "l_bpFaced", 0) or 0,
                })

        player_dates[wid].append(date)
        player_dates[lid].append(date)

        # Games from score
        total_games = 0
        if isinstance(score, str):
            for s in score.split():
                m = re.match(r"(\d+)-(\d+)", s)
                if m:
                    total_games += int(m.group(1)) + int(m.group(2))
        player_games[wid].append(total_games or np.nan)
        player_games[lid].append(total_games or np.nan)

    # Build final stats dict
    today = 20260317  # approximate

    def _date_diff(d1, d2):
        y1, m1, day1 = d1 // 10000, (d1 % 10000) // 100, d1 % 100
        y2, m2, day2 = d2 // 10000, (d2 % 10000) // 100, d2 % 100
        return (y2 - y1) * 365 + (m2 - m1) * 30 + (day2 - day1)

    for pid in set(list(elo_overall.keys())):
        srv = player_serve[pid][-ROLLING_WINDOW:]
        total_svpt = sum(s["svpt"] for s in srv) if srv else 0

        dates = player_dates[pid]
        days_rest = _date_diff(dates[-1], today) if dates else np.nan
        m7 = sum(1 for d in dates if _date_diff(d, today) <= 7) if dates else 0
        m14 = sum(1 for d in dates if _date_diff(d, today) <= 14) if dates else 0

        results = player_results[pid]

        stats[pid] = {
            "elo": elo_overall.get(pid, 1500),
            "surface_elo": dict(elo_surface.get(pid, {})),
            "rank": np.nan,  # would need current rankings file
            "rank_points": np.nan,
            "age": np.nan,
            "height": np.nan,
            "form": np.mean(results[-RECENT_FORM_WINDOW:]) if results else np.nan,
            "days_rest": days_rest,
            "matches_7d": m7,
            "matches_14d": m14,
            "games_last": player_games[pid][-1] if player_games[pid] else np.nan,
            "ace_rate": (sum(s["ace"] for s in srv) / total_svpt) if total_svpt > 0 else np.nan,
            "1st_pct": (sum(s["1stIn"] for s in srv) / total_svpt) if total_svpt > 0 else np.nan,
            "1st_won": (sum(s["1stWon"] for s in srv) / sum(s["1stIn"] for s in srv)) if srv and sum(s["1stIn"] for s in srv) > 0 else np.nan,
            "bp_saved": (sum(s["bpSaved"] for s in srv) / sum(s["bpFaced"] for s in srv)) if srv and sum(s["bpFaced"] for s in srv) > 0 else np.nan,
        }

        # Store H2H lookup
        stats[pid]["_h2h"] = {}
        for (w, l), count in h2h.items():
            if w == pid:
                stats[pid]["_h2h"][l] = stats[pid]["_h2h"].get(l, [0, 0])
                stats[pid]["_h2h"][l][0] += count
            elif l == pid:
                stats[pid]["_h2h"][w] = stats[pid]["_h2h"].get(w, [0, 0])
                stats[pid]["_h2h"][w][1] += count

    # Overlay backfilled Elo state (2025-2026) if available
    from model.backfill_elo import load_elo_state
    elo_state = load_elo_state()
    if elo_state:
        updated_count = 0
        for pid, elo_val in elo_state.get("elo", {}).items():
            if pid in stats:
                stats[pid]["elo"] = elo_val
                updated_count += 1
            else:
                # Player exists in backfill but not in historical stats — create entry
                stats[pid] = {
                    "elo": elo_val,
                    "surface_elo": {},
                    "rank": np.nan, "rank_points": np.nan,
                    "age": np.nan, "height": np.nan,
                    "form": np.nan, "days_rest": np.nan,
                    "matches_7d": 0, "matches_14d": 0,
                    "games_last": np.nan,
                    "ace_rate": np.nan, "1st_pct": np.nan,
                    "1st_won": np.nan, "bp_saved": np.nan,
                    "_h2h": {},
                }
                updated_count += 1

            # Overlay surface Elo
            surf_elo = elo_state.get("surface_elo", {}).get(pid, {})
            if surf_elo:
                stats[pid]["surface_elo"] = surf_elo

        logger.info(f"Overlaid backfilled Elo for {updated_count} players (through {elo_state.get('updated_through', '?')})")

    return stats


def predict_match(
    model,
    player_a_id: str,
    player_b_id: str,
    player_stats: dict,
    surface: str = "Hard",
    tourney_level: str = "A",
    round_name: str = "R32",
    best_of: int = 3,
    is_atp: bool = True,
) -> float:
    """Predict win probability for player_a.

    Returns: probability that player_a wins (0.0 to 1.0)
    """
    a = player_stats.get(player_a_id, {})
    b = player_stats.get(player_b_id, {})

    if not a or not b:
        return np.nan

    surf = SURFACE_MAP.get(surface, "Hard")

    # H2H
    a_h2h = a.get("_h2h", {}).get(player_b_id, [0, 0])
    b_h2h = b.get("_h2h", {}).get(player_a_id, [0, 0])
    h2h_diff = a_h2h[0] - b_h2h[0]

    features = {
        "elo_diff": a.get("elo", 1500) - b.get("elo", 1500),
        "surface_elo_diff": a.get("surface_elo", {}).get(surf, 1500) - b.get("surface_elo", {}).get(surf, 1500),
        "rank_diff": (b.get("rank", np.nan) or np.nan) - (a.get("rank", np.nan) or np.nan) if pd.notna(a.get("rank")) and pd.notna(b.get("rank")) else np.nan,
        "rank_points_diff": (a.get("rank_points", np.nan) or 0) - (b.get("rank_points", np.nan) or 0) if pd.notna(a.get("rank_points")) and pd.notna(b.get("rank_points")) else np.nan,
        "age_diff": (a.get("age", np.nan) or np.nan) - (b.get("age", np.nan) or np.nan) if pd.notna(a.get("age")) and pd.notna(b.get("age")) else np.nan,
        "height_diff": (a.get("height", np.nan) or np.nan) - (b.get("height", np.nan) or np.nan) if pd.notna(a.get("height")) and pd.notna(b.get("height")) else np.nan,
        "h2h_diff": h2h_diff,
        "form_diff": a.get("form", np.nan) - b.get("form", np.nan) if pd.notna(a.get("form")) and pd.notna(b.get("form")) else np.nan,
        "fatigue_rest_diff": (a.get("days_rest", np.nan) - b.get("days_rest", np.nan)) if pd.notna(a.get("days_rest")) and pd.notna(b.get("days_rest")) else np.nan,
        "fatigue_7d_diff": b.get("matches_7d", 0) - a.get("matches_7d", 0),
        "fatigue_14d_diff": b.get("matches_14d", 0) - a.get("matches_14d", 0),
        "games_last_diff": (b.get("games_last", np.nan) - a.get("games_last", np.nan)) if pd.notna(a.get("games_last")) and pd.notna(b.get("games_last")) else np.nan,
        "ace_rate_diff": a.get("ace_rate", np.nan) - b.get("ace_rate", np.nan) if pd.notna(a.get("ace_rate")) and pd.notna(b.get("ace_rate")) else np.nan,
        "first_serve_pct_diff": a.get("1st_pct", np.nan) - b.get("1st_pct", np.nan) if pd.notna(a.get("1st_pct")) and pd.notna(b.get("1st_pct")) else np.nan,
        "first_serve_won_diff": a.get("1st_won", np.nan) - b.get("1st_won", np.nan) if pd.notna(a.get("1st_won")) and pd.notna(b.get("1st_won")) else np.nan,
        "bp_saved_diff": a.get("bp_saved", np.nan) - b.get("bp_saved", np.nan) if pd.notna(a.get("bp_saved")) and pd.notna(b.get("bp_saved")) else np.nan,
        "round_num": ROUND_NUM.get(round_name, 3),
        "tourney_level_num": LEVEL_NUM.get(tourney_level, 1),
        "best_of": best_of,
        "surface_hard": 1 if surf == "Hard" else 0,
        "surface_clay": 1 if surf == "Clay" else 0,
        "surface_grass": 1 if surf == "Grass" else 0,
        "is_atp": 1 if is_atp else 0,
    }

    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    prob = model.predict_proba(X)[0, 1]
    return prob


def scan_markets(use_live=True):
    """Scan all active PM markets and compare model predictions to PM odds.

    Args:
        use_live: If True, fetch current data from SportRadar (recommended).
                  If False, use stale 2024 Sackmann data (for testing only).
    """
    print("Loading model...")
    model = load_model()

    print("Loading player map...")
    player_map = load_player_map()

    print("Loading historical player stats...")
    player_stats = _load_player_stats()

    if use_live:
        print("Fetching live data from SportRadar...")
        from model.live_features import load_sr_map, get_live_player_stats

        sr_map = load_sr_map()

        # Find which SR IDs we need
        pm_to_sr = {}
        for sr_id, info in sr_map.items():
            if info.get("pm_name"):
                pm_to_sr[info["pm_name"]] = sr_id

        # Collect SR IDs for active PM players
        conn_tmp = sqlite3.connect(str(DB_PATH))
        active_names = set()
        for row in conn_tmp.execute("""
            SELECT DISTINCT outcome_a FROM markets m JOIN events e ON e.id=m.event_id
            WHERE e.active=1 AND e.closed=0
            UNION
            SELECT DISTINCT outcome_b FROM markets m JOIN events e ON e.id=m.event_id
            WHERE e.active=1 AND e.closed=0
        """).fetchall():
            active_names.add(row[0])
        conn_tmp.close()

        # Only fetch players who are in active markets AND have a snapshot
        # (no point fetching for zero-liquidity markets)
        conn_tmp2 = sqlite3.connect(str(DB_PATH))
        active_with_odds = set()
        for row in conn_tmp2.execute("""
            SELECT DISTINCT m.outcome_a, m.outcome_b FROM markets m
            JOIN events e ON e.id = m.event_id
            JOIN snapshots s ON s.market_id = m.id
            WHERE e.active=1 AND e.closed=0
            AND s.price_a IS NOT NULL AND s.volume > 0
        """).fetchall():
            active_with_odds.add(row[0])
            active_with_odds.add(row[1])
        conn_tmp2.close()

        sr_ids_needed = list(set(pm_to_sr[n] for n in active_with_odds if n in pm_to_sr))
        print(f"  Fetching live stats for {len(sr_ids_needed)} players with active odds...")
        live_stats = get_live_player_stats(sr_ids_needed, sr_map)

        # Merge live stats into historical stats (live overrides where available)
        for sack_id, live in live_stats.items():
            if sack_id in player_stats:
                # Keep historical Elo, override everything else with live data
                hist = player_stats[sack_id]
                for key in ["rank", "points", "form", "days_rest", "matches_7d",
                            "matches_14d", "games_last", "ace_rate", "1st_pct", "1st_won"]:
                    if key in live and (pd.notna(live[key]) if not isinstance(live[key], (int, float)) else True):
                        hist[key] = live[key]
                # Map rank_points from SR 'points'
                if live.get("points"):
                    hist["rank_points"] = live["points"]
                if live.get("rank"):
                    hist["rank"] = live["rank"]
            else:
                # New player not in historical data — add with default Elo
                player_stats[sack_id] = {
                    "elo": ELO_START_RATING,
                    "surface_elo": {},
                    "rank": live.get("rank", np.nan),
                    "rank_points": live.get("points", np.nan),
                    "age": np.nan,
                    "height": np.nan,
                    "form": live.get("form", np.nan),
                    "days_rest": live.get("days_rest", np.nan),
                    "matches_7d": live.get("matches_7d", 0),
                    "matches_14d": live.get("matches_14d", 0),
                    "games_last": live.get("games_last", np.nan),
                    "ace_rate": live.get("ace_rate", np.nan),
                    "1st_pct": live.get("1st_pct", np.nan),
                    "1st_won": live.get("1st_won", np.nan),
                    "bp_saved": np.nan,
                    "_h2h": {},
                }

        print(f"  Live data merged for {len(live_stats)} players")

    print("Loading active markets from DB...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    markets = conn.execute("""
        SELECT e.id as event_id, e.title, e.series,
               m.id as market_id, m.outcome_a, m.outcome_b,
               s.price_a, s.price_b, s.volume, s.liquidity
        FROM events e
        JOIN markets m ON m.event_id = e.id
        LEFT JOIN snapshots s ON s.market_id = m.id
            AND s.scraped_at = (SELECT MAX(scraped_at) FROM snapshots WHERE market_id = m.id)
        WHERE e.active = 1 AND e.closed = 0
        ORDER BY s.volume DESC
    """).fetchall()
    conn.close()

    print(f"\nFound {len(markets)} active markets\n")

    edges = []
    skipped = 0

    for mkt in markets:
        name_a = mkt["outcome_a"]
        name_b = mkt["outcome_b"]
        pm_price_a = mkt["price_a"]
        pm_price_b = mkt["price_b"]

        # Look up players
        id_a = lookup(name_a, player_map)
        id_b = lookup(name_b, player_map)

        if not id_a or not id_b:
            skipped += 1
            continue

        if id_a not in player_stats or id_b not in player_stats:
            skipped += 1
            continue

        if pm_price_a is None or pm_price_b is None:
            skipped += 1
            continue

        # Skip low-liquidity markets — price is noise
        from config.settings import MIN_VOLUME
        if (mkt["volume"] or 0) < MIN_VOLUME:
            skipped += 1
            continue

        # Skip markets where either price is near 0 or 1 (effectively resolved)
        if pm_price_a < 0.02 or pm_price_b < 0.02 or pm_price_a > 0.98 or pm_price_b > 0.98:
            skipped += 1
            continue

        is_atp = mkt["series"] == "atp"

        # Predict
        prob_a = predict_match(
            model, id_a, id_b, player_stats,
            is_atp=is_atp,
        )

        if np.isnan(prob_a):
            skipped += 1
            continue

        prob_b = 1.0 - prob_a
        edge_a = prob_a - pm_price_a
        edge_b = prob_b - pm_price_b

        result = {
            "title": mkt["title"],
            "series": mkt["series"].upper(),
            "player_a": name_a,
            "player_b": name_b,
            "model_prob_a": prob_a,
            "model_prob_b": prob_b,
            "pm_price_a": pm_price_a,
            "pm_price_b": pm_price_b,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "volume": mkt["volume"] or 0,
        }

        max_edge = max(edge_a, edge_b)
        if max_edge >= MIN_EDGE_THRESHOLD:
            edges.append(result)

    # Print results
    print(f"Analyzed {len(markets) - skipped} markets (skipped {skipped})")
    print(f"Edge threshold: {MIN_EDGE_THRESHOLD:.0%}\n")

    if edges:
        edges.sort(key=lambda x: -max(x["edge_a"], x["edge_b"]))
        print(f"{'='*80}")
        print(f"MARKETS WITH EDGE >= {MIN_EDGE_THRESHOLD:.0%} ({len(edges)} found)")
        print(f"{'='*80}")
        for e in edges:
            best = "A" if e["edge_a"] > e["edge_b"] else "B"
            player = e[f"player_{best.lower()}"]
            edge = e[f"edge_{best.lower()}"]
            model_p = e[f"model_prob_{best.lower()}"]
            pm_p = e[f"pm_price_{best.lower()}"]
            print(f"\n  [{e['series']}] {e['title']}")
            print(f"  BET: {player}")
            print(f"    Model: {model_p:.1%}  |  PM: {pm_p:.1%}  |  Edge: {edge:+.1%}")
            print(f"    Volume: ${e['volume']:,.0f}")
    else:
        print("No edges found above threshold.")

    # Log paper trades
    if edges:
        _log_paper_trades(edges)

    return edges


def _log_paper_trades(edges: list[dict]) -> None:
    """Append edge signals to paper trading log for tracking."""
    from datetime import datetime, timezone
    from config.settings import PAPER_TRADES_PATH

    existing = []
    if PAPER_TRADES_PATH.exists():
        import json
        with open(PAPER_TRADES_PATH) as f:
            existing = json.load(f)

    timestamp = datetime.now(timezone.utc).isoformat()

    for e in edges:
        best = "a" if e["edge_a"] > e["edge_b"] else "b"
        entry = {
            "timestamp": timestamp,
            "title": e["title"],
            "series": e["series"],
            "bet_on": e[f"player_{best}"],
            "opponent": e[f"player_{'b' if best == 'a' else 'a'}"],
            "model_prob": round(e[f"model_prob_{best}"], 4),
            "pm_price": round(e[f"pm_price_{best}"], 4),
            "edge": round(e[f"edge_{best}"], 4),
            "volume": e["volume"],
            "outcome": None,  # filled in later when market resolves
        }
        existing.append(entry)

    import json
    PAPER_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAPER_TRADES_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nLogged {len(edges)} paper trades to {PAPER_TRADES_PATH}")


if __name__ == "__main__":
    import sys
    use_live = "--no-live" not in sys.argv
    if not use_live:
        print("Running without live SR data (using committed Elo state only)\n")
    scan_markets(use_live=use_live)
