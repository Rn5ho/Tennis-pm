"""Feature engineering for tennis match prediction.

Builds a feature matrix from Sackmann match data + Elo ratings.
All features are difference-based (player_a - player_b) to handle symmetry.
Rolling features use only past data to prevent leakage.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import (
    SACKMANN_ATP_DIR,
    SACKMANN_WTA_DIR,
    MATCH_YEARS,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
    ELO_BURN_IN_YEARS,
    ROLLING_WINDOW,
    RECENT_FORM_WINDOW,
)
from model.elo import compute_elo, SURFACE_MAP

ROUND_NUM = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 3, "ER": 0, "BR": 6,
}

LEVEL_NUM = {
    "C": 0,   # Challenger
    "S": 0,   # Satellite (old)
    "A": 1,   # ATP 250/500
    "I": 1,   # International (WTA equivalent)
    "P": 1,   # Premier (WTA)
    "PM": 2,  # Premier Mandatory (WTA)
    "M": 2,   # Masters 1000
    "G": 3,   # Grand Slam
    "F": 3,   # Tour Finals
    "D": 1,   # Davis/Fed Cup
    "O": 2,   # Olympics
}


def load_matches() -> pd.DataFrame:
    """Load all ATP + WTA match data for the configured year range."""
    frames = []

    for tour, data_dir in [("atp", SACKMANN_ATP_DIR), ("wta", SACKMANN_WTA_DIR)]:
        data_dir = Path(data_dir)
        prefix = tour

        for year in range(MATCH_YEARS[0], MATCH_YEARS[1] + 1):
            # Tour-level matches
            f = data_dir / f"{prefix}_matches_{year}.csv"
            if f.exists():
                df = pd.read_csv(f, low_memory=False)
                df["tour"] = tour
                frames.append(df)

            # Challenger/qualifying matches
            fc = data_dir / f"{prefix}_matches_qual_chall_{year}.csv"
            if fc.exists():
                df = pd.read_csv(fc, low_memory=False)
                df["tour"] = tour
                frames.append(df)

    matches = pd.concat(frames, ignore_index=True)

    # Parse tourney_date as int (YYYYMMDD format in Sackmann)
    matches["tourney_date"] = pd.to_numeric(matches["tourney_date"], errors="coerce")
    matches = matches.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    matches["tourney_date"] = matches["tourney_date"].astype(int)
    matches["year"] = matches["tourney_date"] // 10000

    # Ensure IDs are strings for consistent hashing
    matches["winner_id"] = matches["winner_id"].astype(str)
    matches["loser_id"] = matches["loser_id"].astype(str)

    return matches


def _is_walkover_or_retirement(score: str) -> bool:
    if not isinstance(score, str):
        return True
    return bool(re.search(r"\bW/O\b|\bDEF\b|\bRET\b|\bWalkover\b", score, re.IGNORECASE))


def _compute_h2h(matches: pd.DataFrame) -> dict:
    """Compute running H2H records for all player pairs.

    Returns dict mapping (idx) -> (h2h_a_wins, h2h_b_wins) where
    a=winner, b=loser at that point in time (pre-match).
    """
    # Track cumulative wins: h2h[(p1,p2)] = wins by p1 vs p2
    h2h = defaultdict(int)
    result = {}

    for row in matches.itertuples():
        wid, lid = row.winner_id, row.loser_id
        # Pre-match H2H
        result[row.Index] = (h2h[(wid, lid)], h2h[(lid, wid)])
        # Update
        if not _is_walkover_or_retirement(row.score):
            h2h[(wid, lid)] += 1

    return result


def _compute_rolling_stats(matches: pd.DataFrame) -> dict:
    """Compute rolling serve stats and form for each player.

    Returns dict mapping idx -> {
        'w_form': float, 'l_form': float,
        'w_ace_rate': float, 'l_ace_rate': float,
        'w_1st_pct': float, 'l_1st_pct': float,
        'w_1st_won': float, 'l_1st_won': float,
        'w_bp_saved': float, 'l_bp_saved': float,
    }
    """
    # Per-player rolling history
    player_results = defaultdict(list)   # list of 0/1 (loss/win)
    player_serve = defaultdict(list)     # list of serve stat dicts

    result = {}

    for row in matches.itertuples():
        wid, lid = row.winner_id, row.loser_id
        stats = {}

        # Recent form (win rate in last N matches)
        w_hist = player_results[wid]
        l_hist = player_results[lid]
        stats["w_form"] = np.mean(w_hist[-RECENT_FORM_WINDOW:]) if w_hist else np.nan
        stats["l_form"] = np.mean(l_hist[-RECENT_FORM_WINDOW:]) if l_hist else np.nan

        # Rolling serve stats
        for prefix, pid in [("w", wid), ("l", lid)]:
            srv = player_serve[pid][-ROLLING_WINDOW:]
            if srv:
                total_svpt = sum(s.get("svpt", 0) for s in srv)
                total_ace = sum(s.get("ace", 0) for s in srv)
                total_1stIn = sum(s.get("1stIn", 0) for s in srv)
                total_1stWon = sum(s.get("1stWon", 0) for s in srv)
                total_bpSaved = sum(s.get("bpSaved", 0) for s in srv)
                total_bpFaced = sum(s.get("bpFaced", 0) for s in srv)

                stats[f"{prefix}_ace_rate"] = total_ace / total_svpt if total_svpt > 0 else np.nan
                stats[f"{prefix}_1st_pct"] = total_1stIn / total_svpt if total_svpt > 0 else np.nan
                stats[f"{prefix}_1st_won"] = total_1stWon / total_1stIn if total_1stIn > 0 else np.nan
                stats[f"{prefix}_bp_saved"] = total_bpSaved / total_bpFaced if total_bpFaced > 0 else np.nan
            else:
                stats[f"{prefix}_ace_rate"] = np.nan
                stats[f"{prefix}_1st_pct"] = np.nan
                stats[f"{prefix}_1st_won"] = np.nan
                stats[f"{prefix}_bp_saved"] = np.nan

        result[row.Index] = stats

        # Update histories (after recording pre-match stats)
        if not _is_walkover_or_retirement(row.score):
            player_results[wid].append(1)
            player_results[lid].append(0)

            # Winner serve stats
            w_svpt = row.w_svpt if pd.notna(getattr(row, "w_svpt", None)) else 0
            if w_svpt > 0:
                player_serve[wid].append({
                    "svpt": w_svpt,
                    "ace": getattr(row, "w_ace", 0) or 0,
                    "1stIn": getattr(row, "w_1stIn", 0) or 0,
                    "1stWon": getattr(row, "w_1stWon", 0) or 0,
                    "bpSaved": getattr(row, "w_bpSaved", 0) or 0,
                    "bpFaced": getattr(row, "w_bpFaced", 0) or 0,
                })

            # Loser serve stats
            l_svpt = row.l_svpt if pd.notna(getattr(row, "l_svpt", None)) else 0
            if l_svpt > 0:
                player_serve[lid].append({
                    "svpt": l_svpt,
                    "ace": getattr(row, "l_ace", 0) or 0,
                    "1stIn": getattr(row, "l_1stIn", 0) or 0,
                    "1stWon": getattr(row, "l_1stWon", 0) or 0,
                    "bpSaved": getattr(row, "l_bpSaved", 0) or 0,
                    "bpFaced": getattr(row, "l_bpFaced", 0) or 0,
                })

    return result


def _compute_fatigue(matches: pd.DataFrame) -> dict:
    """Compute fatigue features: days since last match, matches in last 7/14 days.

    Returns dict mapping idx -> {
        'w_days_rest': float, 'l_days_rest': float,
        'w_matches_7d': int, 'l_matches_7d': int,
        'w_matches_14d': int, 'l_matches_14d': int,
        'w_games_last_match': float, 'l_games_last_match': float,
    }
    """
    # Track each player's match dates and game counts
    player_dates = defaultdict(list)      # list of date ints (YYYYMMDD)
    player_games = defaultdict(list)      # total games in last match

    result = {}

    for row in matches.itertuples():
        wid, lid = row.winner_id, row.loser_id
        date = row.tourney_date
        stats = {}

        for prefix, pid in [("w", wid), ("l", lid)]:
            dates = player_dates[pid]
            if dates:
                last = dates[-1]
                # Convert YYYYMMDD to rough days (approximate, good enough)
                days_rest = _date_diff(last, date)
                stats[f"{prefix}_days_rest"] = days_rest

                # Matches in last 7/14 days
                m7 = sum(1 for d in dates if _date_diff(d, date) <= 7)
                m14 = sum(1 for d in dates if _date_diff(d, date) <= 14)
                stats[f"{prefix}_matches_7d"] = m7
                stats[f"{prefix}_matches_14d"] = m14
            else:
                stats[f"{prefix}_days_rest"] = np.nan
                stats[f"{prefix}_matches_7d"] = 0
                stats[f"{prefix}_matches_14d"] = 0

            # Games in last match (proxy for physical toll)
            games = player_games[pid]
            stats[f"{prefix}_games_last"] = games[-1] if games else np.nan

        result[row.Index] = stats

        # Update (after recording pre-match)
        player_dates[wid].append(date)
        player_dates[lid].append(date)

        # Count total games from score
        total_games = _count_games(row.score)
        player_games[wid].append(total_games)
        player_games[lid].append(total_games)

    return result


def _date_diff(d1: int, d2: int) -> int:
    """Approximate day difference between two YYYYMMDD ints."""
    y1, m1, day1 = d1 // 10000, (d1 % 10000) // 100, d1 % 100
    y2, m2, day2 = d2 // 10000, (d2 % 10000) // 100, d2 % 100
    return (y2 - y1) * 365 + (m2 - m1) * 30 + (day2 - day1)


def _count_games(score) -> float:
    """Count total games from a score string like '6-4 7-6(5) 6-3'."""
    if not isinstance(score, str):
        return np.nan
    total = 0
    for s in score.split():
        m = re.match(r"(\d+)-(\d+)", s)
        if m:
            total += int(m.group(1)) + int(m.group(2))
    return total if total > 0 else np.nan


def build_feature_matrix(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the full feature matrix from raw matches.

    Returns:
        X: Feature DataFrame
        y: Binary target (1 = player_a won)
        meta: Metadata (year, player IDs, etc.) for splitting/analysis
    """
    print("Computing Elo ratings...")
    matches = compute_elo(matches)

    # Filter out walkovers (keep retirements for now — they have a real winner)
    mask_wo = matches["score"].apply(
        lambda s: bool(re.search(r"\bW/O\b|\bDEF\b|\bWalkover\b", s, re.IGNORECASE))
        if isinstance(s, str) else True
    )
    matches = matches[~mask_wo].copy()
    matches = matches.reset_index(drop=True)

    print(f"Matches after removing walkovers: {len(matches)}")

    print("Computing H2H records...")
    h2h = _compute_h2h(matches)

    print("Computing rolling serve stats + form...")
    rolling = _compute_rolling_stats(matches)

    print("Computing fatigue features...")
    fatigue = _compute_fatigue(matches)

    print("Building feature matrix...")

    # Randomly assign winner/loser to player_a/player_b slots
    rng = np.random.RandomState(42)
    swap = rng.random(len(matches)) > 0.5

    rows = []
    targets = []
    meta_rows = []

    for i, row in enumerate(matches.itertuples()):
        idx = row.Index
        s = swap[i]

        # Determine player_a and player_b
        if s:
            # player_a = loser, player_b = winner -> target = 0
            target = 0
            a_elo, b_elo = row.loser_elo, row.winner_elo
            a_surf, b_surf = row.loser_surface_elo, row.winner_surface_elo
            a_rank = getattr(row, "loser_rank", np.nan)
            b_rank = getattr(row, "winner_rank", np.nan)
            a_rp = getattr(row, "loser_rank_points", np.nan)
            b_rp = getattr(row, "winner_rank_points", np.nan)
            a_age = getattr(row, "loser_age", np.nan)
            b_age = getattr(row, "winner_age", np.nan)
            a_ht = getattr(row, "loser_ht", np.nan)
            b_ht = getattr(row, "winner_ht", np.nan)
            h2h_a, h2h_b = h2h[idx][1], h2h[idx][0]  # reversed
            r = rolling[idx]
            a_form, b_form = r["l_form"], r["w_form"]
            a_ace, b_ace = r["l_ace_rate"], r["w_ace_rate"]
            a_1st, b_1st = r["l_1st_pct"], r["w_1st_pct"]
            a_1stw, b_1stw = r["l_1st_won"], r["w_1st_won"]
            a_bp, b_bp = r["l_bp_saved"], r["w_bp_saved"]
            f = fatigue[idx]
            a_rest, b_rest = f["l_days_rest"], f["w_days_rest"]
            a_m7, b_m7 = f["l_matches_7d"], f["w_matches_7d"]
            a_m14, b_m14 = f["l_matches_14d"], f["w_matches_14d"]
            a_gl, b_gl = f["l_games_last"], f["w_games_last"]
            a_id, b_id = row.loser_id, row.winner_id
        else:
            # player_a = winner, player_b = loser -> target = 1
            target = 1
            a_elo, b_elo = row.winner_elo, row.loser_elo
            a_surf, b_surf = row.winner_surface_elo, row.loser_surface_elo
            a_rank = getattr(row, "winner_rank", np.nan)
            b_rank = getattr(row, "loser_rank", np.nan)
            a_rp = getattr(row, "winner_rank_points", np.nan)
            b_rp = getattr(row, "loser_rank_points", np.nan)
            a_age = getattr(row, "winner_age", np.nan)
            b_age = getattr(row, "loser_age", np.nan)
            a_ht = getattr(row, "winner_ht", np.nan)
            b_ht = getattr(row, "loser_ht", np.nan)
            h2h_a, h2h_b = h2h[idx]
            r = rolling[idx]
            a_form, b_form = r["w_form"], r["l_form"]
            a_ace, b_ace = r["w_ace_rate"], r["l_ace_rate"]
            a_1st, b_1st = r["w_1st_pct"], r["l_1st_pct"]
            a_1stw, b_1stw = r["w_1st_won"], r["l_1st_won"]
            a_bp, b_bp = r["w_bp_saved"], r["l_bp_saved"]
            f = fatigue[idx]
            a_rest, b_rest = f["w_days_rest"], f["l_days_rest"]
            a_m7, b_m7 = f["w_matches_7d"], f["l_matches_7d"]
            a_m14, b_m14 = f["w_matches_14d"], f["l_matches_14d"]
            a_gl, b_gl = f["w_games_last"], f["l_games_last"]
            a_id, b_id = row.winner_id, row.loser_id

        # Safe float conversion for rank (lower rank = better, so b - a)
        a_rank_f = float(a_rank) if pd.notna(a_rank) else np.nan
        b_rank_f = float(b_rank) if pd.notna(b_rank) else np.nan
        a_rp_f = float(a_rp) if pd.notna(a_rp) else np.nan
        b_rp_f = float(b_rp) if pd.notna(b_rp) else np.nan

        surface = SURFACE_MAP.get(row.surface, "Hard")

        feat = {
            "elo_diff": a_elo - b_elo,
            "surface_elo_diff": a_surf - b_surf,
            "rank_diff": b_rank_f - a_rank_f,  # positive = a is ranked higher (better)
            "rank_points_diff": a_rp_f - b_rp_f,
            "age_diff": (float(a_age) if pd.notna(a_age) else np.nan) -
                        (float(b_age) if pd.notna(b_age) else np.nan),
            "height_diff": (float(a_ht) if pd.notna(a_ht) else np.nan) -
                           (float(b_ht) if pd.notna(b_ht) else np.nan),
            "h2h_diff": h2h_a - h2h_b,
            "form_diff": a_form - b_form if pd.notna(a_form) and pd.notna(b_form) else np.nan,
            "fatigue_rest_diff": (a_rest - b_rest) if pd.notna(a_rest) and pd.notna(b_rest) else np.nan,
            "fatigue_7d_diff": b_m7 - a_m7,   # positive = b played more recently (more tired)
            "fatigue_14d_diff": b_m14 - a_m14,
            "games_last_diff": (b_gl - a_gl) if pd.notna(a_gl) and pd.notna(b_gl) else np.nan,
            "ace_rate_diff": a_ace - b_ace if pd.notna(a_ace) and pd.notna(b_ace) else np.nan,
            "first_serve_pct_diff": a_1st - b_1st if pd.notna(a_1st) and pd.notna(b_1st) else np.nan,
            "first_serve_won_diff": a_1stw - b_1stw if pd.notna(a_1stw) and pd.notna(b_1stw) else np.nan,
            "bp_saved_diff": a_bp - b_bp if pd.notna(a_bp) and pd.notna(b_bp) else np.nan,
            "round_num": ROUND_NUM.get(row.round, 3),
            "tourney_level_num": LEVEL_NUM.get(getattr(row, "tourney_level", "A"), 1),
            "best_of": int(row.best_of) if pd.notna(getattr(row, "best_of", None)) else 3,
            "surface_hard": 1 if surface == "Hard" else 0,
            "surface_clay": 1 if surface == "Clay" else 0,
            "surface_grass": 1 if surface == "Grass" else 0,
            "is_atp": 1 if row.tour == "atp" else 0,
        }

        rows.append(feat)
        targets.append(target)
        meta_rows.append({
            "year": row.year,
            "player_a_id": a_id,
            "player_b_id": b_id,
            "tourney_date": row.tourney_date,
            "tour": row.tour,
        })

    X = pd.DataFrame(rows)
    y = pd.Series(targets, name="target")
    meta = pd.DataFrame(meta_rows)

    print(f"Feature matrix: {X.shape[0]} matches, {X.shape[1]} features")
    return X, y, meta


FEATURE_NAMES = [
    "elo_diff", "surface_elo_diff", "rank_diff", "rank_points_diff",
    "age_diff", "height_diff", "h2h_diff", "form_diff",
    "fatigue_rest_diff", "fatigue_7d_diff", "fatigue_14d_diff", "games_last_diff",
    "ace_rate_diff", "first_serve_pct_diff", "first_serve_won_diff", "bp_saved_diff",
    "round_num", "tourney_level_num", "best_of",
    "surface_hard", "surface_clay", "surface_grass", "is_atp",
]
