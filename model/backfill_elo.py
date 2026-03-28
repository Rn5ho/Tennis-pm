"""Backfill Elo ratings from Jan 2025 to present using SportRadar data.

Fetches daily match results from SR, filters to ATP/WTA level,
maps SR IDs to Sackmann IDs, then extends the Elo computation
forward from where Sackmann data ends (Dec 2024).

Caches daily results to avoid re-fetching.
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    PROJECT_ROOT, ELO_START_RATING,
    ELO_K_INITIAL, ELO_K_MEDIUM, ELO_K_STABLE,
    ELO_SURFACE_K_MULTIPLIER,
    SACKMANN_ATP_DIR, SACKMANN_WTA_DIR,
)
from model.elo import compute_elo, expected_score, SURFACE_MAP
from model.tournament import lookup_surface_from_sr

logger = logging.getLogger(__name__)

CACHE_DIR = PROJECT_ROOT / "data" / "sr_daily_cache"
ELO_STATE_PATH = PROJECT_ROOT / "data" / "elo_state.json"

# ATP/WTA competition name patterns
ATP_WTA_PATTERNS = [
    "atp", "wta", "grand slam", "australian open", "roland garros",
    "wimbledon", "us open", "miami", "indian wells", "masters",
    "challenger", "open", "classic", "cup", "championship",
]

# Non-tennis or very low-level patterns to exclude
EXCLUDE_PATTERNS = ["utr ", "itf w15", "itf w25", "itf m15"]


def _is_atp_wta_level(competition_name: str) -> bool:
    """Check if a competition is ATP/WTA/Challenger level."""
    name_lower = competition_name.lower()
    # Exclude UTR and very low-level ITF
    for ex in EXCLUDE_PATTERNS:
        if ex in name_lower:
            return False
    # Include anything that looks ATP/WTA/Challenger
    for pat in ATP_WTA_PATTERNS:
        if pat in name_lower:
            return True
    return False


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[-'.(),]", " ", ascii_name)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _sr_name_to_normal(sr_name: str) -> str:
    if "," in sr_name:
        parts = sr_name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return sr_name


def fetch_daily_results(date_str: str) -> list[dict]:
    """Fetch and cache daily match results from SportRadar."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{date_str}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    from scraper.sportradar import get_daily_results
    results = get_daily_results(date_str)

    with open(cache_file, "w") as f:
        json.dump(results, f)

    return results


def build_sr_to_sackmann_map() -> dict:
    """Build SR competitor ID -> Sackmann player ID mapping.

    Uses the sr_player_map.json if available, plus fuzzy matching
    for any new players encountered in the backfill.
    """
    sr_map_path = PROJECT_ROOT / "data" / "sr_player_map.json"
    mapping = {}  # sr_id -> sackmann_id

    if sr_map_path.exists():
        with open(sr_map_path) as f:
            sr_map = json.load(f)
        for sr_id, info in sr_map.items():
            if info.get("sackmann_id"):
                mapping[sr_id] = info["sackmann_id"]

    # Also build name-based lookup from Sackmann data for players not in the map
    sackmann_players = {}
    for tour, data_dir in [("atp", SACKMANN_ATP_DIR), ("wta", SACKMANN_WTA_DIR)]:
        data_dir = Path(data_dir)
        for year in range(2015, 2025):
            for pattern in [f"{tour}_matches_{year}.csv", f"{tour}_matches_qual_chall_{year}.csv"]:
                f = data_dir / pattern
                if not f.exists():
                    continue
                df = pd.read_csv(f, usecols=["winner_id", "winner_name", "loser_id", "loser_name"])
                for _, row in df.iterrows():
                    sackmann_players[_normalize(row["winner_name"])] = str(int(row["winner_id"]))
                    sackmann_players[_normalize(row["loser_name"])] = str(int(row["loser_id"]))

    mapping["_sackmann_by_name"] = sackmann_players
    return mapping


def resolve_sr_id(sr_id: str, sr_name: str, mapping: dict) -> str | None:
    """Resolve a SportRadar ID to a Sackmann ID."""
    # Direct mapping
    if sr_id in mapping:
        return mapping[sr_id]

    # Name-based fallback
    normal = _normalize(_sr_name_to_normal(sr_name))
    sackmann_by_name = mapping.get("_sackmann_by_name", {})

    if normal in sackmann_by_name:
        sack_id = sackmann_by_name[normal]
        mapping[sr_id] = sack_id  # cache for next time
        return sack_id

    # Fuzzy match
    best_score = 0
    best_id = None
    for sack_norm, sack_id in sackmann_by_name.items():
        score = SequenceMatcher(None, normal, sack_norm).ratio()
        if score > best_score:
            best_score = score
            best_id = sack_id
    if best_score >= 0.88:
        mapping[sr_id] = best_id
        return best_id

    return None


def backfill(start_date: str = "2025-01-01", end_date: str = None):
    """Backfill match results and update Elo ratings.

    Args:
        start_date: YYYY-MM-DD, start of backfill period
        end_date: YYYY-MM-DD, end of backfill period (default: yesterday)
    """
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    total_days = (end - start).days + 1

    print(f"Backfilling {start_date} to {end_date} ({total_days} days)")

    # Build ID mapping
    print("Building SR -> Sackmann ID mapping...")
    mapping = build_sr_to_sackmann_map()

    # First, compute Elo from historical Sackmann data up to end of 2024
    print("Computing historical Elo from Sackmann data (2000-2024)...")
    from model.features import load_matches
    hist_matches = load_matches()
    hist_matches = compute_elo(hist_matches)

    # Extract final Elo state from historical data
    elo_overall = defaultdict(lambda: ELO_START_RATING)
    elo_surface = defaultdict(lambda: defaultdict(lambda: ELO_START_RATING))
    match_counts = defaultdict(int)
    surface_counts = defaultdict(lambda: defaultdict(int))

    for row in hist_matches.itertuples():
        wid, lid = str(row.winner_id), str(row.loser_id)
        surf = SURFACE_MAP.get(getattr(row, "surface", "Hard"), "Hard")

        # These are post-match Elo (we track pre-match in the features,
        # but we need the final state here)
        elo_overall[wid] = row.winner_elo
        elo_overall[lid] = row.loser_elo
        elo_surface[wid][surf] = row.winner_surface_elo
        elo_surface[lid][surf] = row.loser_surface_elo

        score = getattr(row, "score", "")
        is_wo = not isinstance(score, str) or bool(re.search(r"\bW/O\b|\bDEF\b", str(score)))
        if not is_wo:
            match_counts[wid] += 1
            match_counts[lid] += 1
            surface_counts[wid][surf] += 1
            surface_counts[lid][surf] += 1

    print(f"Historical Elo computed for {len(elo_overall)} players")

    # Track last match date per player (for zombie detection)
    last_match_date = {}  # sackmann_id -> date string

    # Now process SR daily results to extend Elo forward
    print(f"\nFetching daily results from SportRadar...")
    total_matches = 0
    resolved = 0
    unresolved = 0
    current = start

    while current <= end:
        date_str = current.isoformat()
        day_num = (current - start).days + 1

        if day_num % 30 == 0 or current == end:
            print(f"  [{day_num}/{total_days}] {date_str}...")

        try:
            results = fetch_daily_results(date_str)
        except Exception as e:
            logger.warning(f"Failed to fetch {date_str}: {e}")
            current += timedelta(days=1)
            continue

        for match in results:
            comp_name = match.get("competition", "")
            if not _is_atp_wta_level(comp_name):
                continue

            winner_id = match.get("winner_id")
            if not winner_id:
                continue

            home = match.get("home", {})
            away = match.get("away", {})

            # Determine winner and loser
            if winner_id == home.get("id"):
                w_sr, l_sr = home, away
            elif winner_id == away.get("id"):
                w_sr, l_sr = away, home
            else:
                continue

            # Resolve to Sackmann IDs
            w_sack = resolve_sr_id(w_sr["id"], w_sr["name"], mapping)
            l_sack = resolve_sr_id(l_sr["id"], l_sr["name"], mapping)

            if not w_sack or not l_sack:
                unresolved += 1
                continue

            total_matches += 1
            resolved += 1

            # Detect walkover from score
            score = match.get("score", "")
            if not score or "W/O" in score.upper():
                continue

            # Determine surface from tournament lookup (SR surface field is always empty)
            surf = lookup_surface_from_sr(comp_name)

            # Update Elo
            def _get_k(mc):
                if mc < 20:
                    return ELO_K_INITIAL
                elif mc < 50:
                    return ELO_K_MEDIUM
                return ELO_K_STABLE

            k_w = _get_k(match_counts[w_sack])
            k_l = _get_k(match_counts[l_sack])
            k = (k_w + k_l) / 2

            # Overall Elo update
            e = expected_score(elo_overall[w_sack], elo_overall[l_sack])
            elo_overall[w_sack] += k * (1.0 - e)
            elo_overall[l_sack] += k * (0.0 - (1.0 - e))
            match_counts[w_sack] += 1
            match_counts[l_sack] += 1
            last_match_date[w_sack] = date_str
            last_match_date[l_sack] = date_str

            # Surface Elo update
            sk_w = _get_k(surface_counts[w_sack][surf]) * ELO_SURFACE_K_MULTIPLIER
            sk_l = _get_k(surface_counts[l_sack][surf]) * ELO_SURFACE_K_MULTIPLIER
            sk = (sk_w + sk_l) / 2
            e_s = expected_score(elo_surface[w_sack][surf], elo_surface[l_sack][surf])
            elo_surface[w_sack][surf] += sk * (1.0 - e_s)
            elo_surface[l_sack][surf] += sk * (0.0 - (1.0 - e_s))
            surface_counts[w_sack][surf] += 1
            surface_counts[l_sack][surf] += 1

        current += timedelta(days=1)

    print(f"\nBackfill complete:")
    print(f"  Matches processed: {resolved}")
    print(f"  Unresolved IDs: {unresolved}")
    print(f"  Players with Elo: {len(elo_overall)}")

    # Reset Elo for inactive players (no match in 12+ months).
    # Retired players like Nadal/Raonic keep inflated ratings that compress
    # the active player scale. Resetting them to 1500 is safe because they
    # won't appear in PM markets.
    cutoff = (end - timedelta(days=365)).isoformat()
    zombie_count = 0
    for pid in list(elo_overall.keys()):
        player_last = last_match_date.get(pid)
        if player_last is None or player_last < cutoff:
            elo_overall[pid] = ELO_START_RATING
            for surf in elo_surface[pid]:
                elo_surface[pid][surf] = ELO_START_RATING
            zombie_count += 1
    print(f"  Reset {zombie_count} inactive players (no match since {cutoff})")

    # Save Elo state
    state = {
        "updated_through": end_date,
        "total_players": len(elo_overall),
        "matches_processed": resolved,
        "elo": {pid: rating for pid, rating in elo_overall.items()},
        "surface_elo": {pid: dict(surfaces) for pid, surfaces in elo_surface.items()},
        "match_counts": dict(match_counts),
    }

    ELO_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ELO_STATE_PATH, "w") as f:
        json.dump(state, f)

    print(f"Saved Elo state to {ELO_STATE_PATH}")

    # Show some top players' current Elo
    # Load player map for display names
    pm_map_path = PROJECT_ROOT / "data" / "player_map.json"
    sack_to_pm = {}
    if pm_map_path.exists():
        with open(pm_map_path) as f:
            pm_map = json.load(f)
        for pm_name, info in pm_map.items():
            if info.get("sackmann_id"):
                sack_to_pm[info["sackmann_id"]] = pm_name

    print("\nTop 20 Elo ratings (current):")
    sorted_elo = sorted(elo_overall.items(), key=lambda x: -x[1])
    for i, (pid, elo) in enumerate(sorted_elo[:20]):
        name = sack_to_pm.get(pid, f"ID:{pid}")
        surf_str = ", ".join(f"{s}:{r:.0f}" for s, r in sorted(elo_surface[pid].items()))
        print(f"  {i+1:>3}. {name:<30} Elo: {elo:.0f}  ({surf_str})")

    return state


def load_elo_state() -> dict | None:
    """Load saved Elo state."""
    if ELO_STATE_PATH.exists():
        with open(ELO_STATE_PATH) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill()
