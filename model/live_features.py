"""Build live features from SportRadar data for current PM markets.

Bridges SportRadar (current data) with Sackmann (historical Elo)
to produce up-to-date feature vectors for prediction.
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    PROJECT_ROOT, DB_PATH, MODEL_DIR,
    ROLLING_WINDOW, RECENT_FORM_WINDOW,
    ELO_START_RATING,
)
from model.elo import expected_score, SURFACE_MAP
from model.features import FEATURE_NAMES, ROUND_NUM, LEVEL_NUM
from model.name_match import load_player_map
from scraper.sportradar import get_rankings, get_competitor_results

logger = logging.getLogger(__name__)

SR_MAP_PATH = PROJECT_ROOT / "data" / "sr_player_map.json"
SR_STATS_CACHE_PATH = PROJECT_ROOT / "data" / "sr_stats_cache.json"
SR_STATS_CACHE_TTL_HOURS = 18  # re-fetch after this many hours
SR_MAX_API_CALLS_PER_RUN = 20  # cap cold fetches per run to avoid trial rate limits


def _normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[-'.(),]", " ", ascii_name)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _sr_name_to_normal(sr_name: str) -> str:
    """Convert 'Last, First' to 'First Last'."""
    if "," in sr_name:
        parts = sr_name.split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return sr_name


def build_sr_map(rankings: dict = None) -> dict:
    """Build SportRadar ID -> player info map, linked to Sackmann IDs.

    Returns and saves: {sr_id: {name, rank, points, sackmann_id, pm_name}}
    """
    if rankings is None:
        rankings = get_rankings()

    player_map = load_player_map()

    # Build reverse lookup: normalized sackmann_name -> pm_name
    sackmann_to_pm = {}
    for pm_name, info in player_map.items():
        if info.get("verified") and info.get("sackmann_name"):
            sack_norm = _normalize(info["sackmann_name"])
            sackmann_to_pm[sack_norm] = {
                "pm_name": pm_name,
                "sackmann_id": info["sackmann_id"],
                "sackmann_name": info["sackmann_name"],
            }

    sr_map = {}
    matched = 0

    for tour_key in ["atp", "wta"]:
        for entry in rankings.get(tour_key, []):
            sr_id = entry["sr_id"]
            sr_name = entry["name"]
            normal_name = _sr_name_to_normal(sr_name)
            norm = _normalize(normal_name)

            info = {
                "sr_name": sr_name,
                "name": normal_name,
                "rank": entry["rank"],
                "points": entry["points"],
                "tour": tour_key,
                "sackmann_id": None,
                "pm_name": None,
            }

            # Try exact normalized match
            if norm in sackmann_to_pm:
                link = sackmann_to_pm[norm]
                info["sackmann_id"] = link["sackmann_id"]
                info["pm_name"] = link["pm_name"]
                matched += 1
            else:
                # Fuzzy match
                best_score = 0
                best_key = None
                for sack_norm in sackmann_to_pm:
                    score = SequenceMatcher(None, norm, sack_norm).ratio()
                    if score > best_score:
                        best_score = score
                        best_key = sack_norm
                if best_score >= 0.85 and best_key:
                    link = sackmann_to_pm[best_key]
                    info["sackmann_id"] = link["sackmann_id"]
                    info["pm_name"] = link["pm_name"]
                    matched += 1

            sr_map[sr_id] = info

    # Save
    SR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SR_MAP_PATH, "w") as f:
        json.dump(sr_map, f, indent=2)

    total = len(sr_map)
    logger.info(f"SR map: {matched}/{total} linked to Sackmann ({matched/total*100:.1f}%)")
    return sr_map


def load_sr_map() -> dict:
    if SR_MAP_PATH.exists():
        with open(SR_MAP_PATH) as f:
            return json.load(f)
    return build_sr_map()


def _load_stats_cache() -> dict:
    """Load cached SR player results. Returns {sr_id: {fetched_at, matches}}."""
    if SR_STATS_CACHE_PATH.exists():
        try:
            with open(SR_STATS_CACHE_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_stats_cache(cache: dict) -> None:
    SR_STATS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SR_STATS_CACHE_PATH, "w") as f:
        json.dump(cache, f)


def _is_cache_fresh(entry: dict) -> bool:
    """Check if a cached entry is still within TTL."""
    from datetime import datetime, timezone
    fetched = entry.get("fetched_at", "")
    if not fetched:
        return False
    try:
        fetched_dt = datetime.fromisoformat(fetched)
        age_hours = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 3600
        return age_hours < SR_STATS_CACHE_TTL_HOURS
    except (ValueError, TypeError):
        return False


def get_live_player_stats(sr_ids: list[str], sr_map: dict = None) -> dict:
    """Fetch current stats for a list of SR player IDs.

    Uses SR rankings for rank/points and competitor results for
    recent form, serve stats, and fatigue.

    Results are cached per-player for SR_STATS_CACHE_TTL_HOURS to
    preserve API quota. Cached data is reused across predictor runs.

    Returns: {sackmann_id: {features dict}}
    """
    from datetime import datetime, timezone

    if sr_map is None:
        sr_map = load_sr_map()

    cache = _load_stats_cache()
    stats = {}
    api_calls = 0
    cache_hits = 0
    api_stopped = False  # stop making API calls after rate limit or cap

    for sr_id in sr_ids:
        info = sr_map.get(sr_id, {})
        sack_id = info.get("sackmann_id")
        if not sack_id:
            continue

        # Check cache first
        cached = cache.get(sr_id)
        if cached and _is_cache_fresh(cached):
            matches = cached["matches"]
            cache_hits += 1
        elif api_stopped or api_calls >= SR_MAX_API_CALLS_PER_RUN:
            # Rate limited or hit cap — use stale cache or skip (next run picks up)
            if cached:
                matches = cached["matches"]
                cache_hits += 1
            else:
                continue
        else:
            # Fetch from SR API
            try:
                matches = get_competitor_results(sr_id)
                api_calls += 1
            except Exception as e:
                logger.warning(f"Failed to fetch results for {sr_id}: {e}")
                if "429" in str(e):
                    api_stopped = True
                    logger.warning("Rate limited — stopping API calls for this run (%d fetched)", api_calls)
                # Fall back to stale cache if available
                if cached:
                    matches = cached["matches"]
                    cache_hits += 1
                else:
                    continue

            # Update cache
            cache[sr_id] = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "matches": matches,
            }

        # Compute features from recent matches
        rank = info.get("rank")
        points = info.get("points")

        # Recent form
        results = [1 if m["won"] else 0 for m in matches]
        form = np.mean(results[-RECENT_FORM_WINDOW:]) if results else np.nan

        # Fatigue
        if matches:
            from datetime import datetime
            today = datetime.now()
            dates = []
            for m in matches:
                try:
                    d = datetime.strptime(m["date"], "%Y-%m-%d")
                    dates.append(d)
                except (ValueError, TypeError):
                    pass

            if dates:
                days_rest = (today - dates[0]).days
                m7 = sum(1 for d in dates if (today - d).days <= 7)
                m14 = sum(1 for d in dates if (today - d).days <= 14)
            else:
                days_rest = np.nan
                m7 = 0
                m14 = 0

            # Games in last match
            last_score = matches[0].get("score", "")
            games_last = 0
            for part in last_score.split():
                m_score = re.match(r"(\d+)-(\d+)", part)
                if m_score:
                    games_last += int(m_score.group(1)) + int(m_score.group(2))
            games_last = games_last if games_last > 0 else np.nan
        else:
            days_rest = np.nan
            m7 = 0
            m14 = 0
            games_last = np.nan

        # Rolling serve stats from match stats
        serve_matches = [m for m in matches if m.get("stats")][-ROLLING_WINDOW:]
        total_svpt = 0
        total_ace = 0
        total_1stIn = 0
        total_1stWon = 0
        total_bp_won = 0  # breakpoints saved = service games - breakpoints lost
        total_svc_games = 0

        for m in serve_matches:
            st = m["stats"]
            svpt = st.get("service_points_won", 0) + st.get("service_points_lost", 0)
            total_svpt += svpt
            total_ace += st.get("aces", 0)
            total_1stIn += st.get("first_serve_successful", 0)
            total_1stWon += st.get("first_serve_points_won", 0)
            # BP saved: approximate from service_games_won / total (SR doesn't give bp_saved directly)
            total_svc_games += st.get("service_games_won", 0)

        ace_rate = total_ace / total_svpt if total_svpt > 0 else np.nan
        first_pct = total_1stIn / total_svpt if total_svpt > 0 else np.nan
        first_won = total_1stWon / total_1stIn if total_1stIn > 0 else np.nan

        # Elo — we'll use the historical Elo as a base and won't recompute here
        # The predictor will combine this with historical Elo

        stats[sack_id] = {
            "sr_id": sr_id,
            "rank": rank,
            "points": points,
            "form": form,
            "days_rest": days_rest,
            "matches_7d": m7,
            "matches_14d": m14,
            "games_last": games_last,
            "ace_rate": ace_rate,
            "1st_pct": first_pct,
            "1st_won": first_won,
            "bp_saved": np.nan,  # SR trial doesn't expose bp_saved cleanly
            "n_recent_matches": len(matches),
        }

    # Persist cache
    _save_stats_cache(cache)
    logger.info(f"SR stats: {api_calls} API calls, {cache_hits} cache hits")

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Building SR player map...")
    sr_map = build_sr_map()

    linked = sum(1 for v in sr_map.values() if v.get("sackmann_id"))
    print(f"\nLinked {linked}/{len(sr_map)} SR players to Sackmann IDs")

    # Show some linked players
    print("\nSample linked players:")
    count = 0
    for sr_id, info in sr_map.items():
        if info.get("pm_name") and count < 10:
            print(f"  SR: {info['name']} (#{info['rank']}) -> Sackmann: {info['sackmann_id']} -> PM: {info['pm_name']}")
            count += 1
