"""SportRadar Tennis API integration.

Pulls current rankings, player profiles, and recent match results
to keep Elo ratings and features up to date.

Rate limit: trial tier is ~1 req/sec. We sleep between calls.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("SPORTRADAR_API_KEY", "")
BASE_URL = "https://api.sportradar.com/tennis/trial/v3/en"
CACHE_DIR = PROJECT_ROOT / "data" / "sportradar_cache"
REQUEST_DELAY = 5.0  # seconds between requests (trial burst limit needs >3s for sustained calls)


def _load_env():
    """Load API key from .env if not in environment."""
    global API_KEY
    if API_KEY:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("SPORTRADAR_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                return
    raise ValueError("SPORTRADAR_API_KEY not found in environment or .env file")


def _fetch(endpoint: str) -> dict:
    """Fetch from SportRadar API with rate limiting and caching."""
    _load_env()
    url = f"{BASE_URL}{endpoint}"
    separator = "&" if "?" in endpoint else "?"
    url = f"{url}{separator}api_key={API_KEY}"

    req = Request(url, headers={"User-Agent": "Tennis-PM/0.1"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        time.sleep(REQUEST_DELAY)
        return data
    except HTTPError as e:
        if e.code == 429:
            # Single retry after 30s wait — if it fails again, propagate immediately
            # so the caller can stop making further requests this run
            if not getattr(_fetch, '_retrying', False):
                _fetch._retrying = True
                logger.warning("Rate limited on %s, waiting 30s and retrying once...", endpoint)
                time.sleep(30)
                try:
                    result = _fetch(endpoint)
                    _fetch._retrying = False
                    return result
                except HTTPError:
                    _fetch._retrying = False
                    raise
            raise
        logger.error("HTTP %d fetching %s", e.code, endpoint)
        raise


def get_rankings() -> dict:
    """Get current ATP and WTA rankings.

    Returns:
        {
            "atp": [{"id": "sr:competitor:...", "name": "...", "rank": int, "points": int}, ...],
            "wta": [...]
        }
    """
    data = _fetch("/rankings.json")
    result = {"atp": [], "wta": []}

    for ranking in data.get("rankings", []):
        gender = ranking.get("gender", "")
        key = "atp" if gender == "men" else "wta"

        for entry in ranking.get("competitor_rankings", []):
            comp = entry.get("competitor", {})
            result[key].append({
                "sr_id": comp.get("id", ""),
                "name": comp.get("name", ""),
                "country": comp.get("country_code", ""),
                "rank": entry.get("rank"),
                "points": entry.get("points"),
            })

    return result


def get_competitor_profile(sr_id: str) -> dict:
    """Get a player's profile (height, DOB, handedness, etc.)."""
    data = _fetch(f"/competitors/{sr_id}/profile.json")
    comp = data.get("competitor", {})
    info = data.get("info", {})

    return {
        "sr_id": sr_id,
        "name": comp.get("name", ""),
        "country": comp.get("country_code", ""),
        "height": info.get("height"),
        "weight": info.get("weight"),
        "handedness": info.get("handedness"),
        "date_of_birth": info.get("date_of_birth"),
        "pro_year": info.get("pro_year"),
    }


def get_competitor_results(sr_id: str) -> list[dict]:
    """Get a player's recent match results (up to 30 most recent).

    Returns list of match dicts with:
        - date, competition, round, surface
        - opponent_id, opponent_name
        - won (bool), score
        - serve stats (aces, 1st serve %, etc.)
    """
    data = _fetch(f"/competitors/{sr_id}/summaries.json")
    matches = []

    for s in data.get("summaries", []):
        ev = s.get("sport_event", {})
        status = s.get("sport_event_status", {})

        # Only completed matches
        if status.get("status") not in ("closed", "ended"):
            continue

        context = ev.get("sport_event_context", {})
        competition = context.get("competition", {})
        round_info = context.get("round", {})

        # Find this player and opponent
        comps = ev.get("competitors", [])
        if len(comps) != 2:
            continue

        player_comp = None
        opponent_comp = None
        for c in comps:
            if c.get("id") == sr_id:
                player_comp = c
            else:
                opponent_comp = c

        if not player_comp or not opponent_comp:
            continue

        winner_id = status.get("winner_id")
        won = winner_id == sr_id

        # Score
        period_scores = status.get("period_scores", [])
        score_parts = []
        for ps in period_scores:
            h, a = ps.get("home_score", 0), ps.get("away_score", 0)
            # Flip if player is away
            if player_comp.get("qualifier") == "away":
                h, a = a, h
            score_parts.append(f"{h}-{a}")
        score_str = " ".join(score_parts)

        # Serve stats from statistics
        player_stats = {}
        totals = s.get("statistics", {}).get("totals", {})
        for cs in totals.get("competitors", []):
            if cs.get("id") == sr_id:
                st = cs.get("statistics", {})
                player_stats = {
                    "aces": st.get("aces", 0),
                    "double_faults": st.get("double_faults", 0),
                    "first_serve_successful": st.get("first_serve_successful", 0),
                    "second_serve_successful": st.get("second_serve_successful", 0),
                    "first_serve_points_won": st.get("first_serve_points_won", 0),
                    "second_serve_points_won": st.get("second_serve_points_won", 0),
                    "service_points_won": st.get("service_points_won", 0),
                    "service_points_lost": st.get("service_points_lost", 0),
                    "breakpoints_won": st.get("breakpoints_won", 0),
                    "service_games_won": st.get("service_games_won", 0),
                    "tiebreaks_won": st.get("tiebreaks_won", 0),
                    "games_won": st.get("games_won", 0),
                    "points_won": st.get("points_won", 0),
                }
                break

        # Surface from venue or competition
        venue = ev.get("venue", {})
        surface = context.get("competition", {}).get("surface", "")

        matches.append({
            "date": ev.get("start_time", "")[:10],
            "competition": competition.get("name", ""),
            "round": round_info.get("name", ""),
            "surface": surface,
            "opponent_id": opponent_comp.get("id", ""),
            "opponent_name": opponent_comp.get("name", ""),
            "won": won,
            "score": score_str,
            "stats": player_stats,
        })

    return matches


def get_daily_results(date_str: str) -> list[dict]:
    """Get all match results for a given date.

    Args:
        date_str: YYYY-MM-DD format

    Returns list of match summaries with competitors, scores, stats.
    """
    data = _fetch(f"/schedules/{date_str}/summaries.json")
    results = []

    for s in data.get("summaries", []):
        ev = s.get("sport_event", {})
        status = s.get("sport_event_status", {})

        if status.get("status") not in ("closed", "ended"):
            continue

        context = ev.get("sport_event_context", {})
        competition = context.get("competition", {})
        comp_name = competition.get("name", "").lower()

        # Filter to ATP/WTA level
        is_atp_wta = any(t in comp_name for t in [
            "atp", "wta", "grand slam", "australian open",
            "roland garros", "wimbledon", "us open",
        ])

        comps = ev.get("competitors", [])
        if len(comps) != 2:
            continue

        winner_id = status.get("winner_id")
        period_scores = status.get("period_scores", [])
        score_str = " ".join(
            f"{ps.get('home_score', 0)}-{ps.get('away_score', 0)}"
            for ps in period_scores
        )

        # Per-competitor stats
        totals = s.get("statistics", {}).get("totals", {})
        comp_stats = {}
        for cs in totals.get("competitors", []):
            comp_stats[cs["id"]] = cs.get("statistics", {})

        results.append({
            "date": ev.get("start_time", "")[:10],
            "competition": competition.get("name", ""),
            "round": context.get("round", {}).get("name", ""),
            "surface": competition.get("surface", ""),
            "is_atp_wta": is_atp_wta,
            "home": {
                "id": comps[0].get("id", ""),
                "name": comps[0].get("name", ""),
            },
            "away": {
                "id": comps[1].get("id", ""),
                "name": comps[1].get("name", ""),
            },
            "winner_id": winner_id,
            "score": score_str,
            "home_stats": comp_stats.get(comps[0].get("id"), {}),
            "away_stats": comp_stats.get(comps[1].get("id"), {}),
        })

    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("Testing SportRadar API...\n")

    # Rankings
    rankings = get_rankings()
    print(f"ATP top 5:")
    for p in rankings["atp"][:5]:
        print(f"  #{p['rank']} {p['name']} ({p['points']} pts)")
    print(f"\nWTA top 5:")
    for p in rankings["wta"][:5]:
        print(f"  #{p['rank']} {p['name']} ({p['points']} pts)")

    # Recent results for #1
    top = rankings["atp"][0]
    print(f"\n{top['name']} recent matches:")
    results = get_competitor_results(top["sr_id"])
    for m in results[:5]:
        w = "W" if m["won"] else "L"
        print(f"  {m['date']} [{w}] vs {m['opponent_name']} {m['score']} ({m['competition']})")
        if m["stats"]:
            st = m["stats"]
            svpt = st.get("service_points_won", 0) + st.get("service_points_lost", 0)
            print(f"    Aces: {st['aces']}, 1stSv: {st['first_serve_successful']}/{svpt}")
