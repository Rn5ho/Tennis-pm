"""Compute Elo ratings from Sackmann match history.

Computes overall Elo and surface-specific Elo for each player.
Ratings are attached to each match row as PRE-MATCH values,
then updated after the match.
"""

import re
from collections import defaultdict

import pandas as pd

from config.settings import (
    ELO_START_RATING,
    ELO_K_INITIAL,
    ELO_K_MEDIUM,
    ELO_K_STABLE,
    ELO_SURFACE_K_MULTIPLIER,
)

# Map surfaces to canonical names (Carpet -> Hard)
SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Hard"}

# Round ordering for sorting within a tournament
ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7,
    "RR": 3,  # round robin (tour finals) ~ R32 level
    "ER": 0,  # early rounds
    "BR": 0,  # bronze medal match
}


def _is_walkover(score: str) -> bool:
    """Check if a match was a walkover or default (not a real competitive result)."""
    if not isinstance(score, str):
        return True
    s = score.upper()
    return bool(re.search(r"\bW/O\b|\bDEF\b|\bWalkover\b", s, re.IGNORECASE))


def _get_k(match_count: int) -> float:
    if match_count < 20:
        return ELO_K_INITIAL
    elif match_count < 50:
        return ELO_K_MEDIUM
    return ELO_K_STABLE


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _update(winner_elo: float, loser_elo: float, k: float) -> tuple[float, float]:
    e = expected_score(winner_elo, loser_elo)
    new_w = winner_elo + k * (1.0 - e)
    new_l = loser_elo + k * (0.0 - (1.0 - e))
    return new_w, new_l


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for all matches.

    Args:
        matches: DataFrame with columns: winner_id, loser_id, surface, score,
                 tourney_date, round. Must be sorted chronologically.

    Returns:
        Copy of matches with added columns:
        winner_elo, loser_elo, winner_surface_elo, loser_surface_elo
    """
    matches = matches.copy()

    # Sort: by tournament date, then by round within tournament
    matches["_round_ord"] = matches["round"].map(ROUND_ORDER).fillna(3).astype(int)
    matches = matches.sort_values(["tourney_date", "tourney_id", "_round_ord", "match_num"])
    matches = matches.reset_index(drop=True)

    # State: overall Elo, surface Elo, match counts
    overall = defaultdict(lambda: ELO_START_RATING)
    surface_elo = defaultdict(lambda: defaultdict(lambda: ELO_START_RATING))
    match_counts = defaultdict(int)
    surface_counts = defaultdict(lambda: defaultdict(int))

    # Output columns
    w_elo = []
    l_elo = []
    w_surf = []
    l_surf = []

    for row in matches.itertuples():
        wid = row.winner_id
        lid = row.loser_id
        surf = SURFACE_MAP.get(row.surface, "Hard")

        # Record pre-match Elo
        w_elo.append(overall[wid])
        l_elo.append(overall[lid])
        w_surf.append(surface_elo[wid][surf])
        l_surf.append(surface_elo[lid][surf])

        # Skip walkovers for Elo updates (but still record pre-match values)
        if _is_walkover(row.score):
            continue

        # Update overall Elo
        k_w = _get_k(match_counts[wid])
        k_l = _get_k(match_counts[lid])
        k = (k_w + k_l) / 2  # average K for the match
        new_w, new_l = _update(overall[wid], overall[lid], k)
        overall[wid] = new_w
        overall[lid] = new_l
        match_counts[wid] += 1
        match_counts[lid] += 1

        # Update surface Elo
        sk_w = _get_k(surface_counts[wid][surf]) * ELO_SURFACE_K_MULTIPLIER
        sk_l = _get_k(surface_counts[lid][surf]) * ELO_SURFACE_K_MULTIPLIER
        sk = (sk_w + sk_l) / 2
        new_sw, new_sl = _update(surface_elo[wid][surf], surface_elo[lid][surf], sk)
        surface_elo[wid][surf] = new_sw
        surface_elo[lid][surf] = new_sl
        surface_counts[wid][surf] += 1
        surface_counts[lid][surf] += 1

    matches["winner_elo"] = w_elo
    matches["loser_elo"] = l_elo
    matches["winner_surface_elo"] = w_surf
    matches["loser_surface_elo"] = l_surf

    matches.drop(columns=["_round_ord"], inplace=True)
    return matches
