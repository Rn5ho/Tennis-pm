"""Player name matching between Polymarket and Sackmann.

Two modes:
1. GENERATE: Fuzzy-match PM names to Sackmann, output to verified JSON for human review
2. RUNTIME: Load only the verified mapping. No fuzzy logic. No guessing.

The verified mapping (data/player_map.json) is the single source of truth.
If a PM player isn't in the map, we skip them — better to miss a bet than bet wrong.
"""

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from config.settings import PROJECT_ROOT, SACKMANN_ATP_DIR, SACKMANN_WTA_DIR

PLAYER_MAP_PATH = PROJECT_ROOT / "data" / "player_map.json"


# ── Runtime API (used by the bot/predictor) ────────────────────────────

def load_player_map() -> dict:
    """Load the verified player mapping.

    Returns:
        {pm_name: {"sackmann_id": str, "sackmann_name": str}}
    """
    if not PLAYER_MAP_PATH.exists():
        raise FileNotFoundError(
            f"No player map at {PLAYER_MAP_PATH}. "
            "Run `python -m model.name_match generate` first."
        )
    with open(PLAYER_MAP_PATH) as f:
        return json.load(f)


def lookup(pm_name: str, player_map: dict = None) -> str | None:
    """Look up a Sackmann player ID for a Polymarket name.

    Returns sackmann_id or None if not in the verified map.
    """
    if player_map is None:
        player_map = load_player_map()
    entry = player_map.get(pm_name)
    if entry and entry.get("sackmann_id"):
        return entry["sackmann_id"]
    return None


# ── Generation tools (one-time, for building the map) ──────────────────

def _normalize(name: str) -> str:
    """Normalize: strip accents, hyphens, apostrophes, lowercase."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[-'.()]", " ", ascii_name)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def get_sackmann_players(years: range = None) -> dict[str, str]:
    """Load {player_id: player_name} from Sackmann CSVs."""
    if years is None:
        years = range(2020, 2025)

    players = {}
    for tour, data_dir in [("atp", SACKMANN_ATP_DIR), ("wta", SACKMANN_WTA_DIR)]:
        data_dir = Path(data_dir)
        for year in years:
            for pattern in [f"{tour}_matches_{year}.csv", f"{tour}_matches_qual_chall_{year}.csv"]:
                f = data_dir / pattern
                if not f.exists():
                    continue
                df = pd.read_csv(f, usecols=["winner_id", "winner_name", "loser_id", "loser_name"])
                for _, row in df.iterrows():
                    players[str(int(row["winner_id"]))] = row["winner_name"]
                    players[str(int(row["loser_id"]))] = row["loser_name"]
    return players


def generate_map(pm_names: list[str]) -> dict:
    """Generate a player mapping with confidence scores for review.

    High-confidence exact matches are auto-verified.
    Fuzzy matches are flagged for human review.
    """
    sackmann = get_sackmann_players()

    # Build normalized index: norm_name -> [(id, original_name)]
    norm_index = {}
    for pid, name in sackmann.items():
        norm = _normalize(name)
        norm_index.setdefault(norm, []).append((pid, name))

    # Also build last-name index for single-name PM entries
    lastname_index = {}
    for pid, name in sackmann.items():
        parts = name.split()
        if parts:
            ln = _normalize(parts[-1])
            lastname_index.setdefault(ln, []).append((pid, name))

    result = {}

    for pm_name in sorted(pm_names):
        pm_norm = _normalize(pm_name)

        # 1. Exact normalized match
        if pm_norm in norm_index:
            entries = norm_index[pm_norm]
            if len(entries) == 1:
                pid, sname = entries[0]
                result[pm_name] = {
                    "sackmann_id": pid,
                    "sackmann_name": sname,
                    "confidence": 1.0,
                    "method": "exact",
                    "verified": True,
                }
                continue
            else:
                # Multiple players with same normalized name — flag
                result[pm_name] = {
                    "sackmann_id": entries[0][0],
                    "sackmann_name": entries[0][1],
                    "confidence": 0.5,
                    "method": f"exact_ambiguous ({len(entries)} candidates)",
                    "candidates": [{"id": e[0], "name": e[1]} for e in entries],
                    "verified": False,
                }
                continue

        # 2. Single-word PM name -> last name lookup
        pm_parts = pm_norm.split()
        if len(pm_parts) == 1 and pm_norm in lastname_index:
            entries = lastname_index[pm_norm]
            if len(entries) == 1:
                pid, sname = entries[0]
                result[pm_name] = {
                    "sackmann_id": pid,
                    "sackmann_name": sname,
                    "confidence": 0.9,
                    "method": "lastname_unique",
                    "verified": False,  # still needs review — single-name matches are risky
                }
                continue
            else:
                result[pm_name] = {
                    "sackmann_id": None,
                    "sackmann_name": None,
                    "confidence": 0.0,
                    "method": f"lastname_ambiguous ({len(entries)} candidates)",
                    "candidates": [{"id": e[0], "name": e[1]} for e in entries],
                    "verified": False,
                }
                continue

        # 3. Fuzzy match
        best_score = 0.0
        best_entry = None
        for norm_name, entries in norm_index.items():
            score = _similarity(pm_norm, norm_name)
            # Boost if last names match
            sack_parts = norm_name.split()
            if pm_parts and sack_parts and pm_parts[-1] == sack_parts[-1]:
                score = max(score, 0.6 + score * 0.4)
            if score > best_score:
                best_score = score
                best_entry = entries[0]

        if best_score >= 0.80 and best_entry:
            result[pm_name] = {
                "sackmann_id": best_entry[0],
                "sackmann_name": best_entry[1],
                "confidence": round(best_score, 3),
                "method": "fuzzy",
                "verified": best_score >= 0.95,  # auto-verify very high matches
            }
        else:
            result[pm_name] = {
                "sackmann_id": None,
                "sackmann_name": None,
                "confidence": round(best_score, 3) if best_entry else 0.0,
                "method": "no_match",
                "verified": False,
            }

    return result


def save_map(mapping: dict) -> None:
    """Save the player mapping. Only includes verified entries at runtime."""
    PLAYER_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PLAYER_MAP_PATH, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved player map to {PLAYER_MAP_PATH}")


def print_review_report(mapping: dict) -> None:
    """Print a summary for human review."""
    verified = {k: v for k, v in mapping.items() if v.get("verified")}
    needs_review = {k: v for k, v in mapping.items() if not v.get("verified") and v.get("sackmann_id")}
    unmatched = {k: v for k, v in mapping.items() if not v.get("sackmann_id")}

    print(f"\n{'='*60}")
    print(f"PLAYER MAP SUMMARY")
    print(f"{'='*60}")
    print(f"  Auto-verified (exact match): {len(verified)}")
    print(f"  Needs review (fuzzy/ambiguous): {len(needs_review)}")
    print(f"  Unmatched: {len(unmatched)}")
    print(f"  Total: {len(mapping)}")

    if needs_review:
        print(f"\n--- NEEDS REVIEW ({len(needs_review)}) ---")
        print(f"{'PM Name':<35} {'Sackmann Name':<35} {'Conf':>5} {'Method'}")
        print("-" * 90)
        for pm, info in sorted(needs_review.items(), key=lambda x: x[1]["confidence"]):
            print(f"{pm:<35} {info['sackmann_name']:<35} {info['confidence']:>5.2f} {info['method']}")

    if unmatched:
        print(f"\n--- UNMATCHED ({len(unmatched)}) ---")
        for pm in sorted(unmatched.keys()):
            info = unmatched[pm]
            cands = info.get("candidates", [])
            if cands:
                cand_str = ", ".join(f"{c['name']} ({c['id']})" for c in cands[:3])
                print(f"  {pm} -> candidates: {cand_str}")
            else:
                print(f"  {pm}")


if __name__ == "__main__":
    import sqlite3
    import sys
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT DISTINCT outcome_a FROM markets UNION SELECT DISTINCT outcome_b FROM markets"
    ).fetchall()
    pm_names = sorted(set(r[0] for r in rows if r[0]))
    conn.close()

    print(f"Found {len(pm_names)} unique Polymarket player names")

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        mapping = generate_map(pm_names)
        save_map(mapping)
        print_review_report(mapping)
        print(f"\nEdit {PLAYER_MAP_PATH} to:")
        print(f"  1. Set 'verified': true for correct fuzzy matches")
        print(f"  2. Fix wrong matches by changing sackmann_id/sackmann_name")
        print(f"  3. Fill in unmatched players if you know their Sackmann ID")
    else:
        # Quick check: how many current PM players are in the verified map?
        try:
            pmap = load_player_map()
            verified = sum(1 for n in pm_names if n in pmap and pmap[n].get("verified"))
            total = len(pm_names)
            print(f"Verified matches: {verified}/{total} ({verified/total*100:.1f}%)")
            missing = [n for n in pm_names if n not in pmap or not pmap[n].get("verified")]
            if missing:
                print(f"Unverified ({len(missing)}): {missing[:10]}...")
        except FileNotFoundError:
            print("No player map found. Run: python -m model.name_match generate")
