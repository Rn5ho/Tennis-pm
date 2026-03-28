"""Tournament metadata lookup: surface, level, best_of.

Maps Polymarket league names and SportRadar competition names to
(surface, tourney_level, best_of) using Sackmann historical data
plus manual overrides for names not in Sackmann.

Used by:
- predictor.py: to pass correct surface/level/best_of to predict_match()
- backfill_elo.py: to determine surface from SR competition names
"""

import re
from pathlib import Path

import pandas as pd

from config.settings import SACKMANN_ATP_DIR, SACKMANN_WTA_DIR

# ---------------------------------------------------------------------------
# Manual overrides for tournaments NOT in Sackmann or with different naming
# Format: normalized_city -> (surface, level)
# ---------------------------------------------------------------------------
_MANUAL_OVERRIDES = {
    # Grand Slams (SR uses "French Open", Sackmann uses "Roland Garros")
    "french open": ("Clay", "G"),
    "roland garros": ("Clay", "G"),
    "australian open": ("Hard", "G"),
    "wimbledon": ("Grass", "G"),
    "us open": ("Hard", "G"),
    # Masters 1000
    "indian wells": ("Hard", "M"),
    "indian wells masters": ("Hard", "M"),
    "bnp paribas open": ("Hard", "M"),
    "miami": ("Hard", "M"),
    "miami open": ("Hard", "M"),
    "monte carlo": ("Clay", "M"),
    "monte-carlo": ("Clay", "M"),
    "madrid": ("Clay", "M"),
    "rome": ("Clay", "M"),
    "toronto": ("Hard", "M"),
    "montreal": ("Hard", "M"),
    "cincinnati": ("Hard", "M"),
    "shanghai": ("Hard", "M"),
    "paris": ("Hard", "M"),
    # ATP 500
    "mexican open": ("Hard", "A"),
    "acapulco": ("Hard", "A"),
    "dubai tennis championships": ("Hard", "A"),
    "dubai": ("Hard", "A"),
    "chile open": ("Clay", "A"),
    "houston": ("Clay", "A"),
    "us men's clay court championships": ("Clay", "A"),
    "winston salem": ("Hard", "A"),
    "s-hertogenbosch": ("Grass", "A"),
    "london": ("Grass", "A"),  # Queen's Club
    # WTA specific
    "atx open": ("Hard", "A"),
    "austin": ("Hard", "A"),
    "copa colsanitas": ("Clay", "A"),
    "credit one charleston open": ("Clay", "P"),
    "merida open akron": ("Hard", "A"),
    # WTA 125K
    "mumbai": ("Hard", "C"),
    "dubrovnik": ("Clay", "C"),
    # Challengers missing from Sackmann
    "hersonissos": ("Hard", "C"),
    "cap cana": ("Hard", "C"),
    "bucaramanga": ("Clay", "C"),
    "morelia": ("Hard", "C"),
    "morelos": ("Hard", "C"),
    "thionville": ("Hard", "C"),
    "soma bay": ("Hard", "C"),
    "phan thiet": ("Hard", "C"),
    "athens": ("Clay", "C"),
    "barranquilla": ("Clay", "C"),
    "cancun": ("Hard", "C"),
    "cesenatico": ("Clay", "C"),
    "francavilla": ("Clay", "C"),
    "monza": ("Clay", "C"),
    "barletta": ("Clay", "C"),
    "rende": ("Clay", "C"),
    "grado": ("Clay", "C"),
    "menorca": ("Clay", "C"),
    "makarska": ("Clay", "C"),
    "playford city": ("Hard", "C"),
    "bunschoten": ("Clay", "C"),
    "chisinau": ("Hard", "C"),
    "suzhou": ("Hard", "C"),
    "jingshan": ("Hard", "C"),
    "changsha": ("Hard", "C"),
    "huzhou": ("Hard", "C"),
    "manila": ("Hard", "C"),
    "islamabad": ("Hard", "C"),
    "fujairah": ("Hard", "C"),
    "samsun": ("Hard", "C"),
    "tbilisi": ("Hard", "C"),
    "montreux": ("Clay", "C"),
    "contrexeville": ("Clay", "C"),
    "angers": ("Hard", "C"),
    "royan": ("Clay", "C"),
    "limoges": ("Hard", "C"),
    "les sables d olonne": ("Clay", "C"),
    "saint-malo": ("Hard", "C"),
    "caldas da rainha": ("Clay", "C"),
    "itajai": ("Clay", "C"),
    "costa do sauipe": ("Clay", "C"),
    "colina": ("Clay", "C"),
    "tucuman": ("Clay", "C"),
    "queretaro": ("Hard", "C"),
    "metepec": ("Hard", "C"),
    "tampico": ("Hard", "C"),
    "baton rouge": ("Hard", "C"),
    "midland": ("Hard", "C"),
    "lincoln (ne)": ("Hard", "C"),
    "sumter": ("Clay", "C"),
    "hagen": ("Hard", "C"),
    "tolentino": ("Clay", "C"),
    "san sebastian": ("Hard", "C"),
    "vic": ("Clay", "C"),
    "targu mures": ("Clay", "C"),
    "bratislava": ("Hard", "C"),
    "rosario": ("Clay", "C"),
}


def _build_sackmann_map() -> dict[str, tuple[str, str]]:
    """Build city -> (surface, level) from Sackmann CSV data.

    Uses 2020-2024 data for current surfaces. Returns normalized
    city names as keys.
    """
    result = {}
    for tour, data_dir in [("atp", SACKMANN_ATP_DIR), ("wta", SACKMANN_WTA_DIR)]:
        data_dir = Path(data_dir)
        for year in range(2020, 2025):
            for pattern in [f"{tour}_matches_{year}.csv", f"{tour}_matches_qual_chall_{year}.csv"]:
                f = data_dir / pattern
                if not f.exists():
                    continue
                try:
                    df = pd.read_csv(f, usecols=["tourney_name", "surface", "tourney_level"], low_memory=False)
                except Exception:
                    continue
                for _, row in df.drop_duplicates("tourney_name").iterrows():
                    name = str(row["tourney_name"]).strip()
                    surf = str(row["surface"]).strip()
                    level = str(row.get("tourney_level", "")).strip()
                    if surf == "nan" or not surf:
                        continue
                    # Normalize: strip " CH" suffix, trailing numbers
                    clean = _normalize_name(name)
                    result[clean] = (surf, level)
    return result


def _normalize_name(name: str) -> str:
    """Normalize a tournament/city name for lookup."""
    clean = re.sub(r"\s+CH$", "", name)         # Remove " CH" suffix
    clean = re.sub(r"\s+\d+$", "", clean)        # Remove trailing numbers
    clean = re.sub(r",\s*Qualification$", "", clean)  # Remove ", Qualification"
    return clean.strip().lower()


# Build the combined lookup: manual overrides take precedence
_SACKMANN_MAP = _build_sackmann_map()
_COMBINED_MAP: dict[str, tuple[str, str]] = {}
_COMBINED_MAP.update(_SACKMANN_MAP)
_COMBINED_MAP.update(_MANUAL_OVERRIDES)  # overrides win


def _extract_city_from_sr(competition: str) -> str | None:
    """Extract city/tournament name from SR competition string.

    Examples:
        "ATP Miami, USA Men Singles" -> "miami"
        "ATP Challenger Santiago, Chile Men Singles" -> "santiago"
        "WTA 125K Antalya, Turkiye Women Singles" -> "antalya"
        "Australian Open Men Singles" -> "australian open"
        "French Open Women Singles" -> "french open"
    """
    # Grand Slams
    for gs in ["Australian Open", "French Open", "Wimbledon", "US Open"]:
        if gs in competition:
            return gs.lower()

    # ATP/WTA [Challenger] [125K] City, Country (Men|Women) Singles
    m = re.match(
        r"(?:ATP|WTA)\s+(?:Challenger\s+|125K?\s+)?(.+?),\s+.+?\s+(?:Men|Women)\s+Singles",
        competition,
    )
    if m:
        return m.group(1).strip().lower()

    return None


def _extract_tournament_from_pm(title: str) -> str:
    """Extract tournament name from PM event title.

    Examples:
        "Miami Open: Player A vs Player B" -> "miami open"
        "Miami Open, Qualification: Player A vs Player B" -> "miami open"
    """
    # Split on ": " to separate tournament from players
    if ": " in title:
        tourney_part = title.split(": ", 1)[0]
    else:
        tourney_part = title
    # Strip ", Qualification" etc.
    tourney_part = re.sub(r",\s*Qualification$", "", tourney_part)
    return tourney_part.strip().lower()


def _lookup_name(name: str) -> tuple[str, str] | None:
    """Look up a normalized name in the combined map.

    Tries exact match, then strips trailing numbers for numbered
    variants (e.g., "hersonissos 2" -> "hersonissos").
    """
    if name in _COMBINED_MAP:
        return _COMBINED_MAP[name]

    # Strip trailing numbers (for "Antalya 2", "Kigali 2", etc.)
    stripped = re.sub(r"\s+\d+$", "", name).strip()
    if stripped != name and stripped in _COMBINED_MAP:
        return _COMBINED_MAP[stripped]

    # Try partial match: if our name contains a known key or vice versa
    for key, val in _COMBINED_MAP.items():
        if len(key) >= 4 and (key in name or name in key):
            return val

    return None


def lookup_tournament(league: str | None, title: str, series: str = "atp") -> tuple[str, str, int]:
    """Resolve tournament metadata from PM event data.

    Args:
        league: PM league field (may be None for older events)
        title: PM event title (e.g., "Miami Open: Player A vs Player B")
        series: "atp" or "wta"

    Returns:
        (surface, tourney_level, best_of)
    """
    # Try league first (more reliable when present)
    if league:
        name = _normalize_name(league)
        result = _lookup_name(name)
        if result:
            surface, level = result
            best_of = _get_best_of(level, series)
            return surface, level, best_of

    # Fall back to title
    name = _extract_tournament_from_pm(title)
    result = _lookup_name(name)
    if result:
        surface, level = result
        best_of = _get_best_of(level, series)
        return surface, level, best_of

    # Default: Hard court, Challenger/250 level
    level = "C" if _looks_like_challenger(title, league) else "A"
    return "Hard", level, 3


def lookup_surface_from_sr(competition: str) -> str:
    """Resolve surface from an SR competition name.

    Used by backfill_elo.py to determine the surface for Elo updates.

    Args:
        competition: SR competition name (e.g., "ATP Challenger Santiago, Chile Men Singles")

    Returns:
        Surface string: "Hard", "Clay", or "Grass"
    """
    city = _extract_city_from_sr(competition)
    if city:
        result = _lookup_name(city)
        if result:
            return result[0]

    # Fallback: default Hard
    return "Hard"


def _get_best_of(level: str, series: str) -> int:
    """Determine best_of from tournament level and series."""
    if level == "G" and series == "atp":
        return 5
    return 3


def _looks_like_challenger(title: str, league: str | None) -> bool:
    """Heuristic: is this likely a Challenger event?"""
    text = (title + " " + (league or "")).lower()
    # Known non-Challenger keywords
    if any(kw in text for kw in ["open", "masters", "championships", "grand slam"]):
        return False
    # PM Challengers are typically named just by city
    return True
