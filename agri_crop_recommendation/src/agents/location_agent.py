"""
Location Agent — World Location Hierarchy
==========================================
Manages Country → State → District data from bundled JSON.
No external API calls — pure in-memory lookup.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_LOCATIONS_FILE = Path(__file__).parent.parent.parent / "data" / "reference" / "world_locations.json"

_data: Optional[dict] = None
_last_mtime: float = 0.0


def _load():
    global _data, _last_mtime
    
    if not _LOCATIONS_FILE.exists():
        logger.error(f"world_locations.json not found at {_LOCATIONS_FILE}")
        if _data is None:
            _data = {"countries": [], "states": {}, "districts": {}}
        return

    current_mtime = os.path.getmtime(_LOCATIONS_FILE)
    
    if _data is not None and current_mtime == _last_mtime:
        return

    try:
        with open(_LOCATIONS_FILE, encoding="utf-8") as f:
            _data = json.load(f)
        _last_mtime = current_mtime
        logger.info(f"[LocationAgent] Successfully loaded/reloaded data from { _LOCATIONS_FILE}")
    except Exception as e:
        logger.error(f"[LocationAgent] Error loading JSON file: {e}")
        if _data is None:
            _data = {"countries": [], "states": {}, "districts": {}}


def get_countries() -> List[dict]:
    """Return list of {code, name} for all countries."""
    _load()
    return sorted(_data.get("countries", []), key=lambda c: c["name"])


def get_states(country_code: str) -> List[dict]:
    """Return list of {code, name} states for a country code."""
    _load()
    return sorted(
        _data.get("states", {}).get(country_code.upper(), []),
        key=lambda s: s["name"]
    )


def get_districts(country_code: str, state_code: str) -> List[dict]:
    """Return list of {name, lat, lon} districts for a country+state.
    
    Falls back to the state capital when no granular districts are defined.
    """
    _load()
    key = f"{country_code.upper()}_{state_code.upper()}"
    districts = _data.get("districts", {}).get(key, [])
    if districts:
        return sorted(districts, key=lambda d: d["name"])
    
    # Fallback: use state-level entry as a single selectable region
    states = _data.get("states", {}).get(country_code.upper(), [])
    for s in states:
        if s["code"].upper() == state_code.upper():
            state_lat = s.get("lat", 20.0)
            state_lon = s.get("lon", 78.0)
            # Return the state capital as the only district option
            return [{"name": s["name"], "lat": state_lat, "lon": state_lon}]
    
    return []



def resolve_coordinates(country_code: str, state_code: str, district_name: str) -> Tuple[float, float]:
    """Return (lat, lon) for a given district."""
    districts = get_districts(country_code, state_code)
    for d in districts:
        if d["name"].lower() == district_name.lower():
            return float(d["lat"]), float(d["lon"])
    # Fallback: return state capital coords if district not found
    states = get_states(country_code)
    for s in states:
        if s["code"].upper() == state_code.upper():
            return float(s.get("lat", 20.0)), float(s.get("lon", 78.0))
    return 20.0, 78.0  # India center as last resort
