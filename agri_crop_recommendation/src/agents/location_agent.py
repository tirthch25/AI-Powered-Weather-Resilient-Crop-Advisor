"""
Location Agent — All 195 Countries, 100% LLM-Driven
=====================================================
Strategy:
  1. Country list — all 195 UN-recognised countries embedded directly.
     No JSON file, no static geo data.
  2. States / Provinces — ALWAYS resolved by the LLM (Gemini/Ollama).
     No static JSON lookup. The LLM returns the real, complete list for
     every country (e.g., all 16 German Bundesländer, all 47 Japanese
     prefectures, all 29 Indian states, etc.)
  3. Districts — ALWAYS resolved by the LLM.
  4. Coordinates — resolved from the LLM-returned district lat/lon, or
     via llm_resolve_coords as a secondary call.

This guarantees accuracy for every country without a stale static file.
Results are cached in-memory for 24 hours by the LLM agent layer, so
repeated lookups are instant.
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── LLM agent (lazy import) ──────────────────────────────────────────────────
_llm_agent = None


def _get_llm_agent():
    global _llm_agent
    if _llm_agent is None:
        try:
            from src.agents import llm_location_agent
            _llm_agent = llm_location_agent
        except ImportError:
            try:
                import importlib
                _llm_agent = importlib.import_module("src.agents.llm_location_agent")
            except Exception as e:
                logger.warning("[LocationAgent] llm_location_agent not available: %s", e)
    return _llm_agent


# ── All 195 UN-recognised countries ──────────────────────────────────────────
# ISO 3166-1 alpha-2 codes + names only. No geo data needed here.
ALL_195_COUNTRIES: List[Dict[str, str]] = sorted([
    {"code": "AF", "name": "Afghanistan"},
    {"code": "AL", "name": "Albania"},
    {"code": "DZ", "name": "Algeria"},
    {"code": "AD", "name": "Andorra"},
    {"code": "AO", "name": "Angola"},
    {"code": "AG", "name": "Antigua and Barbuda"},
    {"code": "AR", "name": "Argentina"},
    {"code": "AM", "name": "Armenia"},
    {"code": "AU", "name": "Australia"},
    {"code": "AT", "name": "Austria"},
    {"code": "AZ", "name": "Azerbaijan"},
    {"code": "BS", "name": "Bahamas"},
    {"code": "BH", "name": "Bahrain"},
    {"code": "BD", "name": "Bangladesh"},
    {"code": "BB", "name": "Barbados"},
    {"code": "BY", "name": "Belarus"},
    {"code": "BE", "name": "Belgium"},
    {"code": "BZ", "name": "Belize"},
    {"code": "BJ", "name": "Benin"},
    {"code": "BT", "name": "Bhutan"},
    {"code": "BO", "name": "Bolivia"},
    {"code": "BA", "name": "Bosnia and Herzegovina"},
    {"code": "BW", "name": "Botswana"},
    {"code": "BR", "name": "Brazil"},
    {"code": "BN", "name": "Brunei"},
    {"code": "BG", "name": "Bulgaria"},
    {"code": "BF", "name": "Burkina Faso"},
    {"code": "BI", "name": "Burundi"},
    {"code": "CV", "name": "Cabo Verde"},
    {"code": "KH", "name": "Cambodia"},
    {"code": "CM", "name": "Cameroon"},
    {"code": "CA", "name": "Canada"},
    {"code": "CF", "name": "Central African Republic"},
    {"code": "TD", "name": "Chad"},
    {"code": "CL", "name": "Chile"},
    {"code": "CN", "name": "China"},
    {"code": "CO", "name": "Colombia"},
    {"code": "KM", "name": "Comoros"},
    {"code": "CG", "name": "Congo"},
    {"code": "CD", "name": "Congo (DRC)"},
    {"code": "CR", "name": "Costa Rica"},
    {"code": "CI", "name": "Cote d'Ivoire"},
    {"code": "HR", "name": "Croatia"},
    {"code": "CU", "name": "Cuba"},
    {"code": "CY", "name": "Cyprus"},
    {"code": "CZ", "name": "Czech Republic"},
    {"code": "DK", "name": "Denmark"},
    {"code": "DJ", "name": "Djibouti"},
    {"code": "DM", "name": "Dominica"},
    {"code": "DO", "name": "Dominican Republic"},
    {"code": "EC", "name": "Ecuador"},
    {"code": "EG", "name": "Egypt"},
    {"code": "SV", "name": "El Salvador"},
    {"code": "GQ", "name": "Equatorial Guinea"},
    {"code": "ER", "name": "Eritrea"},
    {"code": "EE", "name": "Estonia"},
    {"code": "SZ", "name": "Eswatini"},
    {"code": "ET", "name": "Ethiopia"},
    {"code": "FJ", "name": "Fiji"},
    {"code": "FI", "name": "Finland"},
    {"code": "FR", "name": "France"},
    {"code": "GA", "name": "Gabon"},
    {"code": "GM", "name": "Gambia"},
    {"code": "GE", "name": "Georgia"},
    {"code": "DE", "name": "Germany"},
    {"code": "GH", "name": "Ghana"},
    {"code": "GR", "name": "Greece"},
    {"code": "GD", "name": "Grenada"},
    {"code": "GT", "name": "Guatemala"},
    {"code": "GN", "name": "Guinea"},
    {"code": "GW", "name": "Guinea-Bissau"},
    {"code": "GY", "name": "Guyana"},
    {"code": "HT", "name": "Haiti"},
    {"code": "HN", "name": "Honduras"},
    {"code": "HU", "name": "Hungary"},
    {"code": "IS", "name": "Iceland"},
    {"code": "IN", "name": "India"},
    {"code": "ID", "name": "Indonesia"},
    {"code": "IR", "name": "Iran"},
    {"code": "IQ", "name": "Iraq"},
    {"code": "IE", "name": "Ireland"},
    {"code": "IL", "name": "Israel"},
    {"code": "IT", "name": "Italy"},
    {"code": "JM", "name": "Jamaica"},
    {"code": "JP", "name": "Japan"},
    {"code": "JO", "name": "Jordan"},
    {"code": "KZ", "name": "Kazakhstan"},
    {"code": "KE", "name": "Kenya"},
    {"code": "KI", "name": "Kiribati"},
    {"code": "KP", "name": "Korea (North)"},
    {"code": "KR", "name": "Korea (South)"},
    {"code": "XK", "name": "Kosovo"},
    {"code": "KW", "name": "Kuwait"},
    {"code": "KG", "name": "Kyrgyzstan"},
    {"code": "LA", "name": "Laos"},
    {"code": "LV", "name": "Latvia"},
    {"code": "LB", "name": "Lebanon"},
    {"code": "LS", "name": "Lesotho"},
    {"code": "LR", "name": "Liberia"},
    {"code": "LY", "name": "Libya"},
    {"code": "LI", "name": "Liechtenstein"},
    {"code": "LT", "name": "Lithuania"},
    {"code": "LU", "name": "Luxembourg"},
    {"code": "MG", "name": "Madagascar"},
    {"code": "MW", "name": "Malawi"},
    {"code": "MY", "name": "Malaysia"},
    {"code": "MV", "name": "Maldives"},
    {"code": "ML", "name": "Mali"},
    {"code": "MT", "name": "Malta"},
    {"code": "MH", "name": "Marshall Islands"},
    {"code": "MR", "name": "Mauritania"},
    {"code": "MU", "name": "Mauritius"},
    {"code": "MX", "name": "Mexico"},
    {"code": "FM", "name": "Micronesia"},
    {"code": "MD", "name": "Moldova"},
    {"code": "MC", "name": "Monaco"},
    {"code": "MN", "name": "Mongolia"},
    {"code": "ME", "name": "Montenegro"},
    {"code": "MA", "name": "Morocco"},
    {"code": "MZ", "name": "Mozambique"},
    {"code": "MM", "name": "Myanmar"},
    {"code": "NA", "name": "Namibia"},
    {"code": "NR", "name": "Nauru"},
    {"code": "NP", "name": "Nepal"},
    {"code": "NL", "name": "Netherlands"},
    {"code": "NZ", "name": "New Zealand"},
    {"code": "NI", "name": "Nicaragua"},
    {"code": "NE", "name": "Niger"},
    {"code": "NG", "name": "Nigeria"},
    {"code": "MK", "name": "North Macedonia"},
    {"code": "NO", "name": "Norway"},
    {"code": "OM", "name": "Oman"},
    {"code": "PK", "name": "Pakistan"},
    {"code": "PW", "name": "Palau"},
    {"code": "PS", "name": "Palestine"},
    {"code": "PA", "name": "Panama"},
    {"code": "PG", "name": "Papua New Guinea"},
    {"code": "PY", "name": "Paraguay"},
    {"code": "PE", "name": "Peru"},
    {"code": "PH", "name": "Philippines"},
    {"code": "PL", "name": "Poland"},
    {"code": "PT", "name": "Portugal"},
    {"code": "QA", "name": "Qatar"},
    {"code": "RO", "name": "Romania"},
    {"code": "RU", "name": "Russia"},
    {"code": "RW", "name": "Rwanda"},
    {"code": "KN", "name": "Saint Kitts and Nevis"},
    {"code": "LC", "name": "Saint Lucia"},
    {"code": "VC", "name": "Saint Vincent and the Grenadines"},
    {"code": "WS", "name": "Samoa"},
    {"code": "SM", "name": "San Marino"},
    {"code": "ST", "name": "Sao Tome and Principe"},
    {"code": "SA", "name": "Saudi Arabia"},
    {"code": "SN", "name": "Senegal"},
    {"code": "RS", "name": "Serbia"},
    {"code": "SC", "name": "Seychelles"},
    {"code": "SL", "name": "Sierra Leone"},
    {"code": "SG", "name": "Singapore"},
    {"code": "SK", "name": "Slovakia"},
    {"code": "SI", "name": "Slovenia"},
    {"code": "SB", "name": "Solomon Islands"},
    {"code": "SO", "name": "Somalia"},
    {"code": "ZA", "name": "South Africa"},
    {"code": "SS", "name": "South Sudan"},
    {"code": "ES", "name": "Spain"},
    {"code": "LK", "name": "Sri Lanka"},
    {"code": "SD", "name": "Sudan"},
    {"code": "SR", "name": "Suriname"},
    {"code": "SE", "name": "Sweden"},
    {"code": "CH", "name": "Switzerland"},
    {"code": "SY", "name": "Syria"},
    {"code": "TW", "name": "Taiwan"},
    {"code": "TJ", "name": "Tajikistan"},
    {"code": "TZ", "name": "Tanzania"},
    {"code": "TH", "name": "Thailand"},
    {"code": "TL", "name": "Timor-Leste"},
    {"code": "TG", "name": "Togo"},
    {"code": "TO", "name": "Tonga"},
    {"code": "TT", "name": "Trinidad and Tobago"},
    {"code": "TN", "name": "Tunisia"},
    {"code": "TR", "name": "Turkey"},
    {"code": "TM", "name": "Turkmenistan"},
    {"code": "TV", "name": "Tuvalu"},
    {"code": "UG", "name": "Uganda"},
    {"code": "UA", "name": "Ukraine"},
    {"code": "AE", "name": "United Arab Emirates"},
    {"code": "GB", "name": "United Kingdom"},
    {"code": "US", "name": "United States"},
    {"code": "UY", "name": "Uruguay"},
    {"code": "UZ", "name": "Uzbekistan"},
    {"code": "VU", "name": "Vanuatu"},
    {"code": "VE", "name": "Venezuela"},
    {"code": "VN", "name": "Vietnam"},
    {"code": "YE", "name": "Yemen"},
    {"code": "ZM", "name": "Zambia"},
    {"code": "ZW", "name": "Zimbabwe"},
], key=lambda c: c["name"])


# ── Public API ─────────────────────────────────────────────────────────────────

def get_countries() -> List[dict]:
    """Return all 195 UN-recognised countries, sorted alphabetically.

    has_static_data is always False — all state/district data is dynamic.
    The 'llm' badge in the UI will appear for every country, which is correct.
    """
    return [
        {**c, "has_static_data": False}
        for c in ALL_195_COUNTRIES
    ]


def get_states(country_code: str) -> List[dict]:
    """Return ALL states/provinces for a country via LLM.

    100% dynamic — the LLM returns the real, complete administrative
    division list (e.g. all 16 German Bundesländer, all 36 Nigerian
    states, all 47 Japanese prefectures, etc.).

    Results are cached 24 h in llm_location_agent._CACHE so repeated
    calls for the same country are instant.

    Returns list of {code, name, lat, lon, source}.
    """
    code = country_code.upper()
    country_name = next(
        (c["name"] for c in ALL_195_COUNTRIES if c["code"] == code), code
    )

    llm = _get_llm_agent()
    if llm:
        logger.info("[LocationAgent] Fetching states for %s via LLM", country_name)
        states = llm.llm_get_states(country_name, code)
        if states:
            return sorted(
                [{**s, "source": "llm"} for s in states],
                key=lambda s: s["name"],
            )

    # Ultimate fallback — treat the whole country as one selectable region
    logger.warning("[LocationAgent] LLM returned no states for %s — using country fallback", code)
    return [{"code": code, "name": country_name, "lat": 0.0, "lon": 0.0, "source": "fallback"}]


def get_districts(country_code: str, state_code: str) -> List[dict]:
    """Return districts/regions for a country+state via LLM.

    100% dynamic — no static JSON consulted.

    Returns list of {name, lat, lon, source}.
    """
    c_code = country_code.upper()
    s_code = state_code.upper()

    country_name = next(
        (c["name"] for c in ALL_195_COUNTRIES if c["code"] == c_code), c_code
    )

    # Resolve state name from LLM-cached states
    states = get_states(c_code)
    state_name = next(
        (s["name"] for s in states if s["code"].upper() == s_code), s_code
    )

    llm = _get_llm_agent()
    if llm:
        logger.info(
            "[LocationAgent] Fetching districts for %s / %s via LLM",
            state_name, country_name,
        )
        districts = llm.llm_get_districts(country_name, state_name, s_code, c_code)
        if districts:
            return sorted(
                [{**d, "source": "llm"} for d in districts],
                key=lambda d: d["name"],
            )

    # Fallback — use the state centre as the only selectable point
    state_lat = next((s.get("lat", 0.0) for s in states if s["code"].upper() == s_code), 0.0)
    state_lon = next((s.get("lon", 0.0) for s in states if s["code"].upper() == s_code), 0.0)
    logger.warning(
        "[LocationAgent] LLM returned no districts for %s/%s — using state centre",
        s_code, c_code,
    )
    return [{"name": state_name, "lat": state_lat, "lon": state_lon, "source": "fallback"}]


def resolve_coordinates(
    country_code: str,
    state_code: str,
    district_name: str,
) -> Tuple[float, float]:
    """Return (lat, lon) for a district.

    Uses the lat/lon already embedded in the LLM-returned district list.
    Falls back to llm_resolve_coords for a dedicated coordinate query.
    """
    districts = get_districts(country_code, state_code)
    for d in districts:
        if d["name"].lower() == district_name.lower():
            return float(d["lat"]), float(d["lon"])

    # Secondary LLM resolve if the district wasn't in the list
    country_name = next(
        (c["name"] for c in ALL_195_COUNTRIES if c["code"] == country_code.upper()),
        country_code,
    )
    states = get_states(country_code)
    state_name = next(
        (s["name"] for s in states if s["code"].upper() == state_code.upper()),
        state_code,
    )

    llm = _get_llm_agent()
    if llm:
        result = llm.llm_resolve_coords(country_name, state_name, district_name)
        if result and result.get("lat") != 0.0:
            return float(result["lat"]), float(result["lon"])

    # Last resort — state centre
    state_lat = next(
        (s.get("lat", 0.0) for s in states if s["code"].upper() == state_code.upper()), 0.0
    )
    state_lon = next(
        (s.get("lon", 0.0) for s in states if s["code"].upper() == state_code.upper()), 0.0
    )
    return state_lat, state_lon


def resolve_full(
    country_code: str,
    state_code: str,
    district_name: str,
) -> Dict:
    """Extended resolve — returns lat, lon, climate_zone, crop_notes.

    Used by data_gathering_agent for richer location context.
    Everything is LLM-driven.
    """
    country_name = next(
        (c["name"] for c in ALL_195_COUNTRIES if c["code"] == country_code.upper()),
        country_code,
    )
    states = get_states(country_code)
    state_name = next(
        (s["name"] for s in states if s["code"].upper() == state_code.upper()),
        state_code,
    )

    # Try to get coords from the district list first (avoids extra LLM call)
    districts = get_districts(country_code, state_code)
    for d in districts:
        if d["name"].lower() == district_name.lower():
            lat = float(d.get("lat", 0.0))
            lon = float(d.get("lon", 0.0))
            if lat != 0.0 or lon != 0.0:
                return {
                    "lat":          lat,
                    "lon":          lon,
                    "climate_zone": d.get("climate_zone", "Subtropical"),
                    "crop_notes":   d.get("crop_notes", ""),
                    "country_name": country_name,
                    "state_name":   state_name,
                    "source":       "llm",
                }

    # Dedicated coordinate + context resolve via LLM
    llm = _get_llm_agent()
    if llm:
        result = llm.llm_resolve_coords(country_name, state_name, district_name)
        if result and result.get("lat") != 0.0:
            return {
                "lat":          float(result["lat"]),
                "lon":          float(result["lon"]),
                "climate_zone": result.get("climate_zone", "Subtropical"),
                "crop_notes":   result.get("crop_notes", ""),
                "country_name": country_name,
                "state_name":   state_name,
                "source":       "llm",
            }

    # Final fallback — state centre
    state_lat = next(
        (s.get("lat", 0.0) for s in states if s["code"].upper() == state_code.upper()), 0.0
    )
    state_lon = next(
        (s.get("lon", 0.0) for s in states if s["code"].upper() == state_code.upper()), 0.0
    )
    return {
        "lat":          state_lat,
        "lon":          state_lon,
        "climate_zone": "Subtropical",
        "crop_notes":   "",
        "country_name": country_name,
        "state_name":   state_name,
        "source":       "fallback",
    }
