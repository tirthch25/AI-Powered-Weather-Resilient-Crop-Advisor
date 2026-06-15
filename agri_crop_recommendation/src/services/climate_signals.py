"""
Climate Intelligence Service — v2 (Comprehensive)
===================================================
Covers ALL climate threats affecting agriculture, not just ENSO:

  1. ENSO (El Niño / La Niña)         — NOAA CPC ONI index (free)
  2. Drought Index                     — Standardised Precipitation Index proxy
  3. Heat Stress                       — Live temperature vs historical baseline
  4. Frost / Cold Stress               — Minimum temperature anomaly
  5. Flood / Extreme Rainfall          — 7-day rainfall vs seasonal norm
  6. Cyclone / Typhoon / Hurricane     — Basin-specific season awareness
  7. Wildfire Risk                     — Fuel moisture + wind + temperature
  8. Soil Moisture Stress              — Derived from rainfall deficit/surplus
  9. Regional Climate Change Trend     — Gemini Search Grounding for current advisories

All data is sourced from:
  - NOAA CPC (ONI index, ENSO advisory) — free, no API key
  - Open-Meteo (live weather already fetched upstream)
  - Gemini Search Grounding — real-time regional climate advisories
"""

import os, re, json, time, logging, datetime, requests
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]
GEMINI_KEY   = GEMINI_KEYS[0] if GEMINI_KEYS else ""
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Cache TTL: 6 hours for ENSO/climate (NOAA updates monthly, but searches are live)
_CACHE: Dict = {}
_CACHE_TTL   = 6 * 3600


# ═══════════════════════════════════════════════════════════════════════════════
# ENSO data + zone mapping (unchanged from v1 — research-based)
# ═══════════════════════════════════════════════════════════════════════════════

_ENSO_IMPACTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "El Nino": {
        "India":              (-0.20, +0.5),
        "Subtropical":        (-0.15, +0.4),
        "Tropical":           (-0.10, +0.3),
        "Arid":               (-0.05, +0.6),
        "Mediterranean":      (+0.05, +0.3),
        "Temperate":          (+0.05, +0.2),
        "Continental":        (+0.05, +0.1),
        "China":              (+0.25, +0.8),
        "South_China":        (+0.30, +0.9),
        "East_Asia":          (+0.15, +0.6),
        "Southeast_Asia":     (-0.10, +0.5),
        "Temperate_Americas": (-0.05, +0.3),
        "Tropical_Americas":  (-0.15, +0.4),
        "Subtropical_S":      (-0.10, +0.5),
        "Arid_Oceania":       (-0.20, +1.0),
    },
    "La Nina": {
        "India":              (+0.20, -0.3),
        "Subtropical":        (+0.15, -0.2),
        "Tropical":           (+0.10, -0.1),
        "Arid":               (+0.05, -0.2),
        "Mediterranean":      (-0.05, -0.1),
        "Temperate":          (-0.05, -0.2),
        "Continental":        (-0.05, -0.3),
        "China":              (-0.10, -0.3),
        "South_China":        (-0.10, -0.2),
        "East_Asia":          (-0.05, -0.4),
        "Southeast_Asia":     (+0.15, -0.2),
        "Temperate_Americas": (+0.10, -0.2),
        "Tropical_Americas":  (+0.20, -0.2),
        "Subtropical_S":      (+0.15, -0.3),
        "Arid_Oceania":       (+0.25, -0.5),
    },
    "El Nino Watch": {
        "India":          (-0.10, +0.3), "Subtropical":    (-0.08, +0.2),
        "Continental":    (+0.02, +0.1), "Temperate":      (+0.02, +0.1),
        "China":          (+0.15, +0.5), "South_China":    (+0.18, +0.6),
        "East_Asia":      (+0.08, +0.3), "Southeast_Asia": (-0.05, +0.3),
        "Arid_Oceania":   (-0.10, +0.5),
    },
    "La Nina Watch": {
        "India":        (+0.10, -0.15), "Subtropical":  (+0.08, -0.10),
        "China":        (-0.05, -0.15), "South_China":  (-0.05, -0.10),
        "Arid_Oceania": (+0.12, -0.25), "Temperate":    (-0.02, -0.10),
        "Continental":  (-0.02, -0.15),
    },
    "Neutral": {},
}

_COUNTRY_TO_ZONE_KEY: Dict[str, str] = {
    "china": "South_China", "hong kong": "South_China",
    "macau": "South_China", "taiwan": "South_China",
    "japan": "East_Asia", "south korea": "East_Asia", "north korea": "East_Asia",
    "thailand": "Southeast_Asia", "vietnam": "Southeast_Asia",
    "cambodia": "Southeast_Asia", "myanmar": "Southeast_Asia",
    "laos": "Southeast_Asia", "philippines": "Southeast_Asia",
    "indonesia": "Southeast_Asia", "malaysia": "Southeast_Asia",
    "singapore": "Southeast_Asia",
    "india": "India",
    "australia": "Arid_Oceania", "new zealand": "Arid_Oceania",
}

_SOUTH_CHINA_REGIONS = {
    "guangdong", "guangzhou", "shenzhen", "dongguan", "foshan",
    "zhuhai", "zhongshan", "jiangmen", "guangxi", "hainan",
    "fujian", "xiamen", "hong kong", "macau", "shantou",
    "chaozhou", "huizhou", "zhanjiang", "pearl river", "delta",
}


def _oni_to_phase_and_strength(oni: float) -> Tuple[str, str]:
    if oni >= 2.0:    return "El Nino", "Strong"
    elif oni >= 1.0:  return "El Nino", "Moderate"
    elif oni >= 0.5:  return "El Nino", "Weak"
    elif oni >= 0.3:  return "El Nino Watch", "Developing"
    elif oni <= -2.0: return "La Nina", "Strong"
    elif oni <= -1.0: return "La Nina", "Moderate"
    elif oni <= -0.5: return "La Nina", "Weak"
    elif oni <= -0.3: return "La Nina Watch", "Developing"
    else:             return "Neutral", "Neutral"


def _fetch_noaa_oni() -> Optional[float]:
    try:
        resp = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=10)
        resp.raise_for_status()
        for line in reversed(resp.text.strip().splitlines()):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
    except Exception as e:
        logger.warning("[ClimateSignals] NOAA ONI fetch failed: %s", e)
    return None


def _fetch_noaa_enso_text() -> str:
    try:
        resp = requests.get(
            "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml",
            timeout=10,
        )
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        return re.sub(r"\s+", " ", text).strip()[:1500]
    except Exception as e:
        logger.warning("[ClimateSignals] NOAA advisory fetch failed: %s", e)
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive climate threat assessment
# ═══════════════════════════════════════════════════════════════════════════════

# Cyclone/typhoon basin season awareness
_CYCLONE_BASINS = {
    # region_keywords: (basin_name, peak_months)
    "north_atlantic":  (
        ["united states", "mexico", "cuba", "bahamas", "caribbean", "florida",
         "texas", "louisiana", "north carolina", "gulf of mexico", "haiti",
         "dominican republic", "puerto rico", "jamaica"],
        "Jun-Nov", "Hurricane"
    ),
    "eastern_pacific": (
        ["mexico pacific", "baja california", "jalisco", "nayarit",
         "sinaloa", "sonora", "colima", "michoacan", "guerrero", "oaxaca",
         "el salvador", "guatemala", "honduras", "nicaragua", "costa rica"],
        "May-Nov", "Hurricane"
    ),
    "western_pacific": (
        ["china", "japan", "philippines", "taiwan", "vietnam", "south korea",
         "hong kong", "macau", "shenzhen", "guangdong", "hainan", "guangxi",
         "fujian", "guam", "micronesia", "marshall islands"],
        "Jun-Nov", "Typhoon"
    ),
    "north_indian_bay": (
        ["bangladesh", "myanmar", "india", "odisha", "andhra pradesh",
         "west bengal", "sri lanka", "andaman", "nicobar"],
        "Apr-Jun & Oct-Dec", "Cyclone"
    ),
    "north_indian_arabian": (
        ["india", "gujarat", "maharashtra", "goa", "karnataka", "kerala",
         "pakistan", "oman", "yemen", "somalia"],
        "Apr-Jun & Oct-Dec", "Cyclone"
    ),
    "south_indian": (
        ["madagascar", "mozambique", "tanzania", "kenya", "réunion",
         "mauritius", "comoros", "malawi", "zimbabwe"],
        "Nov-Apr", "Cyclone"
    ),
    "south_pacific": (
        ["australia", "fiji", "vanuatu", "new caledonia", "tonga",
         "samoa", "solomon islands", "papua new guinea"],
        "Nov-Apr", "Cyclone/Typhoon"
    ),
}

def _get_cyclone_context(location_str: str, country: str) -> Optional[dict]:
    """Return cyclone/typhoon/hurricane context if region is in a basin."""
    loc_lower = (location_str + " " + country).lower()
    for basin_key, (keywords, season, storm_type) in _CYCLONE_BASINS.items():
        if any(kw in loc_lower for kw in keywords):
            now_month = datetime.datetime.now().month
            # Parse peak months roughly
            in_season = False
            if "Jun-Nov" in season and 6 <= now_month <= 11:
                in_season = True
            elif "May-Nov" in season and 5 <= now_month <= 11:
                in_season = True
            elif "Nov-Apr" in season and (now_month >= 11 or now_month <= 4):
                in_season = True
            elif "Apr-Jun" in season and 4 <= now_month <= 6:
                in_season = True
            elif "Oct-Dec" in season and 10 <= now_month <= 12:
                in_season = True
            return {
                "storm_type": storm_type,
                "season": season,
                "in_active_season": in_season,
                "basin": basin_key,
            }
    return None


def _assess_heat_stress(current_temp: float, climate_zone: str) -> Optional[dict]:
    """Assess agricultural heat stress based on current temperature."""
    # Heat stress thresholds by crop type
    thresholds = {
        "Temperate":     {"moderate": 30, "severe": 35, "extreme": 40},
        "Continental":   {"moderate": 30, "severe": 35, "extreme": 40},
        "Mediterranean": {"moderate": 32, "severe": 38, "extreme": 42},
        "Subtropical":   {"moderate": 35, "severe": 40, "extreme": 44},
        "Tropical":      {"moderate": 36, "severe": 41, "extreme": 45},
        "Arid":          {"moderate": 38, "severe": 43, "extreme": 47},
    }
    zone_thresh = thresholds.get(climate_zone, thresholds["Subtropical"])

    if current_temp is None:
        return None
    if current_temp >= zone_thresh["extreme"]:
        return {"level": "Extreme", "temp": current_temp,
                "message": f"Extreme heat ({current_temp}°C) — severe crop damage risk for most crops. Irrigation and shade essential.",
                "affected_crops": ["wheat", "maize", "tomato", "potato", "all leafy greens"]}
    elif current_temp >= zone_thresh["severe"]:
        return {"level": "Severe", "temp": current_temp,
                "message": f"Severe heat stress ({current_temp}°C) — pollen sterility risk for cereals and legumes.",
                "affected_crops": ["wheat", "rice", "chickpea", "soybean"]}
    elif current_temp >= zone_thresh["moderate"]:
        return {"level": "Moderate", "temp": current_temp,
                "message": f"Moderate heat stress ({current_temp}°C) — monitor soil moisture and apply mulch.",
                "affected_crops": ["lettuce", "spinach", "pea", "potato"]}
    return None


def _assess_drought(rainfall_7d: float, climate_zone: str) -> Optional[dict]:
    """Assess drought risk based on 7-day rainfall deficit."""
    # Expected 7-day rainfall by zone (mm) — low = drought threshold
    zone_norms = {
        "Tropical": 60, "Subtropical": 30, "Arid": 5,
        "Mediterranean": 20, "Temperate": 25, "Continental": 20,
        "Temperate_Americas": 28, "Tropical_Americas": 65,
        "Subtropical_S": 25, "Arid_Oceania": 8,
    }
    norm = zone_norms.get(climate_zone, 25)
    if rainfall_7d is None:
        return None
    deficit_pct = 100 * (1 - rainfall_7d / norm) if norm > 0 else 0
    if deficit_pct >= 80:
        return {"level": "Severe", "rainfall_7d": rainfall_7d, "deficit_pct": round(deficit_pct),
                "message": f"Severe drought stress — only {rainfall_7d}mm in last 7 days vs {norm}mm norm. Immediate irrigation needed."}
    elif deficit_pct >= 50:
        return {"level": "Moderate", "rainfall_7d": rainfall_7d, "deficit_pct": round(deficit_pct),
                "message": f"Moderate drought risk — rainfall {round(deficit_pct)}% below normal. Schedule irrigation."}
    elif rainfall_7d > norm * 2.5:
        return {"level": "Excess", "rainfall_7d": rainfall_7d,
                "message": f"Excess rainfall ({rainfall_7d}mm in 7 days vs {norm}mm norm) — waterlogging and root disease risk."}
    return None


def _assess_frost(current_temp: float, climate_zone: str) -> Optional[dict]:
    """Assess frost risk."""
    if current_temp is None:
        return None
    if current_temp <= 0:
        return {"level": "Frost", "temp": current_temp,
                "message": f"Frost conditions ({current_temp}°C) — protect tender crops immediately. Cover seedlings overnight."}
    elif current_temp <= 4:
        return {"level": "Near-Frost", "temp": current_temp,
                "message": f"Near-frost temperatures ({current_temp}°C) — high risk for tropical/subtropical crops. Monitor overnight lows."}
    return None


def _assess_wildfire(current_temp: float, rainfall_7d: float, climate_zone: str) -> Optional[dict]:
    """Assess wildfire/fire risk for agricultural areas."""
    fire_zones = {"Mediterranean", "Arid", "Temperate_Americas", "Subtropical_S", "Arid_Oceania", "Subtropical"}
    if climate_zone not in fire_zones:
        return None
    if current_temp is None or rainfall_7d is None:
        return None
    # High fire risk: hot + dry
    if current_temp >= 35 and rainfall_7d < 5:
        return {"level": "High", "temp": current_temp, "rainfall_7d": rainfall_7d,
                "message": f"High wildfire/fire risk — hot ({current_temp}°C) and dry ({rainfall_7d}mm/week). Protect field boundaries."}
    elif current_temp >= 30 and rainfall_7d < 10:
        return {"level": "Moderate", "temp": current_temp, "rainfall_7d": rainfall_7d,
                "message": f"Moderate fire risk — warm and dry conditions. Clear field edges and maintain firebreaks."}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini Search Grounding — real-time regional climate advisories
# ═══════════════════════════════════════════════════════════════════════════════

_SEARCH_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-001"]
_PLAIN_MODELS  = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.0-flash"]


def _gemini_comprehensive_climate(
    location_str: str,
    climate_zone: str,
    enso_phase: str,
    enso_strength: str,
    oni_value: float,
    advisory_text: str,
    heat_stress: Optional[dict],
    drought: Optional[dict],
    frost: Optional[dict],
    wildfire: Optional[dict],
    cyclone: Optional[dict],
) -> Optional[dict]:
    """
    Ask Gemini (with Google Search Grounding) for a comprehensive climate
    intelligence report covering ALL active threats at this location.
    """
    if not GEMINI_KEYS:
        return None

    # Build context for all active threats
    threats_context = []
    if enso_phase != "Neutral":
        threats_context.append(f"ENSO: {enso_phase} ({enso_strength}), ONI={oni_value}")
    if heat_stress:
        threats_context.append(f"Heat stress: {heat_stress['level']} ({heat_stress['temp']}°C)")
    if drought:
        threats_context.append(f"Drought: {drought['level']}")
    if frost:
        threats_context.append(f"Frost risk: {frost['level']} ({frost['temp']}°C)")
    if wildfire:
        threats_context.append(f"Fire risk: {wildfire['level']}")
    if cyclone:
        in_season = "IN ACTIVE SEASON" if cyclone["in_active_season"] else "off-season"
        threats_context.append(f"{cyclone['storm_type']} basin ({in_season}, peak: {cyclone['season']})")

    threats_str = "; ".join(threats_context) if threats_context else "No major active threats detected"

    prompt = (
        f"You are a senior agroclimate analyst. Search for current climate conditions "
        f"and agricultural advisories for {location_str} (climate zone: {climate_zone}).\n\n"
        f"Active threats detected: {threats_str}\n"
        f"NOAA ENSO advisory (partial): {advisory_text[:400] if advisory_text else 'N/A'}\n\n"
        f"Search for:\n"
        f"1. Current drought index or rainfall anomaly for {location_str}\n"
        f"2. Active heat waves, cold snaps, or extreme weather for {location_str}\n"
        f"3. Any government crop advisories or agricultural warnings for the region\n"
        f"4. Climate change trends specifically affecting agriculture in {location_str}\n"
        f"5. Active pest/disease outbreaks linked to current climate\n\n"
        f"Provide a comprehensive analysis for a farmer at {location_str}.\n"
        f"Return ONLY valid compact JSON (no markdown):\n"
        f'{{"summary":"<3 sentences covering ALL active climate threats at this specific location>",'
        f'"enso_impact":"<1 sentence on ENSO effect specifically for this region>",'
        f'"heat_stress_risk":"<None/Low/Moderate/Severe/Extreme>",'
        f'"drought_risk":"<None/Low/Moderate/Severe>",'
        f'"flood_risk":"<None/Low/Moderate/High>",'
        f'"frost_risk":"<None/Low/Moderate/High>",'
        f'"cyclone_risk":"<None/Low/Moderate/High>",'
        f'"wildfire_risk":"<None/Low/Moderate/High>",'
        f'"climate_change_trend":"<1 sentence on long-term trend for this region>",'
        f'"crop_risks":["<specific risk 1>","<specific risk 2>","<specific risk 3>","<risk 4>","<risk 5>"],'
        f'"immediate_actions":["<action 1 for THIS week>","<action 2>","<action 3>"],'
        f'"seasonal_outlook":"<2 sentences on next 3 months>",'
        f'"alert_level":"<None/Advisory/Watch/Warning/Emergency>",'
        f'"rainfall_outlook":"<Below Normal/Near Normal/Above Normal>",'
        f'"temp_outlook":"<Below Normal/Near Normal/Above Normal>"}}'
    )

    # Try search-grounded models first
    try:
        from google import genai as _g
        from google.genai import types as _gt
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _SEARCH_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=_gt.GenerateContentConfig(
                            tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                        ),
                    )
                    text = resp.text.strip() if resp.text else None
                    result = _extract_json(text) if text else None
                    if result and isinstance(result, dict) and result.get("summary"):
                        logger.info("[ClimateSignals] Gemini search-grounded OK (%s)", model)
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    if "not supported" in err.lower() or "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[ClimateSignals] Search model %s: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[ClimateSignals] Search grounding failed: %s", e)

    # Fall back to plain Gemini (no search)
    plain_prompt = prompt.replace(
        "Search for:\n",
        "Based on your knowledge, analyze:\n",
    ).replace(
        "Search for current climate conditions and agricultural advisories for",
        "Analyze the climate and agricultural situation for",
    )

    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _PLAIN_MODELS:
                try:
                    resp = client.models.generate_content(model=model, contents=plain_prompt)
                    text = resp.text.strip() if resp.text else None
                    result = _extract_json(text) if text else None
                    if result and isinstance(result, dict) and result.get("summary"):
                        logger.info("[ClimateSignals] Gemini plain OK (%s)", model)
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    logger.debug("[ClimateSignals] Plain %s: %s", model, err[:80])
                    continue
    except ImportError:
        pass

    # Final fallback: Ollama
    try:
        import ollama as _o
        r = _o.Client(host=OLLAMA_URL).chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": plain_prompt}],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        return _extract_json(r["message"]["content"])
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback interpretation (no LLM)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_fallback(
    enso_phase: str, enso_strength: str, climate_zone: str,
    heat_stress: Optional[dict], drought: Optional[dict],
    frost: Optional[dict], wildfire: Optional[dict], cyclone: Optional[dict],
    location_str: str,
) -> dict:
    """Build a comprehensive fallback without any LLM."""
    risks = []
    actions = []
    alert = "None"
    summary_parts = []

    # ENSO contribution
    if enso_phase == "El Nino":
        summary_parts.append(f"El Niño ({enso_strength}) is active and will affect rainfall and temperature patterns for {location_str}.")
        if "India" in climate_zone or "Subtropical" in climate_zone:
            risks += ["Reduced monsoon rainfall — drought risk for Kharif crops", "Heat stress at crop flowering stages"]
            actions.append("Plant drought-tolerant crop varieties and ensure irrigation backup.")
            alert = "Watch"
        elif "Tropical" in climate_zone:
            risks += ["Below-normal rainfall — water deficit for paddy", "Heat stress on tropical crops"]
            alert = "Watch"
        elif "South_China" in climate_zone or "China" in climate_zone:
            risks += ["Typhoon activity intensified", "Flash flooding in farm areas", "Heat stress on vegetables"]
            actions.append("Reinforce greenhouse structures and improve field drainage before typhoon season.")
            alert = "Warning"
        else:
            risks.append("Drier and warmer than normal conditions likely")
    elif enso_phase == "La Nina":
        summary_parts.append(f"La Niña ({enso_strength}) is active, typically bringing wetter and cooler conditions to {location_str}.")
        risks += ["Waterlogging risk for root crops", "Fungal disease pressure increases with excess moisture"]
        actions.append("Ensure field drainage channels are clear to prevent waterlogging.")
        alert = "Watch"
    else:
        summary_parts.append(f"ENSO is currently neutral — no major El Niño or La Niña influence at {location_str}.")

    # Heat stress
    if heat_stress:
        risks.append(f"Heat stress ({heat_stress['level']}) — {heat_stress['message']}")
        actions.append("Apply mulch, increase irrigation frequency, consider shade nets for heat-sensitive crops.")
        if heat_stress["level"] in ("Severe", "Extreme"):
            alert = max(alert, "Warning", key=lambda x: ["None","Advisory","Watch","Warning","Emergency"].index(x))

    # Drought
    if drought:
        if drought["level"] == "Excess":
            risks.append(f"Excess rainfall — waterlogging and root disease risk ({drought['rainfall_7d']}mm/week)")
            actions.append("Open drainage channels; delay sowing of waterlogging-sensitive crops.")
        else:
            risks.append(f"Drought stress ({drought['level']}) — {drought['message']}")
            actions.append("Activate irrigation, prioritize moisture-conserving tillage practices.")
            if drought["level"] == "Severe":
                alert = max(alert, "Warning", key=lambda x: ["None","Advisory","Watch","Warning","Emergency"].index(x))
            else:
                alert = max(alert, "Watch", key=lambda x: ["None","Advisory","Watch","Warning","Emergency"].index(x))

    # Frost
    if frost:
        risks.append(f"Frost risk — {frost['message']}")
        actions.append("Cover tender crops overnight. Delay transplanting of frost-sensitive seedlings.")
        alert = max(alert, "Watch", key=lambda x: ["None","Advisory","Watch","Warning","Emergency"].index(x))

    # Wildfire
    if wildfire:
        risks.append(f"Fire risk ({wildfire['level']}) — {wildfire['message']}")
        actions.append("Clear dry vegetation around field margins; prepare firebreaks.")

    # Cyclone
    if cyclone:
        storm = cyclone["storm_type"]
        if cyclone["in_active_season"]:
            risks.append(f"{storm} season active (peak: {cyclone['season']}) — high wind and flood damage risk to crops")
            actions.append(f"Monitor {storm.lower()} advisories daily. Harvest mature crops before any storm warning.")
            alert = max(alert, "Advisory", key=lambda x: ["None","Advisory","Watch","Warning","Emergency"].index(x))
        else:
            risks.append(f"{storm} basin — off-peak season, low risk currently")

    if not risks:
        risks = ["No major climate threats currently active", "Conditions appear near-normal for this season"]

    if not actions:
        actions = ["Continue regular crop monitoring", "Maintain irrigation schedule as planned"]

    summary = " ".join(summary_parts)
    if not summary:
        summary = f"Climate conditions at {location_str} are within normal seasonal range."

    # Determine outlooks
    el = enso_phase in ("El Nino", "El Nino Watch")
    la = enso_phase in ("La Nina", "La Nina Watch")
    cn = climate_zone in ("South_China", "China", "East_Asia")
    tropical_la = la and "Tropical" in climate_zone

    return {
        "summary":             summary,
        "enso_impact":         f"{enso_phase} ({enso_strength}) — see ENSO details below.",
        "heat_stress_risk":    heat_stress["level"] if heat_stress else "None",
        "drought_risk":        drought["level"] if drought else "None",
        "flood_risk":          "Moderate" if (la or (drought and drought["level"] == "Excess")) else "Low",
        "frost_risk":          frost["level"] if frost else "None",
        "cyclone_risk":        ("Moderate" if cyclone and cyclone["in_active_season"] else "Low") if cyclone else "None",
        "wildfire_risk":       wildfire["level"] if wildfire else "None",
        "climate_change_trend": "Long-term warming trend — increasing frequency of extreme weather events globally.",
        "crop_risks":          risks[:6],
        "immediate_actions":   actions[:4],
        "seasonal_outlook":    f"Next 3 months: {'Warmer and drier' if el else 'Cooler and wetter' if la else 'Near-normal'} conditions expected.",
        "alert_level":         alert,
        "rainfall_outlook":    "Above Normal" if (la or tropical_la or (cn and el)) else "Below Normal" if el else "Near Normal",
        "temp_outlook":        "Above Normal" if el else "Below Normal" if la else "Near Normal",
    }


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_climate_signals(
    location_str: str = "Unknown",
    climate_zone: str = "Subtropical",
    country: str = "india",
    current_temp: Optional[float] = None,
    rainfall_7d: Optional[float] = None,
) -> dict:
    """
    Get comprehensive climate intelligence for a location.
    Covers ENSO, heat stress, drought, frost, flood, cyclone, wildfire,
    and real-time advisories via Gemini Search Grounding.

    Args:
        location_str:  Full location string e.g. "Munich, Bavaria, Germany"
        climate_zone:  Zone key e.g. "Temperate"
        country:       Country name (lowercase)
        current_temp:  Current temperature °C (from live weather, if available)
        rainfall_7d:   Rainfall last 7 days mm (from live weather, if available)
    """
    country_lc  = country.lower().strip()
    location_lc = location_str.lower()

    # Determine ENSO zone key
    if any(r in location_lc for r in _SOUTH_CHINA_REGIONS):
        zone_key = "South_China"
    else:
        zone_key = _COUNTRY_TO_ZONE_KEY.get(country_lc, climate_zone)

    cache_key = (zone_key, country_lc, round(current_temp or 25), round(rainfall_7d or 30))
    cached_ts, cached_val = _CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CACHE_TTL:
        logger.info("[ClimateSignals] Cache hit for %s", location_str)
        return cached_val

    logger.info("[ClimateSignals] Full climate assessment for %s (zone=%s)", location_str, zone_key)

    # ── 1. ENSO ──────────────────────────────────────────────────────────────
    oni_value = _fetch_noaa_oni()
    advisory_text = ""
    if oni_value is None:
        logger.warning("[ClimateSignals] NOAA unavailable, defaulting Neutral")
        oni_value = 0.0
    else:
        try:
            advisory_text = _fetch_noaa_enso_text()
        except Exception:
            pass

    enso_phase, enso_strength = _oni_to_phase_and_strength(oni_value)
    impacts = _ENSO_IMPACTS.get(enso_phase, {})
    rain_factor, temp_offset = impacts.get(zone_key, impacts.get(climate_zone, (0.0, 0.0)))

    # ── 2. Local threat assessments ───────────────────────────────────────────
    heat_stress = _assess_heat_stress(current_temp, climate_zone)
    drought     = _assess_drought(rainfall_7d, climate_zone)
    frost       = _assess_frost(current_temp, climate_zone)
    wildfire    = _assess_wildfire(current_temp, rainfall_7d, climate_zone)
    cyclone     = _get_cyclone_context(location_str, country)

    # ── 3. Comprehensive LLM interpretation (search-grounded) ─────────────────
    ai_result = _gemini_comprehensive_climate(
        location_str, climate_zone, enso_phase, enso_strength, oni_value,
        advisory_text, heat_stress, drought, frost, wildfire, cyclone,
    )

    if not ai_result:
        ai_result = _build_fallback(
            enso_phase, enso_strength, climate_zone,
            heat_stress, drought, frost, wildfire, cyclone, location_str,
        )

    # ── 4. Ensure all required keys exist ────────────────────────────────────
    ai_result.setdefault("heat_stress_risk",    heat_stress["level"] if heat_stress else "None")
    ai_result.setdefault("drought_risk",        drought["level"] if drought else "None")
    ai_result.setdefault("flood_risk",          "Low")
    ai_result.setdefault("frost_risk",          frost["level"] if frost else "None")
    ai_result.setdefault("cyclone_risk",        ("Moderate" if cyclone and cyclone["in_active_season"] else "Low") if cyclone else "None")
    ai_result.setdefault("wildfire_risk",       wildfire["level"] if wildfire else "None")
    ai_result.setdefault("climate_change_trend","Long-term warming trend increasing extreme weather frequency.")
    ai_result.setdefault("immediate_actions",   ["Monitor local weather forecasts daily"])
    ai_result.setdefault("seasonal_outlook",    "Monitor seasonal forecasts from your national meteorological service.")
    ai_result.setdefault("alert_level",         "None")

    # ── 5. Build phase labels ─────────────────────────────────────────────────
    if   enso_phase == "El Nino":       phase_label = "El Niño (" + enso_strength + ")"
    elif enso_phase == "El Nino Watch": phase_label = "El Niño Watch (Developing)"
    elif enso_phase == "La Nina":       phase_label = "La Niña (" + enso_strength + ")"
    elif enso_phase == "La Nina Watch": phase_label = "La Niña Watch (Developing)"
    else:                               phase_label = "Neutral (Normal)"

    result = {
        # ENSO core
        "enso_phase":    enso_phase,
        "enso_strength": enso_strength,
        "oni_value":     round(oni_value, 2),
        "phase_label":   phase_label,
        "enso_zone_key": zone_key,

        # Local threat flags (for frontend badges)
        "threats": {
            "heat_stress":  heat_stress,
            "drought":      drought,
            "frost":        frost,
            "wildfire":     wildfire,
            "cyclone":      cyclone,
        },

        # Comprehensive AI interpretation
        "ai_interpretation": ai_result,

        # Forecast adjustments (used by data_gathering_agent)
        "forecast_adjustments": {
            "rainfall_factor": round(1.0 + rain_factor, 3),
            "temp_offset_c":   temp_offset,
            "description": (
                "Rainfall adjusted by " + str(int(rain_factor * 100)) + "%, "
                "temp adjusted by " + str(temp_offset) + " °C"
                if enso_phase != "Neutral" else "No ENSO adjustment (Neutral conditions)"
            ),
        },

        "fetched_at":     datetime.datetime.now().isoformat(),
        "source":         "NOAA CPC + Gemini Search Grounding + Live Weather",
        "data_freshness": "ENSO: monthly; live threats: real-time",
    }

    _CACHE[cache_key] = (time.time(), result)
    logger.info(
        "[ClimateSignals] %s ONI=%s | heat=%s drought=%s frost=%s cyclone=%s wildfire=%s",
        enso_phase, oni_value,
        heat_stress["level"] if heat_stress else "None",
        drought["level"] if drought else "None",
        frost["level"] if frost else "None",
        cyclone["storm_type"] if cyclone else "None",
        wildfire["level"] if wildfire else "None",
    )
    return result


def apply_enso_to_forecast(forecast_6month: list, climate_signals: dict) -> list:
    """Apply ENSO forecast adjustments to the 6-month forecast (unchanged)."""
    adj = climate_signals.get("forecast_adjustments", {})
    rain_factor = adj.get("rainfall_factor", 1.0)
    temp_offset = adj.get("temp_offset_c",  0.0)

    if rain_factor == 1.0 and temp_offset == 0.0:
        return forecast_6month

    adjusted = []
    for month in forecast_6month:
        m = dict(month)
        m["rainfall_mm"]   = round(m.get("rainfall_mm",   0) * rain_factor, 1)
        m["temp_avg"]      = round(m.get("temp_avg",      25) + temp_offset, 1)
        m["temp_max"]      = round(m.get("temp_max",      32) + temp_offset, 1)
        m["temp_min"]      = round(m.get("temp_min",      18) + temp_offset, 1)
        m["soil_temp_c"]   = round(m.get("soil_temp_c",   23) + temp_offset, 1)
        m["enso_adjusted"] = True
        adjusted.append(m)

    logger.info("[ClimateSignals] Applied %s to forecast: rain x%s, temp %s",
                climate_signals.get("enso_phase"), rain_factor, temp_offset)
    return adjusted
