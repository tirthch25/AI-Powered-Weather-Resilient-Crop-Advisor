"""
LLM Location Agent
==================
Resolves states, districts, and coordinates for ANY country on Earth
using the existing Gemini/LLaMA agents — no new API keys required.

Called as a fallback by location_agent.py when a country is not in
the bundled world_locations.json (which covers the top 50 countries
in detail). This module makes the system truly global: all 195 UN
countries work out of the box.

Cache: 24-hour in-memory TTL (location names rarely change).

Fixes applied:
  - google.genai client must NOT run inside ThreadPoolExecutor: the
    httpx async client is closed when the worker thread exits, causing
    "Cannot send a request, as the client has been closed" errors.
    Gemini is now called directly; only Ollama uses the executor.
  - Model fallback list: gemini-2.5-flash-lite → gemini-2.0-flash-lite
    → gemini-flash-lite-latest to survive per-model quota limits.
"""

import os
import re
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]
GEMINI_KEY = GEMINI_KEYS[0] if GEMINI_KEYS else ""  # kept for backward compat
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Ordered list of Gemini models to try (free-tier first, lite models preferred)
_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# 24-hour cache — location data is stable
_CACHE: Dict[str, Tuple[float, object]] = {}
_CACHE_TTL = 24 * 3600


def _cached(key: str, fn):
    ts, val = _CACHE.get(key, (0, None))
    if val is not None and (time.time() - ts) < _CACHE_TTL:
        return val
    result = fn()
    if result:
        _CACHE[key] = (time.time(), result)
    return result


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_gemini_direct(prompt: str) -> Optional[str]:
    """
    Call Gemini directly (no ThreadPoolExecutor).
    Rotates through all available API keys and models on 429 quota errors.
    """
    if not GEMINI_KEYS:
        return None

    # Try new google.genai SDK first
    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_MODELS:
                try:
                    r = client.models.generate_content(model=model, contents=prompt)
                    text = r.text.strip() if r.text else None
                    if text:
                        logger.debug("[LLMLocationAgent] Gemini model %s (key ...%s) succeeded",
                                     model, api_key[-6:])
                        return text
                except Exception as model_err:
                    err_str = str(model_err)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logger.warning("[LLMLocationAgent] %s quota on key ...%s, trying next",
                                       model, api_key[-6:])
                        continue
                    elif "404" in err_str or "NOT_FOUND" in err_str:
                        logger.debug("[LLMLocationAgent] %s not found, trying next model", model)
                        continue
                    else:
                        logger.warning("[LLMLocationAgent] %s error: %s", model, err_str[:120])
                        continue
    except ImportError:
        pass  # fall through to legacy SDK

    # Fallback: legacy google.generativeai SDK
    try:
        import google.generativeai as genai
        legacy_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
        for api_key in GEMINI_KEYS:
            genai.configure(api_key=api_key)
            for model in legacy_models:
                try:
                    r = genai.GenerativeModel(model).generate_content(prompt)
                    text = r.text.strip() if r.text else None
                    if text:
                        return text
                except Exception as model_err:
                    err_str = str(model_err)
                    if "429" in err_str or "quota" in err_str.lower():
                        continue
                    logger.warning("[LLMLocationAgent] legacy SDK %s: %s", model, err_str[:120])
                    continue
    except ImportError:
        pass

    return None


def _call_ollama(prompt: str, timeout: int = 15) -> Optional[str]:
    """Call Ollama in a thread with timeout."""
    def _run():
        try:
            import ollama as _o
            r = _o.Client(host=OLLAMA_URL).chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return r["message"]["content"]
        except Exception as e:
            logger.debug("[LLMLocationAgent] Ollama error: %s", e)
            return None

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=timeout)
    except (FutTimeout, Exception) as e:
        logger.warning("[LLMLocationAgent] Ollama timed out / unavailable: %s", e)
        return None


def _call_llm(prompt: str, timeout: int = 20) -> Optional[str]:
    """Try Gemini (direct, no executor), then Ollama. Returns raw text or None."""
    # 1. Gemini — synchronous call, no thread pool needed
    result = _call_gemini_direct(prompt)
    if result:
        return result

    # 2. Ollama — uses a thread pool only for timeout control
    result = _call_ollama(prompt, timeout=timeout)
    if result:
        return result

    logger.warning("[LLMLocationAgent] All LLM providers failed or returned empty")
    return None


def _extract_json_array(text: str) -> Optional[List]:
    """Robustly pull a JSON array out of LLM output."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    # Try direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    # Find first [...] block
    m = re.search(r"\[[\s\S]*?\]", cleaned)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    # Try to find any JSON array even with nested content
    m = re.search(r"\[[\s\S]+\]", cleaned)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def _extract_json_obj(text: str) -> Optional[Dict]:
    """Robustly pull a JSON object out of LLM output."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*?\}", cleaned)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def llm_get_states(country_name: str, country_code: str) -> List[Dict]:
    """
    Ask the LLM for the major agricultural states/provinces of a country.
    Returns list of {code, name, lat, lon}.
    Results are cached for 24 hours.
    """
    cache_key = "states_" + country_code.upper()

    def _fetch():
        prompt = (
            "List ALL official administrative states, provinces, or regions of " + country_name + ". "
            "Include every single one — do not skip any. "
            "Return ONLY a JSON array with no explanation and no markdown. "
            "Each element must have these exact keys: "
            "code (official 2-4 letter ISO or standard abbreviation), "
            "name (full official state/province name), "
            "lat (geographic center latitude as float), "
            "lon (geographic center longitude as float). "
            "Return ALL states/provinces, not just agricultural ones. "
            'Example format: [{"code":"BY","name":"Bavaria","lat":48.7,"lon":11.5}]'
        )
        raw = _call_llm(prompt, timeout=20)
        items = _extract_json_array(raw)
        if not items:
            logger.warning("[LLMLocationAgent] Could not parse states for %s. Raw: %s",
                           country_name, repr(raw)[:200] if raw else "None")
            return None
        # Validate and clean
        result = []
        for item in items:
            if isinstance(item, dict) and item.get("name"):
                result.append({
                    "code": str(item.get("code", item["name"][:3].upper())).upper(),
                    "name": str(item["name"]),
                    "lat":  float(item.get("lat", 0.0)),
                    "lon":  float(item.get("lon", 0.0)),
                })
        logger.info("[LLMLocationAgent] Got %d states for %s via LLM", len(result), country_name)
        return result if result else None

    return _cached(cache_key, _fetch) or []


def llm_get_districts(country_name: str, state_name: str, state_code: str, country_code: str) -> List[Dict]:
    """
    Ask the LLM for key agricultural districts/cities in a state.
    Returns list of {name, lat, lon}.
    Results are cached for 24 hours.
    """
    cache_key = "districts_" + country_code.upper() + "_" + state_code.upper()

    def _fetch():
        prompt = (
            "List ALL districts, counties, cities, or administrative sub-regions in "
            + state_name + ", " + country_name + ". "
            "Include both agricultural and non-agricultural areas. "
            "Return ONLY a JSON array with no explanation and no markdown. "
            "Each element must have: "
            "name (district/county/city name), "
            "lat (geographic center latitude as float), "
            "lon (geographic center longitude as float). "
            "Return 10 to 25 districts/regions covering the whole state. "
            'Example format: [{"name":"Fresno","lat":36.7,"lon":-119.8}]'
        )
        raw = _call_llm(prompt, timeout=20)
        items = _extract_json_array(raw)
        if not items:
            logger.warning("[LLMLocationAgent] Could not parse districts for %s, %s. Raw: %s",
                           state_name, country_name, repr(raw)[:200] if raw else "None")
            return None
        result = []
        for item in items:
            if isinstance(item, dict) and item.get("name"):
                result.append({
                    "name": str(item["name"]),
                    "lat":  float(item.get("lat", 0.0)),
                    "lon":  float(item.get("lon", 0.0)),
                })
        logger.info("[LLMLocationAgent] Got %d districts for %s/%s via LLM",
                    len(result), state_name, country_name)
        return result if result else None

    return _cached(cache_key, _fetch) or []


def llm_resolve_coords(
    country: str,
    state: str,
    district: str,
) -> Dict:
    """
    Ask the LLM for the geographic coordinates + farming context of a location.
    Returns dict with: lat, lon, climate_zone, crop_notes.
    Used when the location is not in world_locations.json.
    """
    cache_key = "coords_" + "_".join([country, state, district]).lower().replace(" ", "_")

    def _fetch():
        location_str = district + ", " + state + ", " + country
        prompt = (
            "Provide geographic and farming context for this location: " + location_str + ". "
            "Return ONLY a JSON object with no explanation and no markdown. "
            "Keys required: "
            "lat (decimal latitude), "
            "lon (decimal longitude), "
            "climate_zone (one of: Tropical, Subtropical, Arid, Mediterranean, Temperate, Continental), "
            "crop_notes (1-2 sentences about major crops grown in this region). "
            'Example: {"lat":12.97,"lon":77.59,"climate_zone":"Tropical",'
            '"crop_notes":"Major crops include coffee, ragi, and vegetables."}'
        )
        raw = _call_llm(prompt, timeout=20)
        obj = _extract_json_obj(raw)
        if not obj or "lat" not in obj:
            logger.warning("[LLMLocationAgent] Could not resolve coords for %s", location_str)
            return None
        result = {
            "lat":          float(obj.get("lat", 0.0)),
            "lon":          float(obj.get("lon", 0.0)),
            "climate_zone": str(obj.get("climate_zone", "Subtropical")),
            "crop_notes":   str(obj.get("crop_notes", "")),
        }
        logger.info("[LLMLocationAgent] Resolved %s -> lat=%.2f, lon=%.2f, zone=%s",
                    location_str, result["lat"], result["lon"], result["climate_zone"])
        return result

    return _cached(cache_key, _fetch) or {
        "lat": 0.0, "lon": 0.0,
        "climate_zone": "Subtropical",
        "crop_notes": "",
    }


def clear_cache():
    """Clear the in-memory location cache (useful for testing)."""
    _CACHE.clear()
    logger.info("[LLMLocationAgent] Cache cleared")
