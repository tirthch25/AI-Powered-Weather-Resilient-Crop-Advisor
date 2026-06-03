"""
LLM Farmer Chat
===============
Powers the /chat endpoint using a local LLaMA model (via Ollama) to answer
farmer questions grounded in full regional context (district, state, zone,
soil, weather, crops).

Primary provider : Ollama  (llama3.2 / gemma3 — runs locally, free, private)
Fallback provider: Gemini  (used automatically if Ollama is not running)

Set LLM_PROVIDER=ollama  (default) in .env to use Ollama.
Set LLM_PROVIDER=gemini  in .env to force Gemini even if Ollama is running.

Supports multi-turn conversation history and streaming (SSE) for real-time
token delivery. Falls back gracefully with a clear message if both providers
are unavailable.
"""

import os
import logging
import re
import time
from typing import Optional, List, Dict, Any, Tuple, Generator
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Provider configuration ───────────────────────────────────────────────────
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",    "ollama")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",    "llama3.2")
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_MODEL   = "gemini-2.5-flash-lite"

# Disable Gemini thinking for low-latency chat
_NO_THINK_CONFIG = {"thinking_config": {"thinking_budget": 0}}

# Max history turns to retain (fewer = less tokens = faster response)
_MAX_HISTORY_TURNS = 6

# ── Weather cache: avoid re-fetching on every chat turn ─────────────────────
_weather_cache: Dict[str, Dict[str, Any]] = {}
_WEATHER_CACHE_TTL = 300  # 5 minutes

# ── Cached clients ───────────────────────────────────────────────────────────
_ollama_client = None
_gemini_client = None


# ── System persona ───────────────────────────────────────────────────────────
_SYSTEM_PERSONA = (
    "You are an expert Indian agricultural advisor helping small farmers. "
    "Give practical, concise advice on crops, soil, pests, irrigation, market prices, and govt schemes. "
    "Tailor answers to the provided region/season/soil/weather context. "
    "IMPORTANT: When a 'Weather:' line is present in the Context block below, it contains LIVE real-time data "
    "fetched from Open-Meteo API. When the farmer asks about today's temperature or current weather, "
    "always quote the exact figures from the Weather line (e.g. '38°C high, 24°C low today'). "
    "Never say you don't have real-time data if the Weather line is present. "
    "Max 200 words. Use short bullet points. Be warm, jargon-free. Reply in English."
)


# ── Ollama client ────────────────────────────────────────────────────────────

def _get_ollama_client():
    """Return a live Ollama client or None if Ollama is not running."""
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        client.list()  # ping — raises ConnectionError if Ollama not running
        _ollama_client = client
        logger.info(f"Ollama client connected ({OLLAMA_MODEL} @ {OLLAMA_URL})")
        return _ollama_client
    except ImportError:
        logger.warning("ollama package not installed — run: pip install ollama")
        return None
    except Exception as e:
        logger.warning(f"Ollama not available ({e}) — will try Gemini fallback")
        return None


# ── Gemini fallback client ───────────────────────────────────────────────────

def _get_gemini_client():
    """Return a Gemini client or None if API key is missing."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
        logger.info(f"Gemini fallback client loaded ({GEMINI_MODEL})")
        return _gemini_client
    except Exception as e:
        logger.warning(f"Gemini fallback client failed: {e}")
        return None


def _resolve_client() -> Tuple[str, Any]:
    """
    Resolve which LLM provider to use.
    Returns (provider_name, client) where provider_name is 'ollama' or 'gemini'.
    Returns ('none', None) if neither is available.
    """
    if LLM_PROVIDER == "gemini":
        client = _get_gemini_client()
        if client:
            return ("gemini", client)
        # Even if forced to Gemini, fall through to Ollama if Gemini unavailable
        logger.warning("LLM_PROVIDER=gemini but Gemini unavailable — trying Ollama")

    # Try Ollama first (default)
    client = _get_ollama_client()
    if client:
        return ("ollama", client)

    # Fallback to Gemini
    client = _get_gemini_client()
    if client:
        return ("gemini", client)

    return ("none", None)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _get_cached_weather(region_id: str) -> str:
    """Return cached weather summary string, or '' if expired/absent."""
    entry = _weather_cache.get(region_id)
    if entry and (time.time() - entry["ts"]) < _WEATHER_CACHE_TTL:
        return entry["data"]
    return ""


def set_weather_cache(region_id: str, weather_summary: str) -> None:
    """Store a weather summary string for region_id with a TTL timestamp."""
    _weather_cache[region_id] = {"data": weather_summary, "ts": time.time()}


def _build_context(
    region_id: str,
    region_name: str,
    season: str,
    state_name: str,
    climate_zone: str,
    soil_info: str,
    weather_summary: str,
    crop_context: str,
) -> str:
    """Build the combined system persona + context block string."""
    ctx_lines = []
    if region_name:
        ctx_lines.append(f"District: {region_name}")
    elif region_id:
        ctx_lines.append(f"Region: {region_id}")
    if state_name:
        ctx_lines.append(f"State: {state_name}")
    if climate_zone:
        ctx_lines.append(f"Zone: {climate_zone}")
    if season:
        ctx_lines.append(f"Season: {season}")
    if soil_info:
        ctx_lines.append(f"Soil: {soil_info}")
    if weather_summary:
        ctx_lines.append(f"Weather: {weather_summary}")
    if crop_context:
        ctx_lines.append(f"Top Crops: {crop_context}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "General Indian farming context"
    return f"{_SYSTEM_PERSONA}\n\nContext:\n{context_block}"


def _cap_history(
    history: List[Dict[str, Any]],
    question: str,
    answer: str,
) -> List[Dict[str, Any]]:
    """Append the latest turn and cap to _MAX_HISTORY_TURNS."""
    updated = list(history)
    updated.append({"role": "user",  "parts": [question]})
    updated.append({"role": "model", "parts": [answer]})
    max_entries = _MAX_HISTORY_TURNS * 2
    if len(updated) > max_entries:
        updated = updated[-max_entries:]
    return updated


# ── Format converters ─────────────────────────────────────────────────────────

def _history_to_ollama_messages(
    history: List[Dict[str, Any]],
    system_text: str,
    question: str,
) -> List[Dict[str, str]]:
    """
    Convert project history list (Gemini format) → Ollama messages list.

    Gemini history format:  [{"role": "user"|"model", "parts": ["text"]}, ...]
    Ollama messages format: [{"role": "system"|"user"|"assistant", "content": "text"}, ...]
    """
    messages = [{"role": "system", "content": system_text}]
    for turn in history:
        role = turn.get("role", "user")
        if role == "model":
            role = "assistant"
        parts = turn.get("parts", [])
        text = parts[0] if parts else ""
        if isinstance(text, dict):
            text = text.get("text", "")
        if text:
            messages.append({"role": role, "content": str(text)})
    messages.append({"role": "user", "content": question})
    return messages


def _history_to_gemini_contents(
    history: List[Dict[str, Any]],
    system_text: str,
    question: str,
) -> List[Dict[str, Any]]:
    """Build the Gemini `contents` list from history + current question."""
    contents = []
    for turn in history:
        role = turn.get("role", "user")
        parts = turn.get("parts", [""])
        text = parts[0] if parts else ""
        if isinstance(text, dict):
            text = text.get("text", "")
        if text:
            contents.append({"role": role, "parts": [{"text": str(text)}]})

    # Inject system context only on the very first turn
    if not history:
        full_question = (
            f"{system_text}\n\n"
            f"Farmer's question: {question}\n\n"
            f"Answer concisely (max 200 words):"
        )
    else:
        full_question = question

    contents.append({"role": "user", "parts": [{"text": full_question}]})
    return contents


def _clean_answer(text: str) -> str:
    """Strip markdown code fences if the model wraps output in them."""
    return re.sub(r"```[a-z]*\s*", "", text).replace("```", "").strip()


# ── Public API ────────────────────────────────────────────────────────────────

def answer_farmer_question(
    question: str,
    region_id: str = "",
    region_name: str = "",
    season: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    crop_context: str = "",
    state_name: str = "",
    climate_zone: str = "",
    soil_info: str = "",
    weather_summary: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Answer a farmer's free-form question using LLaMA (Ollama) with Gemini fallback.
    Returns (answer_str, updated_history_list).
    """
    provider, client = _resolve_client()

    if provider == "none":
        fallback = (
            "AI chat is currently unavailable. "
            "Please start Ollama (ollama serve) or add a GEMINI_API_KEY in your .env file. "
            "All crop recommendation features continue to work without AI chat."
        )
        return fallback, history or []

    history = history or []

    if not weather_summary and region_id:
        weather_summary = _get_cached_weather(region_id)

    system_text = _build_context(
        region_id, region_name, season, state_name,
        climate_zone, soil_info, weather_summary, crop_context,
    )

    try:
        if provider == "ollama":
            messages = _history_to_ollama_messages(history, system_text, question)
            response = client.chat(model=OLLAMA_MODEL, messages=messages)
            answer = _clean_answer(response["message"]["content"])

        else:  # gemini
            contents = _history_to_gemini_contents(history, system_text, question)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_NO_THINK_CONFIG,
            )
            answer = _clean_answer(response.text)

        return answer, _cap_history(history, question, answer)

    except Exception as e:
        logger.warning(f"Chat response failed ({provider}): {e}")
        err_msg = "I'm having trouble connecting to the AI service right now. Please try again in a moment."
        return err_msg, history


def stream_farmer_answer(
    question: str,
    region_id: str = "",
    region_name: str = "",
    season: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    crop_context: str = "",
    state_name: str = "",
    climate_zone: str = "",
    soil_info: str = "",
    weather_summary: str = "",
) -> Generator[str, None, None]:
    """
    Stream the LLM answer token-by-token.

    Yields SSE-formatted strings: 'data: <chunk>\\n\\n'.
    On completion yields 'data: [DONE]<history_json>\\n\\n'.
    On error yields 'data: [ERROR] <message>\\n\\n'.
    """
    provider, client = _resolve_client()

    if provider == "none":
        yield "data: AI chat requires Ollama running locally or a GEMINI_API_KEY in .env.\\n\\n"
        yield "data: [DONE]\\n\\n"
        return

    history = history or []

    if not weather_summary and region_id:
        weather_summary = _get_cached_weather(region_id)

    system_text = _build_context(
        region_id, region_name, season, state_name,
        climate_zone, soil_info, weather_summary, crop_context,
    )

    try:
        full_text = ""

        if provider == "ollama":
            messages = _history_to_ollama_messages(history, system_text, question)
            for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
                token = chunk.get("message", {}).get("content", "") or ""
                if token:
                    full_text += token
                    safe = token.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

        else:  # gemini
            contents = _history_to_gemini_contents(history, system_text, question)
            for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=_NO_THINK_CONFIG,
            ):
                token = getattr(chunk, "text", "") or ""
                if token:
                    full_text += token
                    safe = token.replace("\n", "\\n")
                    yield f"data: {safe}\n\n"

        full_text = _clean_answer(full_text)

        import json
        updated_history = _cap_history(history, question, full_text)
        yield f"data: [DONE]{json.dumps(updated_history)}\n\n"

    except Exception as e:
        logger.warning(f"Stream chat failed ({provider}): {e}")
        yield "data: [ERROR] I'm having trouble connecting to the AI service right now. Please try again.\n\n"
        yield "data: [DONE]\n\n"
