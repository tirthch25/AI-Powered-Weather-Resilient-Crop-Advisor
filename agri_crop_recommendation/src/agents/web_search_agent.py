"""
Web Search Agent
================
Free web search for the crop advisor pipeline.

Two search backends (tried in order):
  1. DuckDuckGo Instant Answer API (free, no key, no rate-limit registration)
  2. Gemini Search Grounding (built-in Google Search via the Gemini API)

Ollama tool-calling pipeline:
  - Defines a 'search_web' tool schema for Ollama models that support
    tool-calling (llama3.1, llama3.2, mistral, qwen2.5 etc.)
  - When Ollama requests a tool call, we execute the search here and
    return results back into the conversation

Usage:
    from src.agents.web_search_agent import search_web, call_ollama_with_search

    # Simple search
    results = search_web("crop disease alerts Germany Bavaria 2025")

    # Ollama with tool-calling
    answer = call_ollama_with_search(
        prompt="What crops are best to plant in Bavaria in autumn 2025?",
        location="Bavaria, Germany"
    )
"""

import os
import re
import json
import logging
import time
from typing import Optional, List, Dict

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]

# Simple in-memory cache for search results (5 min TTL)
_SEARCH_CACHE: Dict[str, tuple] = {}
_SEARCH_CACHE_TTL = 300


# ── Backend 1: DuckDuckGo Instant Answer API ──────────────────────────────────

def _duckduckgo_search(query: str, max_results: int = 4) -> List[Dict]:
    """
    Search via DuckDuckGo Instant Answer API (free, no API key needed).
    Returns list of {title, snippet, url}.
    """
    try:
        import requests
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params=params,
            timeout=6,
            headers={"User-Agent": "CropAdvisor/3.0"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append({
                "title":   data.get("Heading", "DuckDuckGo Abstract"),
                "snippet": data["AbstractText"][:300],
                "url":     data.get("AbstractURL", ""),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title":   topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "snippet": topic["Text"][:250],
                    "url":     topic.get("FirstURL", ""),
                })

        return results[:max_results]
    except Exception as e:
        logger.debug("[WebSearch] DuckDuckGo failed: %s", e)
        return []


def _duckduckgo_html_search(query: str, max_results: int = 4) -> List[Dict]:
    """
    Fallback: scrape DuckDuckGo HTML search (no API needed).
    """
    try:
        import requests
        from html.parser import HTMLParser

        class _SnippetParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._in_result = False
                self._current = {}

            def handle_starttag(self, tag, attrs):
                attrs_d = dict(attrs)
                if tag == "a" and "result__a" in attrs_d.get("class", ""):
                    self._current = {"url": attrs_d.get("href", ""), "title": ""}
                    self._in_result = True

            def handle_data(self, data):
                if self._in_result and "title" in self._current and not self._current["title"]:
                    self._current["title"] = data.strip()

            def handle_endtag(self, tag):
                if tag == "a" and self._in_result:
                    if self._current.get("title"):
                        self.results.append(dict(self._current))
                    self._in_result = False
                    self._current = {}

        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 CropAdvisor/3.0"},
        )
        # Simple snippet extraction
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
        results = []
        for t, s in zip(titles[:max_results], snippets[:max_results]):
            results.append({
                "title":   re.sub(r"<[^>]+>", "", t).strip()[:120],
                "snippet": re.sub(r"<[^>]+>", "", s).strip()[:280],
                "url":     "",
            })
        return results
    except Exception as e:
        logger.debug("[WebSearch] DuckDuckGo HTML fallback failed: %s", e)
        return []


# ── Backend 2: Gemini Search Grounding ───────────────────────────────────────

def _gemini_grounded_search(query: str) -> Optional[str]:
    """
    Use Gemini's built-in Google Search grounding to answer a query.
    Returns a grounded text answer, or None if unavailable.
    Only models that support search grounding will work (gemini-2.0-flash+).
    """
    if not GEMINI_KEYS:
        return None

    _SEARCH_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-001",
    ]

    try:
        from google import genai as _g
        from google.genai import types as _gt

        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _SEARCH_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=query,
                        config=_gt.GenerateContentConfig(
                            tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                        ),
                    )
                    text = resp.text.strip() if resp.text else None
                    if text:
                        logger.info("[WebSearch] Gemini grounded search OK (%s)", model)
                        return text
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    if "not supported" in err.lower() or "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[WebSearch] Gemini search model %s error: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[WebSearch] Gemini grounded search failed: %s", e)

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 4) -> str:
    """
    Search the web and return a text summary of findings.
    Tries: DuckDuckGo → DuckDuckGo HTML → returns empty string on total failure.
    Results are cached 5 minutes.
    """
    cache_key = query.lower().strip()
    cached_ts, cached_val = _SEARCH_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _SEARCH_CACHE_TTL:
        return cached_val

    # Try DuckDuckGo instant answers
    results = _duckduckgo_search(query, max_results)
    if not results:
        results = _duckduckgo_html_search(query, max_results)

    if results:
        text = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in results if r.get("snippet")
        )
        _SEARCH_CACHE[cache_key] = (time.time(), text)
        return text

    _SEARCH_CACHE[cache_key] = (time.time(), "")
    return ""


def gemini_search_answer(prompt: str, context: str = "") -> Optional[str]:
    """
    Ask Gemini a question with Google Search grounding active.
    Returns grounded answer text, or None if search grounding not available.
    This is the primary web-search mechanism for crop advisories.
    """
    full_prompt = (context + "\n\n" + prompt).strip() if context else prompt
    return _gemini_grounded_search(full_prompt)


# ── Ollama Tool-Calling Pipeline ──────────────────────────────────────────────

_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for current agricultural information such as "
            "crop advisories, market prices, pest alerts, weather patterns, "
            "or farming practices for a specific region."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
}


def call_ollama_with_search(
    prompt: str,
    location: str = "",
    max_tool_calls: int = 3,
    timeout: int = 45,
) -> Optional[str]:
    """
    Call Ollama with web search tool-calling enabled.
    The model can call 'search_web' up to max_tool_calls times to gather
    real-time data before producing its final answer.

    Returns the final text answer, or None if Ollama is unavailable.
    """
    try:
        import ollama as _o
        client = _o.Client(host=OLLAMA_URL)

        system_msg = (
            "You are an expert agricultural advisor. "
            "Use the search_web tool to find current, real-time information "
            "about crop advisories, market prices, and farming conditions "
            f"for {location}. " if location else
            "You are an expert agricultural advisor. "
            "Use the search_web tool to find current information when needed. "
        )
        system_msg += (
            "After gathering information, provide a comprehensive recommendation "
            "in valid JSON format as requested."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]

        tool_calls_made = 0

        while tool_calls_made < max_tool_calls:
            try:
                resp = client.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    tools=[_SEARCH_TOOL_SCHEMA],
                    options={"temperature": 0.1, "num_ctx": 8192},
                )
            except Exception as e:
                logger.debug("[WebSearch] Ollama chat failed: %s", e)
                return None

            msg = resp.get("message", {})
            tool_calls = msg.get("tool_calls", [])

            if not tool_calls:
                # No more tool calls — return the final answer
                return msg.get("content", "").strip() or None

            # Execute each tool call the model requested
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

            for tc in tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "search_web":
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    query = args.get("query", "")
                    if query:
                        logger.info("[WebSearch] Ollama tool call: search_web(%r)", query)
                        search_result = search_web(query)
                        messages.append({
                            "role": "tool",
                            "content": search_result or "No results found.",
                        })
                        tool_calls_made += 1

        # Final call after all tool calls exhausted
        try:
            final_resp = client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={"temperature": 0.1, "num_ctx": 8192},
            )
            return final_resp.get("message", {}).get("content", "").strip() or None
        except Exception:
            return None

    except ImportError:
        logger.debug("[WebSearch] Ollama not installed")
        return None
    except Exception as e:
        logger.debug("[WebSearch] Ollama tool-calling failed: %s", e)
        return None
