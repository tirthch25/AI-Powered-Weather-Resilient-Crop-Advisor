"""
Crop Recommendation Agent — v3 (Dynamic, Search-Grounded)
==========================================================
Strategy (priority order):
  1. In-memory cache (instant) — same location+season+zone returns cached result
  2. Gemini with Google Search Grounding (real-time advisories, live prices)
  3. Gemini without search (4-key rotation, model fallback list)
  4. Ollama with web search tool-calling (uses DuckDuckGo via tool API)
  5. Ollama plain (no search)
  6. Geography-aware zone fallback (no API) — final fallback

Key fixes vs v2:
  - Gemini called directly (NOT inside ThreadPoolExecutor) — fixes
    "Cannot send a request, as the client has been closed" errors
  - All 4 API keys rotated on 429 quota errors
  - Model fallback list: gemini-2.5-flash-lite → gemini-2.0-flash-lite → ...
  - Prompt is geography-aware: country, hemisphere, agro-climate, local names
  - Fallback crops keyed by climate zone + hemisphere (not India-only Hindi names)
  - Web search grounding for real-time crop advisories and market prices
"""

import os
import json
import logging
import re
import time
from typing import Optional, Dict, List

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# All 4 Gemini keys — rotated on 429
GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]
GEMINI_KEY = GEMINI_KEYS[0] if GEMINI_KEYS else ""  # backward compat

# Model fallback list — same as llm_location_agent
_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Models that support Google Search Grounding
_SEARCH_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
]

# ── In-memory result cache ────────────────────────────────────────────────────
_CROP_CACHE: Dict[tuple, tuple] = {}
_CROP_CACHE_TTL = 3600  # 1 hour


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[list]:
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
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return None


# Country-specific crop hint lists — shown to LLM as few-shot context
# so it understands what crops are actually grown in each region.
# Keys MUST be lowercase — matching is done via substring search.
_COUNTRY_CROP_HINTS: Dict[str, str] = {
    # ── WESTERN EUROPE ────────────────────────────────────────────────────────
    "germany":        "Winterweizen (Winter Wheat), Winterraps (Canola/Rapeseed), Zuckerrübe (Sugar Beet), Mais (Maize), Kartoffel (Potato), Wintergerste (Winter Barley), Roggen (Rye), Triticale, Hafer (Oat), Ackerbohne (Field Bean), Sonnenblume (Sunflower)",
    "france":         "Blé tendre (Soft Wheat), Colza (Canola), Maïs (Maize), Orge (Barley), Betterave sucrière (Sugar Beet), Tournesol (Sunflower), Pomme de terre (Potato), Pois protéagineux (Protein Pea), Soja, Vin (Grapevine), Lin oléagineux (Linseed)",
    "united kingdom": "Winter Wheat, Oilseed Rape (Canola), Spring Barley, Winter Barley, Potatoes, Sugar Beet, Oats, Field Beans, Linseed, Peas",
    "netherlands":    "Aardappel (Potato), Suikerbiet (Sugar Beet), Tarwe (Wheat), Gerst (Barley), Ui (Onion), Koolzaad (Canola), Tulpenbollen (Tulip bulbs), Prei (Leek)",
    "belgium":        "Tarwe (Wheat), Suikerbiet (Sugar Beet), Aardappel (Potato), Gerst (Barley), Koolzaad (Canola), Prei (Leek), Cichorei (Chicory)",
    "switzerland":    "Winterweizen (Winter Wheat), Zuckerrübe (Sugar Beet), Kartoffel (Potato), Gerste (Barley), Raps (Canola), Körnermais (Grain Maize), Apfel (Apple), Wein (Grapevine)",
    "austria":        "Winterweizen (Winter Wheat), Mais (Maize), Zuckerrübe (Sugar Beet), Wintergerste (Winter Barley), Kartoffel (Potato), Winterraps (Winter Canola), Hafer (Oat), Rüben (Turnips)",
    "portugal":       "Trigo (Wheat), Milho (Maize), Arroz (Rice), Cevada (Barley), Girassol (Sunflower), Tomate (Tomato), Azeite (Olive oil), Vinho (Grape/Wine), Cortiça (Cork Oak)",
    "spain":          "Trigo (Wheat), Cebada (Barley), Oliva (Olive), Vid (Grape/Vine), Girasol (Sunflower), Maíz (Maize), Naranja (Orange), Almendro (Almond), Arroz (Rice), Remolacha azucarera (Sugar Beet)",
    "italy":          "Grano tenero (Wheat), Mais (Maize), Pomodoro (Tomato), Olivo (Olive), Uva (Grape), Riso (Rice), Girasole (Sunflower), Soia (Soybean), Barbabietola da zucchero (Sugar Beet)",
    "greece":         "Σιτάρι (Wheat), Βαμβάκι (Cotton), Ελιά (Olive), Αμπέλι (Grape), Βερίκοκο (Apricot), Ρύζι (Rice), Κριθάρι (Barley), Καλαμπόκι (Maize), Τομάτα (Tomato)",
    # ── NORTHERN EUROPE ───────────────────────────────────────────────────────
    "sweden":         "Höstvete (Winter Wheat), Korn (Barley), Havre (Oat), Höstraps (Winter Canola), Potatis (Potato), Sockerbetor (Sugar Beet), Råg (Rye), Ärta (Pea)",
    "denmark":        "Vinterbyg (Winter Barley), Vinterhvede (Winter Wheat), Vinterraps (Winter Canola), Havre (Oat), Kartofler (Potato), Sukkerroer (Sugar Beet), Rug (Rye)",
    "finland":        "Ohra (Barley), Kaura (Oat), Vehnä (Wheat), Rypsi (Turnip Rape), Peruna (Potato), Sokerijuurikas (Sugar Beet), Herne (Pea)",
    "norway":         "Bygg (Barley), Havre (Oat), Hvete (Wheat), Raps (Canola), Potet (Potato), Gulrot (Carrot), Kål (Cabbage)",
    # ── CENTRAL/EASTERN EUROPE ────────────────────────────────────────────────
    "poland":         "Pszenica (Wheat), Żyto (Rye), Rzepak (Canola), Ziemniaki (Potato), Buraki cukrowe (Sugar Beet), Kukurydza (Maize), Owies (Oat), Groch (Pea), Słonecznik (Sunflower)",
    "czech":          "Pšenice (Wheat), Ječmen (Barley), Řepka (Canola), Kukuřice (Maize), Cukrová řepa (Sugar Beet), Slunečnice (Sunflower), Brambory (Potato), Oves (Oat)",
    "slovakia":       "Pšenica (Wheat), Jačmeň (Barley), Repka (Canola), Kukurica (Maize), Cukrová repa (Sugar Beet), Slnečnica (Sunflower), Zemiaky (Potato)",
    "hungary":        "Búza (Wheat), Kukorica (Maize), Napraforgó (Sunflower), Repce (Canola), Cukorrépa (Sugar Beet), Burgonya (Potato), Árpa (Barley), Szójabab (Soybean)",
    "romania":        "Grâu (Wheat), Porumb (Maize), Floarea-soarelui (Sunflower), Rapița (Canola), Soia (Soybean), Cartofi (Potato), Sfeclă de zahăr (Sugar Beet), Orz (Barley)",
    "bulgaria":       "Пшеница (Wheat), Слънчоглед (Sunflower), Царевица (Maize), Рапица (Canola), Ечемик (Barley), Захарно цвекло (Sugar Beet), Картофи (Potato)",
    "serbia":         "Pšenica (Wheat), Kukuruz (Maize), Suncokret (Sunflower), Soja (Soybean), Šećerna repa (Sugar Beet), Ječam (Barley), Krompir (Potato)",
    "ukraine":        "Пшениця (Wheat), Соняшник (Sunflower), Кукурудза (Maize), Соя (Soybean), Ріпак (Canola), Ячмінь (Barley), Буряк цукровий (Sugar Beet), Жито (Rye)",
    # ── RUSSIA / CENTRAL ASIA ─────────────────────────────────────────────────
    "russia":         "Пшеница (Wheat), Ячмень (Barley), Кукуруза (Maize), Подсолнечник (Sunflower), Рапс (Canola), Сахарная свёкла (Sugar Beet), Соя (Soybean), Рожь (Rye), Овёс (Oat)",
    "kazakhstan":     "Пшеница (Wheat), Ячмень (Barley), Хлопок (Cotton), Подсолнечник (Sunflower), Рапс (Canola), Рис (Rice), Кукуруза (Maize), Сахарная свёкла (Sugar Beet)",
    # ── NORTH AMERICA ────────────────────────────────────────────────────────
    "united states":  "Corn, Soybean, Winter Wheat, Cotton, Rice, Canola, Sorghum, Sunflower, Alfalfa, Sugar Beet, Peanuts, Tobacco",
    "canada":         "Spring Wheat, Canola, Barley, Corn, Soybean, Oats, Flaxseed, Durum Wheat, Peas, Lentils, Mustard",
    "mexico":         "Maíz (Maize), Sorgo (Sorghum), Trigo (Wheat), Caña de azúcar (Sugarcane), Frijol (Bean), Chile (Chili Pepper), Jitomate (Tomato), Aguacate (Avocado), Alfalfa",
    "cuba":           "Caña de azúcar (Sugarcane), Arroz (Rice), Tabaco (Tobacco), Maíz (Maize), Boniato (Sweet Potato), Yuca (Cassava), Plátano (Plantain), Café (Coffee)",
    # ── CENTRAL / SOUTH AMERICA ───────────────────────────────────────────────
    "brazil":         "Soja (Soybean), Milho (Maize), Cana-de-açúcar (Sugarcane), Café (Coffee), Arroz (Rice), Algodão (Cotton), Feijão (Bean), Mandioca (Cassava), Laranja (Orange), Eucalipto",
    "argentina":      "Soja (Soybean), Maíz (Maize), Trigo (Wheat), Girasol (Sunflower), Cebada (Barley), Algodón (Cotton), Caña de azúcar (Sugarcane), Pera (Pear), Manzana (Apple)",
    "colombia":       "Café (Coffee), Caña de azúcar (Sugarcane), Maíz (Maize), Plátano (Plantain), Yuca (Cassava), Arroz (Rice), Papa (Potato), Flores (Cut Flowers), Cacao",
    "peru":           "Papa (Potato), Maíz (Maize), Caña de azúcar (Sugarcane), Arroz (Rice), Café (Coffee), Espárrago (Asparagus), Uva (Grape), Quinua (Quinoa), Cacao",
    "chile":          "Trigo (Wheat), Maíz (Maize), Uva (Grape/Vine), Manzana (Apple), Arándano (Blueberry), Cereza (Cherry), Remolacha (Sugar Beet), Raps (Canola), Papa (Potato)",
    "ecuador":        "Cacao, Banano (Banana), Caña de azúcar (Sugarcane), Maíz (Maize), Arroz (Rice), Papa (Potato), Café (Coffee), Flores (Cut Flowers), Palma africana (Oil Palm)",
    "bolivia":        "Soja (Soybean), Maíz (Maize), Caña de azúcar (Sugarcane), Quinua (Quinoa), Papa (Potato), Arroz (Rice), Girasol (Sunflower), Cacao",
    # ── MIDDLE EAST / NORTH AFRICA ────────────────────────────────────────────
    "turkey":         "Buğday (Wheat), Arpa (Barley), Mısır (Maize), Pamuk (Cotton), Şeker pancarı (Sugar Beet), Domates (Tomato), Üzüm (Grape), Fındık (Hazelnut), Zeytinyağı (Olive), Tütün (Tobacco)",
    "egypt":          "Qamh (Wheat), Arz (Rice), Qotn (Cotton), Beet El Sukkar (Sugar Beet), Dhora (Maize), Burtuqal (Orange), Batatiss (Potato), Foul (Fava Bean), Qasab el Sukkar (Sugarcane)",
    "iran":           "Gandum (Wheat), Jou (Barley), Choqondar (Sugar Beet), Berenj (Rice), Panbeh (Cotton), Pistachio, Saffron, Grape, Tomato, Potato",
    "iraq":           "Qamh (Wheat), Sha'ir (Barley), Tamr (Dates), Ruzz (Rice), Shummar (Maize), Tuffah (Apple), Brtuqal (Orange), Cotton, Sunflower",
    "saudi arabia":   "Tamr (Dates), Qamh (Wheat), Sha'ir (Barley), Alfalfa, Tomato, Cucumber, Sorghum, Millet, Potato, Onion",
    "syria":          "Qamh (Wheat), Sha'ir (Barley), Zaytoun (Olive), Qotn (Cotton), Tamr (Dates), Beet el Sukkar (Sugar Beet), Tuffah (Apple), Zeytoon (Olive), Grape",
    "morocco":        "Blé (Wheat), Orge (Barley), Maïs (Maize), Betterave (Sugar Beet), Agrumes (Citrus), Olive, Tomate (Tomato), Pomme de terre (Potato), Tournesol (Sunflower)",
    "algeria":        "Blé dur (Durum Wheat), Orge (Barley), Pomme de terre (Potato), Tomate (Tomato), Datte (Dates), Olive, Vigne (Grape), Agrumes (Citrus), Figue (Fig)",
    "afghanistan":    "Gandum (Winter Wheat), Jo (Barley), Zardalu (Apricot), Kishmish (Grapes/Raisins), Kachalu (Potato), Anaar (Pomegranate), Zira (Cumin), Maize, Peas, Saffron",
    # ── SOUTH ASIA ────────────────────────────────────────────────────────────
    "india":          "Chawal/Dhan (Rice), Gehun (Wheat), Makka (Maize), Chana (Chickpea), Sarson (Mustard), Kapas (Cotton), Mungfali (Groundnut), Tur/Arhar (Pigeon Pea), Soybean, Ganna (Sugarcane)",
    "pakistan":       "Gandum (Wheat), Chawal (Rice), Kapas (Cotton), Ganna (Sugarcane), Makkai (Maize), Sarson (Mustard), Mash (Lentil), Masoor (Red Lentil), Moong (Mung Bean)",
    "bangladesh":     "Dhan (Rice), Gom (Wheat), Pat (Jute), Alu (Potato), Aam (Mango), Piyaj (Onion), Sarisha (Mustard), Ak (Sugarcane), Morich (Chili)",
    "nepal":          "Dhan (Rice), Makai (Maize), Gehun (Wheat), Aalu (Potato), Ukhoo (Sugarcane), Sarson (Mustard), Chamomile, Ginger, Cardamom",
    "sri lanka":      "Hal (Rice), Thé (Tea), Rabba Kakao (Rubber), Naaraththaa (Coconut), Dimbulaa thé (Tea), Innala (Yam), Maize, Sugarcane, Cinnamon",
    # ── EAST / SOUTHEAST ASIA ────────────────────────────────────────────────
    "china":          "Xiǎomài (Wheat), Shuǐdào (Rice), Yùmǐ (Maize), Dàdòu (Soybean), Miánhuā (Cotton), Gānzhe (Sugarcane), Huāshēng (Peanut), Yóucài (Canola), Shāngmǐ (Potato)",
    "japan":          "Kome (Rice), Mugi (Wheat/Barley), Daizu (Soybean), Satsumaimo (Sweet Potato), Jagaimo (Potato), Natane (Canola), Körnerkais (Sugar Beet), Kōcha (Tea)",
    "south korea":    "Sssal (Rice), Baechu (Cabbage/Kimchi), Gochutgaru (Chili Pepper), Mu (Radish), Insamn (Ginseng), Bori (Barley), Injong (Potato), Maize, Garlic",
    "vietnam":        "Lúa (Rice), Cà phê (Coffee), Cao su (Rubber), Mía (Sugarcane), Cacao, Ngô (Maize), Rau (Vegetables), Điều (Cashew), Hồ tiêu (Black Pepper)",
    "thailand":       "Khao (Rice), Mân sắn (Cassava), Ói (Sugarcane), Khao phod (Maize), Yiang phara (Rubber), Mah phrao (Coconut), Palao (Oil Palm), Mango, Durian",
    "myanmar":        "Sein Yway (Rice), Pyauk Seik (Sesame), Kyauk Nyunt (Lentil), Myauk Chin (Sunflower), Pè (Bean), Kauk Nyunt (Maize), Ngapyawè (Cotton), Rubber",
    "cambodia":       "Sraov (Rice), Kachaang (Cassava), Kompot Skor (Sugar Palm), Angkoh (Maize), Kh'tum (Pepper), Banana, Mango, Rubber, Cashew",
    "laos":           "Khao (Rice), Khaophod (Maize), Cassava, Coffee, Sugarcane, Tobacco, Rubber, Banana, Sesame",
    "malaysia":       "Padi (Rice), Kelapa Sawit (Oil Palm), Getah (Rubber), Nanas (Pineapple), Pisang (Banana), Durian, Cacao, Coconut, Pepper",
    "indonesia":      "Padi (Rice), Kelapa sawit (Oil Palm), Karet (Rubber), Jagung (Maize), Singkong (Cassava), Tebu (Sugarcane), Kopi (Coffee), Cengkeh (Clove)",
    "philippines":    "Palay (Rice), Mais (Maize), Niyog (Coconut), Tubo (Sugarcane), Nana (Pineapple), Saging (Banana), Kape (Coffee), Gulay (Vegetables), Kamote (Sweet Potato)",
    # ── AFRICA ───────────────────────────────────────────────────────────────
    "nigeria":        "Rice, Maize (Corn), Sorghum, Cassava, Yam, Cowpea, Groundnut (Peanut), Millet, Soybean, Oil Palm, Cotton, Plantain",
    "ethiopia":       "Teff, Wheat, Maize, Sorghum, Barley, Coffee, Chickpea, Lentil, Enset (Ensete), Sesame, Faba Bean",
    "kenya":          "Maize, Tea, Coffee, Wheat, Rice, Sugarcane, Sorghum, Pyrethrum, French Beans, Cut Flowers, Avocado, Macadamia",
    "ghana":          "Cocoa, Cassava, Maize, Oil Palm, Groundnut (Peanut), Rice, Plantain, Tomato, Cotton, Tobacco, Coconut",
    "tanzania":       "Maize, Rice, Cassava, Sorghum, Coffee, Tea, Cotton, Sisal, Cashew, Clove, Pyrethrum, Tobacco",
    "uganda":         "Matoke (Banana/Plantain), Maize, Cassava, Sweet Potato, Coffee, Tea, Cotton, Sugarcane, Millet, Sorghum",
    "zimbabwe":       "Maize, Tobacco, Cotton, Sorghum, Wheat, Groundnut, Sunflower, Soybean, Sugar Beet, Vegetables",
    "zambia":         "Maize, Cassava, Sorghum, Cotton, Tobacco, Groundnut, Sunflower, Soybean, Wheat, Rice, Vegetables",
    "south africa":   "Maize, Wheat, Sunflower, Sugarcane, Soybean, Sorghum, Groundnut, Cotton, Barley, Canola, Tobacco, Citrus, Grapes",
    # ── OCEANIA ───────────────────────────────────────────────────────────────
    "australia":      "Wheat, Barley, Canola, Sorghum, Cotton, Rice, Sugarcane, Oats, Lentil, Chickpea, Lupin, Wool/Sheep",
    "new zealand":    "Wheat, Barley, Ryegrass, Kiwifruit, Apple, Grapes, Sweetcorn, Onion, Potato, Dairy pasture, Squash",
}

def _get_country_hints(country: str) -> str:
    """Return crop hint string for the given country."""
    c_lower = country.lower()
    for key, hints in _COUNTRY_CROP_HINTS.items():
        if key in c_lower:
            return hints
    return ""

# Known Indian/Hindi crop names — if these appear for a non-South-Asian country,
# Gemini has hallucinated the wrong geography
_HINDI_CROP_NAMES = {
    "chawal", "gehun", "makka", "chana", "sarson", "kapas", "mungfali",
    "bajra", "jowar", "tur", "arhar", "moong", "urad", "dhan", "ganna",
    "lahsun", "tamatar", "aloo", "pyaaz", "matar", "bhindi", "palak",
    "shimla mirch", "karela", "lauki", "turai",
}
_SOUTH_ASIAN_COUNTRIES = {"india", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan"}


def _validate_crops(crops: Optional[list], country: str) -> Optional[list]:
    """
    Validate that Gemini returned geographically correct crops.
    If Gemini returned Hindi crop names for a non-South-Asian country,
    returns None so the fallback table is used instead.
    """
    if not crops or not isinstance(crops, list):
        return None

    c_lower = country.lower()
    is_south_asian = any(sa in c_lower for sa in _SOUTH_ASIAN_COUNTRIES)

    if not is_south_asian:
        # Count how many crops have Hindi local names
        hindi_count = 0
        for crop in crops:
            local = (crop.get("local_name") or "").lower()
            name  = (crop.get("crop_name") or "").lower()
            if any(h in local for h in _HINDI_CROP_NAMES) or any(h in name for h in _HINDI_CROP_NAMES):
                hindi_count += 1
        if hindi_count >= 2:
            logger.warning(
                "[CropAgent] Gemini returned %d Hindi crop names for %s — rejecting, using fallback",
                hindi_count, country,
            )
            return None

    # Basic sanity: need at least 3 crop entries with real names
    valid = [c for c in crops if isinstance(c, dict) and c.get("crop_name") and c.get("crop_name") != "Unknown Crop"]
    if len(valid) < 3:
        return None

    return crops


def _build_prompt(
    country, state, district, season, climate, planning_days,
    irrigation, temp, humidity, current, forecast, soil, market, summary,
    include_search_context: bool = False,
) -> str:
    """Build a rich, geography-aware crop recommendation prompt with country-specific examples."""
    forecast_str = ""
    for f in forecast[:3]:
        forecast_str += (
            f"  {f.get('month', '')}: {f.get('temp_avg', '?')}°C avg, "
            f"{f.get('rainfall_mm', '?')}mm rain\n"
        )

    market_str = (
        ", ".join(f"{k}: {v}" for k, v in list(market.items())[:5])
        if market else "Use current local prices"
    )

    search_instruction = ""
    if include_search_context:
        search_instruction = (
            f"Search for current crop advisories, pest alerts, and market prices "
            f"in {district}, {state}, {country} before answering. "
        )

    # Embed country-specific crop hints directly in the prompt
    crop_hints = _get_country_hints(country)
    hints_str = (
        f"\nIMPORTANT — crops actually grown in {country}: {crop_hints}. "
        f"ONLY recommend crops from this list or close relatives. "
        f"NEVER recommend crops native to other continents.\n"
        if crop_hints else
        f"\nOnly recommend crops genuinely grown in {country}.\n"
    )

    hemisphere = (
        "Southern"
        if any(c in country.lower() for c in [
            "australia", "brazil", "argentina", "south africa",
            "new zealand", "chile", "peru", "uruguay", "zambia", "zimbabwe"
        ]) else "Northern"
    )

    return (
        f"SYSTEM: You are a senior agricultural expert ONLY for {district}, {state}, {country}. "
        f"Your recommendations MUST reflect actual farming practices in {country}. "
        f"{hints_str}"
        f"{search_instruction}"
        f"Season: {season}, Climate zone: {climate}, Hemisphere: {hemisphere}. "
        f"Irrigation: {irrigation}, Planning horizon: {planning_days} days. "
        f"Current conditions: {temp}°C, humidity {humidity}%, "
        f"rainfall last 7 days: {current.get('rainfall_7d_mm', '?')} mm. "
        f"Soil: {soil.get('type', 'Loam')} pH {soil.get('ph', 7.0)}, "
        f"organic matter: {soil.get('organic_matter', 'Medium')}, "
        f"drainage: {soil.get('drainage', 'Medium')}. "
        f"3-month forecast:\n{forecast_str.strip()}\n"
        f"Local market prices: {market_str}. "
        f"District context: {summary[:150] if summary else 'Standard agricultural region'}.\n\n"
        f"Return EXACTLY 6 crops for {country} ({season} season). "
        f"Sort by suitability_score descending. "
        f"Return ONLY a valid JSON array — no markdown, no explanation:\n"
        f'[{{"crop_name":"<official {country} crop name>","local_name":"<name in {country}\'s local language, e.g., German/French/etc.>","suitability_score":<0-100>,'
        f'"season_fit":"<Excellent/Good/Fair>","risk_level":"<Low/Medium/High>",'
        f'"duration_days":<int>,"water_need":"<Low/Medium/High>","estimated_yield":"<X-Y tons/ha>",'
        f'"planting_window":"<e.g. Oct 1 - Nov 15, in {country} context>","market_demand":"<High/Medium/Low>",'
        f'"reasons":["<reason specific to {state}, {country}>","<reason 2>"],'
        f'"warnings":["<real risk for {district}, {state}>"],'
        f'"growing_tip":"<tip from {country} agriculture extension practices>"}},...]\n'
    )


# ── Gemini callers ────────────────────────────────────────────────────────────

def _call_gemini_with_search(prompt: str) -> Optional[list]:
    """
    Call Gemini with Google Search Grounding for real-time crop advisories.
    Returns parsed crop list or None if search grounding not available.
    """
    if not GEMINI_KEYS:
        return None
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
                    if result:
                        logger.info(
                            "[CropAgent] Gemini search-grounded OK (%s, key ...%s) → %d crops",
                            model, api_key[-6:], len(result)
                        )
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    if "not supported" in err.lower() or "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[CropAgent] Search model %s: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[CropAgent] Gemini search grounding unavailable: %s", e)
    return None


def _call_gemini(prompt: str) -> Optional[list]:
    """
    Call Gemini directly (no ThreadPoolExecutor).
    Rotates through all 4 keys and all models on 429 quota errors.
    """
    if not GEMINI_KEYS:
        return None

    # Try new google.genai SDK
    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_MODELS:
                try:
                    resp = client.models.generate_content(model=model, contents=prompt)
                    text = resp.text.strip() if resp.text else None
                    result = _extract_json(text) if text else None
                    if result:
                        logger.info(
                            "[CropAgent] Gemini OK (%s, key ...%s) → %d crops",
                            model, api_key[-6:], len(result)
                        )
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        logger.debug("[CropAgent] %s quota on key ...%s", model, api_key[-6:])
                        continue
                    if "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[CropAgent] %s error: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[CropAgent] Gemini failed: %s", e)

    # Fallback: legacy SDK
    try:
        import google.generativeai as genai  # type: ignore
        for api_key in GEMINI_KEYS:
            genai.configure(api_key=api_key)
            for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                try:
                    resp = genai.GenerativeModel(model).generate_content(prompt)
                    result = _extract_json(resp.text.strip()) if resp.text else None
                    if result:
                        return result
                except Exception:
                    continue
    except ImportError:
        pass

    return None


# ── Ollama callers ────────────────────────────────────────────────────────────

def _call_ollama_with_search(prompt: str, location: str) -> Optional[list]:
    """Call Ollama with web search tool-calling. Returns crop list or None."""
    try:
        from src.agents.web_search_agent import call_ollama_with_search
        text = call_ollama_with_search(prompt, location=location, timeout=45)
        return _extract_json(text) if text else None
    except Exception as e:
        logger.debug("[CropAgent] Ollama web search failed: %s", e)
        return None


def _call_ollama(prompt: str) -> Optional[list]:
    """Plain Ollama call (no search tools)."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Return valid JSON array only. No markdown or explanation."},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1, "num_ctx": 6144},
        )
        return _extract_json(response["message"]["content"].strip())
    except Exception as e:
        logger.debug("[CropAgent] Ollama failed: %s", e)
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def recommend_crops_agent(
    country: str,
    state: str,
    district: str,
    gathered_data: dict,
    irrigation: str = "Limited",
    planning_days: int = 90,
    soil_override: Optional[dict] = None,
) -> list:
    """
    Generate AI crop recommendations for any global location.

    Pipeline:
      cache → Gemini+Search → Gemini plain → Ollama+Search → Ollama → zone fallback
    """
    current  = gathered_data.get("current", {})
    forecast = gathered_data.get("forecast_6month", [])
    soil     = soil_override or gathered_data.get("soil", {})
    season   = gathered_data.get("season", "")
    climate  = gathered_data.get("climate_zone", "")
    market   = gathered_data.get("market_prices", {})
    summary  = gathered_data.get("district_summary", "")

    temp     = current.get("temperature_c") or 25
    humidity = current.get("humidity_pct")  or 65

    # ── 1. Cache ──────────────────────────────────────────────────────────────
    cache_key = (country, state, district, season, climate, irrigation)
    cached_ts, cached_val = _CROP_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CROP_CACHE_TTL:
        logger.info("[CropAgent] Cache hit for %s", district)
        return cached_val

    location_str = f"{district}, {state}, {country}"

    # ── 2. Build prompt (with search context hint) ────────────────────────────
    prompt_with_search = _build_prompt(
        country, state, district, season, climate, planning_days,
        irrigation, temp, humidity, current, forecast, soil, market, summary,
        include_search_context=True,
    )
    prompt_plain = _build_prompt(
        country, state, district, season, climate, planning_days,
        irrigation, temp, humidity, current, forecast, soil, market, summary,
        include_search_context=False,
    )

    crops = None

    # ── 3. Gemini + Google Search Grounding ───────────────────────────────────
    if GEMINI_KEYS:
        raw = _call_gemini_with_search(prompt_with_search)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using search-grounded Gemini result")

    # ── 4. Gemini plain (key rotation, model fallback) ────────────────────────
    if not crops and GEMINI_KEYS:
        raw = _call_gemini(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Gemini result")

    # ── 5. Ollama with web search tool-calling ────────────────────────────────
    if not crops:
        raw = _call_ollama_with_search(prompt_with_search, location=location_str)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using Ollama+search result")

    # ── 6. Ollama plain ───────────────────────────────────────────────────────
    if not crops:
        raw = _call_ollama(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Ollama result")

    # ── 7. Geography-aware zone fallback ──────────────────────────────────────
    if not crops or not isinstance(crops, list):
        logger.warning("[CropAgent] All LLMs failed — using geography fallback for %s", country)
        crops = _fallback_crops(season, climate, country)


    # ── 8. Sanitize & sort ────────────────────────────────────────────────────
    sanitized = []
    for crop in crops[:8]:
        if not isinstance(crop, dict):
            continue
        crop.setdefault("crop_name",         "Unknown Crop")
        crop.setdefault("local_name",        crop["crop_name"])
        crop.setdefault("suitability_score", 60)
        crop.setdefault("season_fit",        "Good")
        crop.setdefault("risk_level",        "Medium")
        crop.setdefault("duration_days",     90)
        crop.setdefault("water_need",        "Medium")
        crop.setdefault("estimated_yield",   "2-4 tons/hectare")
        crop.setdefault("planting_window",   "Current season")
        crop.setdefault("market_demand",     "Medium")
        crop.setdefault("reasons",           ["Suitable for local climate and season"])
        crop.setdefault("warnings",          [])
        crop.setdefault("growing_tip",       "Follow local agricultural extension guidelines.")
        sanitized.append(crop)

    sanitized.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
    _CROP_CACHE[cache_key] = (time.time(), sanitized)
    return sanitized


# ── Geography-aware fallback crop tables ─────────────────────────────────────
# Keyed by (climate_zone, hemisphere) — never India-only names globally

def _fallback_crops(season: str, climate: str, country: str = "") -> list:
    """
    Return geographically appropriate fallback crops instantly without any API call.
    Climate-zone and hemisphere aware — correct crops for Europe, Americas, Africa,
    Asia, Oceania etc.
    """
    c_lower = country.lower()

    # Southern hemisphere seasons are flipped
    _SH_COUNTRIES = {
        "australia", "new zealand", "south africa", "brazil", "argentina",
        "chile", "peru", "bolivia", "uruguay", "paraguay", "zambia",
        "zimbabwe", "mozambique", "namibia", "botswana", "tanzania",
        "kenya", "ethiopia", "madagascar",
    }
    is_southern = any(sc in c_lower for sc in _SH_COUNTRIES)

    # India / South Asia — special seasons
    _SOUTH_ASIA = {"india", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan"}
    is_south_asia = any(sa in c_lower for sa in _SOUTH_ASIA)

    clim = climate.lower() if climate else ""

    # ── Tropical ─────────────────────────────────────────────────────────────
    if "tropical" in clim:
        return [
            {"crop_name": "Rice",       "local_name": "Rice",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "3-5 tons/ha",   "planting_window": "Main wet season",   "market_demand": "High",   "reasons": ["Staple crop for tropical climates", "High market demand"], "warnings": [], "growing_tip": "Ensure consistent flooding for paddy rice varieties."},
            {"crop_name": "Maize",      "local_name": "Corn/Maize", "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Start of rains",    "market_demand": "High",   "reasons": ["Versatile staple", "Suited for tropical soils"], "warnings": [], "growing_tip": "Apply nitrogen fertilizer at knee-high stage."},
            {"crop_name": "Cassava",    "local_name": "Cassava",    "suitability_score": 83, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Low",    "estimated_yield": "15-25 tons/ha", "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Drought tolerant", "Major tropical staple"], "warnings": [], "growing_tip": "Plant stakes 2-3 months old for best yield."},
            {"crop_name": "Sweet Potato","local_name": "Sweet Potato","suitability_score": 80,"season_fit": "Good",     "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "10-20 tons/ha", "planting_window": "Any season",        "market_demand": "High",   "reasons": ["Fast growing", "High nutrition value"], "warnings": [], "growing_tip": "Use well-drained soils to prevent rot."},
            {"crop_name": "Plantain",   "local_name": "Plantain",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 300, "water_need": "High",   "estimated_yield": "20-30 tons/ha", "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Perennial income", "Strong local demand"], "warnings": [], "growing_tip": "Ensure good drainage, plant on ridges."},
            {"crop_name": "Groundnut",  "local_name": "Peanut",     "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Early rains",      "market_demand": "High",   "reasons": ["Oil crop", "Nitrogen fixer", "Drought tolerant"], "warnings": [], "growing_tip": "Sandy loam soils give best pod fill."},
        ]

    # ── Arid / Semi-Arid ─────────────────────────────────────────────────────
    if "arid" in clim or "semi-arid" in clim:
        return [
            {"crop_name": "Sorghum",    "local_name": "Sorghum",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Rainy season",      "market_demand": "High",   "reasons": ["Most drought-tolerant grain", "Excellent for arid zones"], "warnings": [], "growing_tip": "Plant at the start of rains for best establishment."},
            {"crop_name": "Millet",     "local_name": "Millet",     "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 75,  "water_need": "Low",    "estimated_yield": "1-2.5 tons/ha", "planting_window": "Early rains",       "market_demand": "High",   "reasons": ["Extremely drought tolerant", "Short season"], "warnings": [], "growing_tip": "Thin to 15cm spacing after emergence."},
            {"crop_name": "Sesame",     "local_name": "Sesame",     "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Low",    "estimated_yield": "0.5-1 ton/ha",  "planting_window": "Dry season start",  "market_demand": "High",   "reasons": ["High oil content", "Heat tolerant", "Export value"], "warnings": [], "growing_tip": "Avoid waterlogging — very sensitive to excess water."},
            {"crop_name": "Cowpea",     "local_name": "Cowpea",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 65,  "water_need": "Low",    "estimated_yield": "0.8-1.5 tons/ha","planting_window": "Any rains",        "market_demand": "High",   "reasons": ["Nitrogen fixer", "Protein crop", "Drought tolerant"], "warnings": [], "growing_tip": "Intercrop with sorghum or millet for best results."},
            {"crop_name": "Dates",      "local_name": "Dates",      "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "5-10 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Perennial cash crop", "Extremely heat and drought tolerant"], "warnings": [], "growing_tip": "Hand-pollinate for reliable yields."},
            {"crop_name": "Cotton",     "local_name": "Cotton",     "suitability_score": 70, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 160, "water_need": "Medium", "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Rainy season",     "market_demand": "High",   "reasons": ["Major cash crop for arid regions", "High export value"], "warnings": ["Requires pest monitoring"], "growing_tip": "Drip irrigation significantly improves yield."},
        ]

    # ── Mediterranean ─────────────────────────────────────────────────────────
    if "mediterranean" in clim:
        return [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct - Dec",         "market_demand": "High",   "reasons": ["Ideal cool-wet winter crop", "Highest suitability for Mediterranean"], "warnings": [], "growing_tip": "Sow after first autumn rains for best germination."},
            {"crop_name": "Olive",      "local_name": "Olive",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "2-5 tons/ha",   "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Signature Mediterranean crop", "Drought tolerant perennial"], "warnings": [], "growing_tip": "Prune for open canopy to improve light penetration."},
            {"crop_name": "Grape",      "local_name": "Grape",      "suitability_score": 87, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 180, "water_need": "Low",    "estimated_yield": "5-15 tons/ha",  "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Excellent for Mediterranean climate", "High value crop"], "warnings": [], "growing_tip": "Train on trellis, prune to 2-3 buds in winter."},
            {"crop_name": "Tomato",     "local_name": "Tomato",     "suitability_score": 83, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "40-60 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value summer vegetable", "Suits Mediterranean summers"], "warnings": ["Monitor for late blight"], "growing_tip": "Drip irrigate to reduce disease pressure."},
            {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-3 tons/ha",   "planting_window": "Apr - May",         "market_demand": "Medium", "reasons": ["Drought tolerant oil crop", "Suits hot dry summers"], "warnings": [], "growing_tip": "Plant in deep, well-drained soils for best root development."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "Medium", "reasons": ["Winter cereal", "Drought tolerant", "Multiple uses"], "warnings": [], "growing_tip": "Spring barley also viable with irrigation."},
        ]

    # ── Temperate / Continental (Europe, North America, Northern Asia) ─────────
    if "temperate" in clim or "continental" in clim:
        # European-specific
        if any(eu in c_lower for eu in ["germany", "france", "poland", "ukraine", "netherlands",
                                         "belgium", "czech", "austria", "hungary", "romania",
                                         "sweden", "denmark", "finland", "norway", "switzerland",
                                         "slovakia", "hungary", "bulgaria", "serbia", "croatia",
                                         "slovenia", "estonia", "latvia", "lithuania", "ireland"]):
            season_map = {
                "Spring": [
                    {"crop_name": "Sugar Beet",   "local_name": "Zuckerrübe",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 170, "water_need": "Medium", "estimated_yield": "50-80 tons/ha",  "planting_window": "Mar 15 - Apr 30",   "market_demand": "High",   "reasons": ["Major European cash crop", "Long growing season suits climate"], "warnings": [], "growing_tip": "Precision sow at 5-8cm depth."},
                    {"crop_name": "Spring Barley", "local_name": "Sommergerste",  "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 15",    "market_demand": "High",   "reasons": ["Malting barley premium", "Suits cool spring"], "warnings": [], "growing_tip": "Choose two-row malting varieties for premium prices."},
                    {"crop_name": "Potato",        "local_name": "Kartoffel",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "30-45 tons/ha",  "planting_window": "Apr 1 - May 15",    "market_demand": "High",   "reasons": ["Core European staple", "Excellent cool-season crop"], "warnings": ["Late blight risk — scout regularly"], "growing_tip": "Mound plants as they grow; harvest when tops die back."},
                    {"crop_name": "Spring Wheat",  "local_name": "Sommerweizen",  "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 1",     "market_demand": "High",   "reasons": ["Bread wheat", "Strong European demand"], "warnings": [], "growing_tip": "Apply split nitrogen for best quality protein."},
                    {"crop_name": "Canola",        "local_name": "Raps",          "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "3-4.5 tons/ha",  "planting_window": "Mar 20 - Apr 20",   "market_demand": "High",   "reasons": ["Oil crop", "EU biodiesel market", "Good rotation crop"], "warnings": [], "growing_tip": "Scout for flea beetles and cabbage stem flea beetle early."},
                    {"crop_name": "Pea",           "local_name": "Erbse",         "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Mar 15 - Apr 15",   "market_demand": "Medium", "reasons": ["Nitrogen fixer", "Cool season legume", "Good rotation"], "warnings": [], "growing_tip": "Inoculate with Rhizobium; avoid waterlogging."},
                ],
                "Autumn": [
                    {"crop_name": "Winter Wheat",  "local_name": "Winterweizen",  "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 250, "water_need": "Medium", "estimated_yield": "6-9 tons/ha",    "planting_window": "Sep 15 - Oct 31",   "market_demand": "High",   "reasons": ["Europe's #1 arable crop", "High yield potential", "Stable prices"], "warnings": [], "growing_tip": "Choose fusarium-resistant varieties; apply fungicide at ear emergence."},
                    {"crop_name": "Winter Rape",   "local_name": "Winterraps",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Aug 15 - Sep 15",   "market_demand": "High",   "reasons": ["Oil crop, EU mandate", "Winter hardy varieties available"], "warnings": ["Watch for slugs at establishment"], "growing_tip": "Target 30-40 plants/m² at harvest — don't over-sow."},
                    {"crop_name": "Winter Barley", "local_name": "Wintergerste",  "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Sep 20 - Oct 15",   "market_demand": "High",   "reasons": ["Early harvest", "Good for maltsters"], "warnings": [], "growing_tip": "Earliest cereal harvest — frees up land for catch crops."},
                    {"crop_name": "Sugar Beet",    "local_name": "Zuckerrübe",    "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 180, "water_need": "Medium", "estimated_yield": "50-75 tons/ha",  "planting_window": "Mar 15 - Apr 30",   "market_demand": "High",   "reasons": ["High-value root crop", "Major EU crop"], "warnings": [], "growing_tip": "Leave in field until Nov-Dec for highest sugar content."},
                    {"crop_name": "Winter Rye",    "local_name": "Winterroggen",  "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 260, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep 15 - Oct 20",   "market_demand": "Medium", "reasons": ["Most cold-hardy cereal", "Suits sandy soils"], "warnings": [], "growing_tip": "Excellent on lighter, sandier soils where wheat struggles."},
                    {"crop_name": "Triticale",     "local_name": "Triticale",     "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 250, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Oct 1 - Oct 30",    "market_demand": "Medium", "reasons": ["Wheat × rye hybrid", "High biomass", "Livestock feed"], "warnings": [], "growing_tip": "Good alternative on marginal soils."},
                ],
                "Summer": [
                    {"crop_name": "Maize",         "local_name": "Mais",          "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",   "planting_window": "Apr 20 - May 20",   "market_demand": "High",   "reasons": ["Major European feed crop", "High yield"], "warnings": [], "growing_tip": "Soil temperature must exceed 8°C before sowing."},
                    {"crop_name": "Sunflower",     "local_name": "Sonnenblume",   "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",    "planting_window": "Apr 15 - May 15",   "market_demand": "High",   "reasons": ["Oil crop", "Heat and drought tolerant"], "warnings": [], "growing_tip": "Plant 6-7 seeds/m², thin to 4-5 plants."},
                    {"crop_name": "Potato",        "local_name": "Kartoffel",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "30-45 tons/ha",  "planting_window": "Apr 1 - May 1",     "market_demand": "High",   "reasons": ["European staple", "Multiple markets"], "warnings": ["Late blight: apply preventive fungicides"], "growing_tip": "Irrigate during tuber bulking for consistent yields."},
                    {"crop_name": "Field Bean",    "local_name": "Ackerbohne",    "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Mar 15 - Apr 15",   "market_demand": "High",   "reasons": ["Protein crop", "Nitrogen fixer", "EU protein strategy"], "warnings": [], "growing_tip": "Excellent rotation break from cereals."},
                    {"crop_name": "Hemp",          "local_name": "Hanf",          "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "6-10 tons/ha",   "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["Growing EU industrial hemp market", "Low pesticide need"], "warnings": ["Requires EU license to grow"], "growing_tip": "Sow at 25-35 kg/ha for fiber production."},
                    {"crop_name": "Oat",           "local_name": "Hafer",         "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 15",    "market_demand": "Medium", "reasons": ["Health food market growing", "Suits wetter soils"], "warnings": [], "growing_tip": "Tolerates wetter soils better than wheat."},
                ],
                "Winter": [
                    # Same as Autumn for Europe (winter-sown crops)
                    {"crop_name": "Winter Wheat",  "local_name": "Winterweizen",  "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 250, "water_need": "Medium", "estimated_yield": "6-9 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Core European crop"], "warnings": [], "growing_tip": "Choose varieties with high Hagberg falling number for bread-making."},
                    {"crop_name": "Winter Rape",   "local_name": "Winterraps",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Biodiesel and food oil demand"], "warnings": [], "growing_tip": "Establish a good leaf canopy before winter."},
                    {"crop_name": "Winter Barley", "local_name": "Wintergerste",  "suitability_score": 84, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Early harvest, frees up field"], "warnings": [], "growing_tip": "Use two-row varieties for malting premium."},
                    {"crop_name": "Spinach",       "local_name": "Spinat",        "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 45,  "water_need": "Medium", "estimated_yield": "15-25 tons/ha",  "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Winter salad market", "Fast crop"], "warnings": [], "growing_tip": "Harvest before bolting."},
                    {"crop_name": "Field Bean",    "local_name": "Ackerbohne",    "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 180, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Overwinter variety available", "Nitrogen fixer"], "warnings": [], "growing_tip": "Winter varieties can establish Oct-Nov."},
                    {"crop_name": "Winter Rye",    "local_name": "Winterroggen",  "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 260, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "Medium", "reasons": ["Hardiest cereal", "Low input"], "warnings": [], "growing_tip": "Best on sandy, acid soils."},
                ],
            }
            # Find best season match
            return (
                season_map.get(season) or
                season_map.get("Autumn") or
                season_map["Winter"]
            )

        # Russia / Central Asia (continental climate — similar crops to Eastern Europe)
        if any(ru in c_lower for ru in ["russia", "kazakhstan", "uzbekistan", "kyrgyzstan", "tajikistan", "turkmenistan", "mongolia", "belarus"]):
            return [
                {"crop_name": "Spring Wheat",  "local_name": "Яровая пшеница",  "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 95,  "water_need": "Low",    "estimated_yield": "2-4 tons/ha",    "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["#1 Russian staple crop", "Short-season suited to continental climate"], "warnings": [], "growing_tip": "Use drought-tolerant spring varieties for steppe regions."},
                {"crop_name": "Sunflower",     "local_name": "Подсолнечник",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 105, "water_need": "Low",    "estimated_yield": "2-3.5 tons/ha",  "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["Russia/Ukraine top oil crop", "Drought tolerant"], "warnings": [], "growing_tip": "Plant at 60-70cm row spacing; thin to 5-6 plants/m²."},
                {"crop_name": "Spring Barley", "local_name": "Яровой ячмень",   "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 80,  "water_need": "Low",    "estimated_yield": "2-4 tons/ha",    "planting_window": "May 1 - May 25",    "market_demand": "High",   "reasons": ["Beer malt and animal feed", "Early harvest for short season"], "warnings": [], "growing_tip": "Two-row malting varieties preferred by breweries."},
                {"crop_name": "Canola",        "local_name": "Рапс",            "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "1.5-3 tons/ha",  "planting_window": "May 5 - Jun 1",     "market_demand": "High",   "reasons": ["Growing Russian export crop", "Biodiesel demand"], "warnings": [], "growing_tip": "Spring canola establishment critical — plant into warm soil."},
                {"crop_name": "Soybean",       "local_name": "Соя",             "suitability_score": 76, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "1.5-3 tons/ha",  "planting_window": "May 15 - Jun 15",   "market_demand": "High",   "reasons": ["Russian Far East staple", "Export market growing"], "warnings": ["Requires warm soil — delayed planting risk in cold years"], "growing_tip": "Choose early-maturing varieties for northern regions."},
                {"crop_name": "Maize",         "local_name": "Кукуруза",        "suitability_score": 72, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 110, "water_need": "Medium", "estimated_yield": "5-9 tons/ha",    "planting_window": "May 10 - Jun 1",    "market_demand": "High",   "reasons": ["Silage and grain markets", "Southern Russia high yield"], "warnings": ["Frost risk — plant only after last frost"], "growing_tip": "Use FAO 200-300 early hybrids for most regions."},
            ]

        # Afghanistan — High-altitude semi-arid mountain climate (SEPARATE from Middle East)
        # Panjshir, Baghlan, Kabul, Parwan: cold winters, warm-dry summers, 1500-4000m elevation
        if "afghanistan" in c_lower:
            return [
                {"crop_name": "Winter Wheat",    "local_name": "Gandum",           "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",     "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["#1 Afghan staple — suited to cold highland winters", "Cold vernalisation improves yield"], "warnings": [], "growing_tip": "Sow after first autumn rains; cold-hardy varieties essential for mountain valleys."},
                {"crop_name": "Barley",          "local_name": "Jo",               "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 130, "water_need": "Low",    "estimated_yield": "1.5-3 tons/ha",   "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Most cold/drought-tolerant cereal", "Grows at higher altitude than wheat"], "warnings": [], "growing_tip": "Spring barley sown after snowmelt — key crop for high-altitude valleys like Panjshir."},
                {"crop_name": "Potato",          "local_name": "Kachalu",          "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "12-25 tons/ha",   "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Thrives in cool mountain climate", "High-calorie food security crop"], "warnings": ["Store in cool dark conditions to prevent sprouting"], "growing_tip": "Plant certified seed potatoes at 30cm spacing in well-drained mountain soils."},
                {"crop_name": "Apricot",         "local_name": "Zardalu",          "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "5-12 tons fresh/ha", "planting_window": "Plant saplings Feb-Mar", "market_demand": "High", "reasons": ["Panjshir/Baghlan famous for apricots", "Dried apricots fetch 3-5x fresh price"], "warnings": ["Late frost can damage blossoms"], "growing_tip": "Dry surplus fruit on rooftops — qurut (dried apricot) is a key export."},
                {"crop_name": "Grapes / Kishmish","local_name": "Angoor / Kishmish","suitability_score": 83, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "8-15 tons/ha",    "planting_window": "Plant cuttings Mar-Apr","market_demand": "High",   "reasons": ["Baghlan/Panjshir are historically top Afghan grape regions", "Kishmish raisins are a major export"], "warnings": [], "growing_tip": "Train on traditional stick trellises; hard prune in late winter for best fruit."},
                {"crop_name": "Cumin",           "local_name": "Zira",             "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Low",    "estimated_yield": "0.4-0.8 tons/ha",  "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Afghan cumin is highly prized", "Thrives in semi-arid highland conditions"], "warnings": [], "growing_tip": "Thin-sow in well-drained sandy loam; harvest when seeds turn brown."},
            ]

        # Turkey / Middle East / North Africa (NOT Afghanistan — handled above)
        if any(me in c_lower for me in ["turkey", "egypt", "iran", "iraq", "syria", "jordan", "lebanon",
                                         "saudi", "morocco", "algeria", "tunisia", "libya", "israel",
                                         "azerbaijan", "georgia", "armenia"]):
            return [
                {"crop_name": "Wheat",          "local_name": "Buğday/Qamh",    "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",    "planting_window": "Oct - Dec",         "market_demand": "High",   "reasons": ["Primary cereal of the Middle East", "Winter rains ideal"], "warnings": [], "growing_tip": "Sow after first autumn rains for best germination."},
                {"crop_name": "Barley",         "local_name": "Arpa/Sha'ir",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",    "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Drought tolerant cereal", "Animal feed demand"], "warnings": [], "growing_tip": "More drought tolerant than wheat — extend into drier areas."},
                {"crop_name": "Cotton",         "local_name": "Pamuk/Qotn",     "suitability_score": 83, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 180, "water_need": "High",   "estimated_yield": "1.5-3 tons/ha",  "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Major regional cash crop", "Hot summers ideal"], "warnings": ["Requires irrigation", "Monitor for bollworm"], "growing_tip": "Drip irrigation saves water; apply potassium at boll formation."},
                {"crop_name": "Sugar Beet",     "local_name": "Şeker Pancarı",  "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 180, "water_need": "Medium", "estimated_yield": "50-80 tons/ha",  "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Major Turkish/Egyptian crop", "State-supported"], "warnings": [], "growing_tip": "Precision sow for optimal canopy establishment."},
                {"crop_name": "Tomato",         "local_name": "Domates/Tamatar", "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 90,  "water_need": "High",   "estimated_yield": "40-80 tons/ha", "planting_window": "Apr - Jun",         "market_demand": "High",   "reasons": ["Hot summers ideal", "Export and processing markets"], "warnings": ["Irrigate regularly"], "growing_tip": "Stake or cage plants; apply calcium to prevent blossom-end rot."},
                {"crop_name": "Olive",          "local_name": "Zeytun/Zaytoun", "suitability_score": 78, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "2-5 tons/ha",    "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Classic Mediterranean/Middle East crop", "Perennial income"], "warnings": [], "growing_tip": "Prune biennial bearing by removing alternate year crop."},
            ]

        # Southeast Asia (Malaysia, Indonesia, Philippines, Myanmar, Cambodia, Laos, Vietnam, Thailand)
        if any(se in c_lower for se in ["malaysia", "myanmar", "cambodia", "laos", "philippines", "brunei", "timor"]):
            return [
                {"crop_name": "Rice",           "local_name": "Padi/Sraov",     "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "3-5 tons/ha",    "planting_window": "Start of rains",    "market_demand": "High",   "reasons": ["Primary staple crop", "Wet tropical climate ideal"], "warnings": [], "growing_tip": "Transplant seedlings at 3-4 leaf stage into flooded fields."},
                {"crop_name": "Oil Palm",       "local_name": "Kelapa Sawit",   "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "20-30 tons/ha FFB","planting_window": "Year-round",        "market_demand": "High",   "reasons": ["#1 SE Asia cash crop", "Year-round harvest"], "warnings": [], "growing_tip": "Frond-stacking essential for mulch and erosion control."},
                {"crop_name": "Cassava",        "local_name": "Ubi Kayu/Singkong","suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",   "duration_days": 270, "water_need": "Low",    "estimated_yield": "15-30 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Export starch crop", "Very drought tolerant"], "warnings": [], "growing_tip": "Plant stem cuttings at 45° angle for best rooting."},
                {"crop_name": "Maize",          "local_name": "Jagung/Mais",    "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-7 tons/ha",    "planting_window": "Dry season",        "market_demand": "High",   "reasons": ["Animal feed demand", "Fast growing in warm climate"], "warnings": [], "growing_tip": "Plant at start of dry season with residual moisture."},
                {"crop_name": "Rubber",         "local_name": "Getah/Karet",    "suitability_score": 78, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "1-2 tons latex/ha","planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Major export crop", "Perennial income"], "warnings": [], "growing_tip": "Tap trees early morning for maximum latex yield."},
                {"crop_name": "Banana",         "local_name": "Pisang/Saging",  "suitability_score": 75, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 300, "water_need": "High",   "estimated_yield": "25-40 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Export and local market", "Year-round crop"], "warnings": [], "growing_tip": "Remove suckers to leave 2-3 per stool."},
            ]

        # Mexico / Central America / Caribbean
        if any(mx in c_lower for mx in ["mexico", "guatemala", "honduras", "cuba", "costa rica",
                                         "nicaragua", "el salvador", "panama", "belize", "haiti",
                                         "dominican", "jamaica", "trinidad"]):
            return [
                {"crop_name": "Maize",          "local_name": "Maíz",           "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-8 tons/ha",    "planting_window": "Apr - Jun",         "market_demand": "High",   "reasons": ["Origin of maize — diverse varieties", "Central to regional diet"], "warnings": [], "growing_tip": "Traditional milpa system: grow with beans and squash."},
                {"crop_name": "Sorghum",        "local_name": "Sorgo",          "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",    "planting_window": "May - Jul",         "market_demand": "High",   "reasons": ["Drought tolerant alternative to maize", "Animal feed"], "warnings": [], "growing_tip": "Excellent on dry upland areas unsuitable for maize."},
                {"crop_name": "Bean",           "local_name": "Frijol",         "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "1-2 tons/ha",    "planting_window": "Jun - Aug",         "market_demand": "High",   "reasons": ["Dietary staple — eaten daily", "Nitrogen fixer"], "warnings": [], "growing_tip": "Inoculate seeds with Rhizobium before planting."},
                {"crop_name": "Sugarcane",      "local_name": "Caña de azúcar", "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "60-120 tons/ha", "planting_window": "Oct - Dec",         "market_demand": "High",   "reasons": ["Major regional cash crop", "Sugar and ethanol markets"], "warnings": [], "growing_tip": "Ratoon crop for 3-5 years to reduce establishment cost."},
                {"crop_name": "Avocado",        "local_name": "Aguacate",       "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "10-20 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["High export value", "Origin of Hass avocado"], "warnings": [], "growing_tip": "Graft Hass onto Dusa rootstock for best yield and disease resistance."},
                {"crop_name": "Chili Pepper",   "local_name": "Chile",          "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "8-15 tons/ha",   "planting_window": "Mar - May",         "market_demand": "High",   "reasons": ["Central to Mexican cuisine", "Export demand growing"], "warnings": [], "growing_tip": "Transplant 6-week-old seedlings; mulch to retain moisture."},
            ]

        # Chile — Mediterranean climate (separate from tropical South America)
        # Central Chile: dry hot summers, wet mild winters — similar to Spain/California
        if "chile" in c_lower:
            return [
                {"crop_name": "Wheat",       "local_name": "Trigo",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Jun - Jul",         "market_demand": "High",   "reasons": ["Chile's #1 cereal", "Mediterranean winter rains ideal"], "warnings": [], "growing_tip": "Sow after first winter rains for best germination."},
                {"crop_name": "Grape (Wine)","local_name": "Vid/Uva",     "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "8-20 tons/ha",   "planting_window": "Plant vines Aug-Sep", "market_demand": "High",   "reasons": ["Chile is a world-top wine exporter", "Mediterranean climate ideal for viticulture"], "warnings": [], "growing_tip": "Carmenere, Cabernet Sauvignon and Sauvignon Blanc excel in Colchagua/Maipo valleys."},
                {"crop_name": "Apple",       "local_name": "Manzana",     "suitability_score": 87, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "40-60 tons/ha",  "planting_window": "Plant Aug-Sep",    "market_demand": "High",   "reasons": ["Chile is a major Southern Hemisphere apple exporter", "Maule/Biobio ideal climate"], "warnings": [], "growing_tip": "Requires chilling hours — Gala and Fuji suited for Maule region."},
                {"crop_name": "Blueberry",   "local_name": "Arándano",    "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "8-14 tons/ha",   "planting_window": "Plant Jun-Aug",    "market_demand": "High",   "reasons": ["Chile is world's #2 blueberry exporter", "Counter-season to Northern Hemisphere"], "warnings": [], "growing_tip": "Acidic soil (pH 4.5-5.5) essential; drip irrigate."},
                {"crop_name": "Cherry",      "local_name": "Cereza",      "suitability_score": 82, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 365, "water_need": "Medium", "estimated_yield": "10-18 tons/ha",  "planting_window": "Plant Jun-Jul",    "market_demand": "High",   "reasons": ["Chile is world's #1 cherry exporter to China", "Premium counter-season prices"], "warnings": ["Rain at harvest can crack fruit"], "growing_tip": "Lapins and Sweetheart varieties dominate export market."},
                {"crop_name": "Canola",      "local_name": "Raps",        "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 150, "water_need": "Medium", "estimated_yield": "2-3.5 tons/ha",  "planting_window": "Jun - Jul",        "market_demand": "High",   "reasons": ["Growing Southern Chile biofuel and food oil market"], "warnings": [], "growing_tip": "Plant in Araucania/Los Lagos for best rainfall conditions."},
            ]

        # South America (except Brazil/Argentina handled separately, and Chile above)
        if any(sa in c_lower for sa in ["colombia", "peru", "ecuador", "bolivia",
                                         "venezuela", "paraguay", "uruguay", "guyana", "suriname"]):
            return [
                {"crop_name": "Soybean",        "local_name": "Soja",           "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "2.5-4 tons/ha",  "planting_window": "Oct - Dec",         "market_demand": "High",   "reasons": ["Major South American export", "China demand strong"], "warnings": [], "growing_tip": "Inoculate with Bradyrhizobium japonicum for nitrogen fixation."},
                {"crop_name": "Maize",          "local_name": "Maíz",           "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "4-8 tons/ha",    "planting_window": "Sep - Dec",         "market_demand": "High",   "reasons": ["Food and feed crop", "Multiple growing seasons"], "warnings": [], "growing_tip": "Use no-till in soybean-maize rotation."},
                {"crop_name": "Coffee",         "local_name": "Café",           "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 365, "water_need": "Medium", "estimated_yield": "0.5-3 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Premium export crop", "Highland climate ideal"], "warnings": ["Coffee leaf rust — use resistant varieties"], "growing_tip": "Shade-grown coffee commands specialty price premium."},
                {"crop_name": "Potato",         "local_name": "Papa",           "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "15-30 tons/ha",  "planting_window": "Year-round (highland)","market_demand": "High",  "reasons": ["Origin of the potato — over 4,000 native varieties", "Andean staple"], "warnings": ["Late blight: monitor regularly"], "growing_tip": "Native andean varieties often more blight resistant."},
                {"crop_name": "Sugarcane",      "local_name": "Caña de azúcar", "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "60-100 tons/ha", "planting_window": "Apr - Jun",         "market_demand": "High",   "reasons": ["Sugar and ethanol market", "Tropical lowlands ideal"], "warnings": [], "growing_tip": "Ratoon 3-5 years before replanting."},
                {"crop_name": "Banana",         "local_name": "Banano/Plátano", "suitability_score": 75, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 300, "water_need": "High",   "estimated_yield": "25-40 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Major export crop", "Year-round income"], "warnings": [], "growing_tip": "Remove excess suckers; retain only 1-2 daughter plants per stool."},
            ]

        # North America
        if any(na in c_lower for na in ["united states", "canada"]):
            return [
                {"crop_name": "Corn",       "local_name": "Corn",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "10-14 tons/ha",  "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["#1 US crop", "Strong export demand", "Multiple uses"], "warnings": [], "growing_tip": "Plant at 76cm row spacing for highest yield."},
                {"crop_name": "Soybean",    "local_name": "Soybean",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",    "planting_window": "May 1 - Jun 15",    "market_demand": "High",   "reasons": ["Strong export market", "Nitrogen fixer"], "warnings": [], "growing_tip": "Inoculate with Bradyrhizobium; plant after corn frost risk passes."},
                {"crop_name": "Winter Wheat","local_name": "Wheat",     "suitability_score": 84, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep 15 - Oct 31",   "market_demand": "High",   "reasons": ["Plains staple crop", "Government price support"], "warnings": [], "growing_tip": "Plant Hessian-fly-safe varieties after the fly-free date."},
                {"crop_name": "Canola",     "local_name": "Canola",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 95,  "water_need": "Medium", "estimated_yield": "2-3 tons/ha",    "planting_window": "Apr 15 - May 15",   "market_demand": "High",   "reasons": ["Prairie oil crop", "High value", "Good rotation"], "warnings": [], "growing_tip": "Canola Club practices: 3-4 year rotation essential."},
                {"crop_name": "Cotton",     "local_name": "Cotton",     "suitability_score": 76, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 155, "water_need": "Medium", "estimated_yield": "1.2-2 tons/ha",  "planting_window": "Apr 15 - May 30",   "market_demand": "High",   "reasons": ["Southern US staple", "Strong export market"], "warnings": ["Boll weevil monitoring required"], "growing_tip": "Defoliate before harvest for cleaner cotton."},
                {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-3 tons/ha",    "planting_window": "May 15 - Jun 15",   "market_demand": "Medium", "reasons": ["Plains oil crop", "Drought tolerant"], "warnings": [], "growing_tip": "Leave 2-3 weeks after frost-safe dates."},
            ]

        # General temperate fallback
        return [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Sep - Nov",         "market_demand": "High",   "reasons": ["Global staple crop", "Suits temperate climate"], "warnings": [], "growing_tip": "Choose disease-resistant varieties for your region."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",   "planting_window": "Sep - Oct / Mar - Apr", "market_demand": "High",   "reasons": ["Drought tolerant cereal", "Malting premium available"], "warnings": [], "growing_tip": "Two-row varieties preferred for malting."},
            {"crop_name": "Canola",     "local_name": "Rapeseed",   "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",   "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Oil crop", "Strong global demand"], "warnings": [], "growing_tip": "Establish before winter for best spring growth."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Mar - May",         "market_demand": "High",   "reasons": ["Global staple", "High yield per hectare"], "warnings": ["Late blight"], "growing_tip": "Use certified seed for disease-free crop."},
            {"crop_name": "Maize",      "local_name": "Maize/Corn", "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["High yield", "Multiple uses — feed, food, biogas"], "warnings": [], "growing_tip": "Wait for soil to reach 8°C before planting."},
            {"crop_name": "Oat",        "local_name": "Oat",        "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Mar - Apr",         "market_demand": "Medium", "reasons": ["Growing health food demand", "Suits wetter soils"], "warnings": [], "growing_tip": "Sow early for best yields."},
        ]

    # ── Sub-Saharan Africa ────────────────────────────────────────────────────
    if "ethiopia" in c_lower:
        return [
            {"crop_name": "Teff",           "local_name": "Teffa",      "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Low",    "estimated_yield": "1-2 tons/ha",    "planting_window": "Jun - Jul (Meher)", "market_demand": "High",   "reasons": ["Indigenous Ethiopian grain", "Basis of Injera national staple"], "warnings": [], "growing_tip": "Broadcast sow at 5-10 kg/ha; does not transplant well."},
            {"crop_name": "Coffee (Arabica)","local_name": "Buna",       "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "0.5-2 tons/ha",  "planting_window": "Perennial",         "market_demand": "High",   "reasons": ["Ethiopia is origin of Arabica coffee", "Premium specialty export"], "warnings": ["Coffee leaf rust - use resistant clones"], "growing_tip": "Forest coffee at 1500-2000m altitude commands highest price."},
            {"crop_name": "Wheat",           "local_name": "Senafich",   "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "2-4 tons/ha",    "planting_window": "Jun - Jul",         "market_demand": "High",   "reasons": ["Highland wheat belt - Bale, Arsi zones"], "warnings": [], "growing_tip": "Use improved CIMMYT varieties for best highland performance."},
            {"crop_name": "Maize",           "local_name": "Baqolo",     "suitability_score": 83, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "3-6 tons/ha",    "planting_window": "Apr - May (Belg)",  "market_demand": "High",   "reasons": ["Staple food crop", "Suits mid-altitude zones"], "warnings": [], "growing_tip": "Hybrid varieties from CIMMYT/CESA outperform local."},
            {"crop_name": "Sorghum",         "local_name": "Mashila",    "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",    "planting_window": "Jun - Jul",         "market_demand": "High",   "reasons": ["Drought-tolerant staple for lowlands"], "warnings": [], "growing_tip": "Key crop for Afar and Somali lowland regions."},
            {"crop_name": "Chickpea",        "local_name": "Shimbra",    "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1-2 tons/ha",    "planting_window": "Oct - Nov (Rabi)",  "market_demand": "High",   "reasons": ["Major Ethiopian export pulse", "Nitrogen fixer"], "warnings": [], "growing_tip": "Sow after main rains; requires cool dry finish."},
        ]

    if "kenya" in c_lower:
        return [
            {"crop_name": "Tea",             "local_name": "Chai",       "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "2-4 tons/ha",    "planting_window": "Perennial",         "market_demand": "High",   "reasons": ["Kenya is world #3 tea exporter", "Kericho/Nyeri highlands ideal"], "warnings": [], "growing_tip": "Pluck 2 leaves and a bud every 7-12 days."},
            {"crop_name": "Coffee (AA)",     "local_name": "Kahawa",     "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "0.5-2 tons/ha",  "planting_window": "Perennial",         "market_demand": "High",   "reasons": ["Kenya AA is one of the world's premium coffees"], "warnings": ["Coffee Berry Disease - spray copper-based fungicide"], "growing_tip": "Grow at 1500-2100m elevation for best cup quality."},
            {"crop_name": "Maize",           "local_name": "Mahindi",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "3-6 tons/ha",    "planting_window": "Mar - Apr (LR)",    "market_demand": "High",   "reasons": ["Kenya's staple food", "Used for Ugali"], "warnings": [], "growing_tip": "Use H614D or H628 hybrid varieties."},
            {"crop_name": "Cut Flowers",     "local_name": "Maua",       "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 365, "water_need": "High",   "estimated_yield": "1.2M stems/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Kenya is world #1 cut rose exporter", "Naivasha highlands ideal"], "warnings": ["High input cost", "Cold chain required"], "growing_tip": "Target European auction market (Aalsmeer) for highest prices."},
            {"crop_name": "Wheat",           "local_name": "Ngano",      "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "2-4 tons/ha",    "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Uasin Gishu plateau - Kenya wheat belt"], "warnings": [], "growing_tip": "Use Fahari variety for best yields."},
            {"crop_name": "Avocado",         "local_name": "Parachichi", "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "10-20 tons/ha",  "planting_window": "Perennial",         "market_demand": "High",   "reasons": ["Growing EU export market", "Murang'a county leading producer"], "warnings": [], "growing_tip": "Hass variety dominates export market; graft onto Mexicola rootstock."},
        ]

    if "south africa" in c_lower:
        return [
            {"crop_name": "Maize",           "local_name": "Mielies",    "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "4-8 tons/ha",    "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["SA's #1 grain crop - Free State, North West", "Summer rain ideal"], "warnings": [], "growing_tip": "Plant DK647 or PAN hybrids at 35,000 plants/ha for dryland."},
            {"crop_name": "Grape (Wine)",    "local_name": "Druiwe",     "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "10-25 tons/ha",  "planting_window": "Plant Jul-Aug",     "market_demand": "High",   "reasons": ["Western Cape Mediterranean climate ideal", "Top 10 wine exporter"], "warnings": [], "growing_tip": "Chenin Blanc, Pinotage and Cabernet Sauvignon dominate Stellenbosch."},
            {"crop_name": "Wheat",           "local_name": "Koring",     "suitability_score": 87, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",    "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Swartland and Overberg - SA wheat belt"], "warnings": [], "growing_tip": "Kariega and Krokodil varieties perform best in Western Cape."},
            {"crop_name": "Citrus",          "local_name": "Sitrus",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "40-60 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["SA is world's #1 citrus exporter", "Limpopo, Western Cape ideal"], "warnings": [], "growing_tip": "Navel oranges for fresh export; Valencia for juice market."},
            {"crop_name": "Sugarcane",       "local_name": "Suikerriet", "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "High",   "estimated_yield": "70-100 tons/ha", "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["KwaZulu-Natal coast - SA sugarcane belt"], "warnings": [], "growing_tip": "Ratoon for 5-7 years; mechanised harvest in most areas."},
            {"crop_name": "Canola",          "local_name": "Canola",     "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 150, "water_need": "Medium", "estimated_yield": "1.5-2.5 tons/ha", "planting_window": "Apr - May",        "market_demand": "High",   "reasons": ["Growing SA oilseed crop", "Western Cape rotation crop"], "warnings": [], "growing_tip": "Sow early to avoid late heat during pod fill."},
        ]

    if any(af in c_lower for af in ["nigeria", "ghana", "ivory coast", "cote d'ivoire", "cameroon",
                                      "senegal", "mali", "burkina faso", "guinea", "sierra leone",
                                      "liberia", "togo", "benin", "niger"]):
        return [
            {"crop_name": "Cassava",         "local_name": "Yuca/Manioc","suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Low",    "estimated_yield": "15-25 tons/ha",  "planting_window": "Start of rains",    "market_demand": "High",   "reasons": ["West Africa staple - Nigeria is world #1 producer"], "warnings": [], "growing_tip": "Use improved TME419 varieties; plant stakes from healthy stems."},
            {"crop_name": "Maize",           "local_name": "Corn/Agbado","suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Apr-May / Aug-Sep", "market_demand": "High",   "reasons": ["West Africa staple", "Two seasons possible"], "warnings": [], "growing_tip": "Stagger plantings across both rainy seasons for continuous income."},
            {"crop_name": "Cocoa",           "local_name": "Cacao/Koko", "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 365, "water_need": "High",   "estimated_yield": "0.5-2 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Ivory Coast and Ghana produce 60% of world's cocoa"], "warnings": ["Swollen shoot virus - remove infected trees"], "growing_tip": "Intercrop with plantain for shade during establishment."},
            {"crop_name": "Yam",             "local_name": "Yam/Iyan",   "suitability_score": 85, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 210, "water_need": "Medium", "estimated_yield": "15-25 tons/ha",  "planting_window": "Feb - Mar",         "market_demand": "High",   "reasons": ["Nigeria yam belt - cultural staple", "High market value"], "warnings": [], "growing_tip": "Plant minisetts to reduce planting material cost."},
            {"crop_name": "Groundnut",       "local_name": "Peanut/Epa", "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Protein and oil crop", "Important for Sahel nutrition"], "warnings": [], "growing_tip": "Sandy loam soils best; avoid waterlogging during pegging stage."},
            {"crop_name": "Rice",            "local_name": "Rice/Iresi", "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "3-5 tons/ha",    "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Urban demand for rice growing rapidly in West Africa"], "warnings": [], "growing_tip": "NERICA varieties suited for upland dry environments."},
        ]

    if any(af in c_lower for af in ["tanzania", "uganda", "rwanda", "burundi", "zambia", "zimbabwe",
                                      "malawi", "mozambique", "madagascar", "angola"]):
        return [
            {"crop_name": "Maize",           "local_name": "Mahindi",    "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "3-6 tons/ha",    "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["East/Southern Africa staple", "Nsima/Ugali diet basis"], "warnings": [], "growing_tip": "Use ZM521, SEEDCO or Pioneer varieties for best hybrid yield."},
            {"crop_name": "Tobacco",         "local_name": "Tumbaku",    "suitability_score": 88, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 120, "water_need": "Medium", "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Nov - Dec",         "market_demand": "High",   "reasons": ["Zimbabwe and Malawi top African tobacco exporters"], "warnings": ["Labour intensive", "Market price volatility"], "growing_tip": "Flue-cured Virginia ideal for sandy soils of Zimbabwe."},
            {"crop_name": "Coffee (Robusta)","local_name": "Kahawa",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Medium", "estimated_yield": "1-3 tons/ha",    "planting_window": "Perennial",         "market_demand": "High",   "reasons": ["Uganda and Tanzania major producers", "Robusta suited to lower altitude"], "warnings": [], "growing_tip": "Intercrop with banana/shade trees to improve bean quality."},
            {"crop_name": "Cassava",         "local_name": "Muhogo",     "suitability_score": 83, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 270, "water_need": "Low",    "estimated_yield": "15-25 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Drought-tolerant staple", "Food security crop"], "warnings": [], "growing_tip": "Use CMD-resistant varieties to avoid Cassava Mosaic Disease."},
            {"crop_name": "Sunflower",       "local_name": "Alizeti",    "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1-2 tons/ha",    "planting_window": "Feb - Mar",         "market_demand": "High",   "reasons": ["Drought-tolerant oilseed", "Tanzania/Zimbabwe growing market"], "warnings": [], "growing_tip": "Plant at 75x25cm spacing for open-pollinated varieties."},
            {"crop_name": "Soybean",         "local_name": "Soya",       "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "1.5-3 tons/ha",  "planting_window": "Nov - Dec",         "market_demand": "High",   "reasons": ["Growing animal feed demand", "Zambia/Zimbabwe expanding area"], "warnings": [], "growing_tip": "Inoculate seeds with Bradyrhizobium; avoid heavy clay soils."},
        ]

    # ── South Asian (India, Pakistan, Bangladesh) ─────────────────────────────
    if is_south_asia:

        return [
            {"crop_name": "Rice",       "local_name": "Dhan",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 15 - Jul 15", "market_demand": "High",   "reasons": ["Primary Kharif crop", "Thrives in monsoon rainfall"], "warnings": [], "growing_tip": "Transplant seedlings when soil is saturated."},
            {"crop_name": "Wheat",      "local_name": "Gehun",      "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",   "planting_window": "Nov 1 - Dec 1",   "market_demand": "High",   "reasons": ["Primary Rabi crop", "MSP support", "Cooler conditions"], "warnings": [], "growing_tip": "Irrigate at crown root initiation stage."},
            {"crop_name": "Maize",      "local_name": "Makka",      "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 1 - Jul 1",   "market_demand": "High",   "reasons": ["Fast growing", "Good market prices"], "warnings": [], "growing_tip": "Apply nitrogen at knee-high stage."},
            {"crop_name": "Chickpea",   "local_name": "Chana",      "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1-2 tons/ha",   "planting_window": "Oct 20 - Nov 10", "market_demand": "High",   "reasons": ["Nitrogen fixing legume", "Rabi crop"], "warnings": [], "growing_tip": "Avoid irrigation after flowering."},
            {"crop_name": "Cotton",     "local_name": "Kapas",      "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 180, "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "May 15 - Jun 15", "market_demand": "High",   "reasons": ["Cash crop", "Warm conditions ideal"], "warnings": ["Monitor for bollworm"], "growing_tip": "Use Bt cotton varieties."},
            {"crop_name": "Mustard",    "local_name": "Sarson",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Oct 15 - Nov 15", "market_demand": "High",   "reasons": ["Rabi oil crop", "Short duration", "Low water"], "warnings": [], "growing_tip": "Sow in sandy loam for best yield."},
        ]

    # ── Default: use season as key ────────────────────────────────────────────
    _SEASON_DEFAULTS = {
        "Spring": [
            {"crop_name": "Pea",        "local_name": "Pea",        "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 70,  "water_need": "Medium", "estimated_yield": "5-8 tons/ha",   "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Cool season legume", "Spring ideal"], "warnings": [], "growing_tip": "Plant as soon as soil can be worked."},
            {"crop_name": "Lettuce",    "local_name": "Lettuce",    "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 45,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Fast growing", "High value", "Cool weather"], "warnings": ["Bolt in heat"], "growing_tip": "Harvest before temperature exceeds 25°C."},
            {"crop_name": "Carrot",     "local_name": "Carrot",     "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "30-40 tons/ha", "planting_window": "Mar - May",         "market_demand": "High",   "reasons": ["Cool season crop", "High nutritional value"], "warnings": [], "growing_tip": "Loose, deep soil gives best root formation."},
            {"crop_name": "Onion",      "local_name": "Onion",      "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["High demand", "Long shelf life"], "warnings": ["Monitor for thrips"], "growing_tip": "Stop irrigation 2 weeks before harvest."},
            {"crop_name": "Spinach",    "local_name": "Spinach",    "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 40,  "water_need": "Medium", "estimated_yield": "10-15 tons/ha", "planting_window": "Feb - Apr",         "market_demand": "High",   "reasons": ["Cold tolerant", "Fast crop", "Nutritious"], "warnings": [], "growing_tip": "Multiple harvests — take outer leaves only."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 70, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Staple crop", "High yield"], "warnings": ["Late blight"], "growing_tip": "Use certified seed potatoes."},
        ],
        "Summer": [
            {"crop_name": "Maize",      "local_name": "Corn",       "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Warm season crop", "High yield", "Multiple uses"], "warnings": [], "growing_tip": "Plant in well-drained soil with full sun."},
            {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Drought tolerant", "Good oil crop", "Heat resistant"], "warnings": [], "growing_tip": "Ensure 60cm spacing between plants."},
            {"crop_name": "Tomato",     "local_name": "Tomato",     "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 75,  "water_need": "High",   "estimated_yield": "30-40 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value crop", "Warm season ideal"], "warnings": ["Irrigate regularly"], "growing_tip": "Use shade nets during peak summer."},
            {"crop_name": "Pepper",     "local_name": "Pepper",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 80,  "water_need": "High",   "estimated_yield": "20-25 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value vegetable", "Good summer crop"], "warnings": ["Monitor for aphids"], "growing_tip": "Mulch to keep roots cool."},
            {"crop_name": "Cucumber",   "local_name": "Cucumber",   "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 55,  "water_need": "Medium", "estimated_yield": "15-20 tons/ha", "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Short duration", "High summer demand"], "warnings": [], "growing_tip": "Use mulching to retain soil moisture."},
            {"crop_name": "Watermelon", "local_name": "Watermelon", "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 75,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["Heat tolerant", "High market demand"], "warnings": [], "growing_tip": "Sandy loam soils give best flavor."},
        ],
        "Autumn": [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Prime autumn crop", "Cold tolerant", "High demand"], "warnings": [], "growing_tip": "Choose winter-hardy varieties for your region."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "Medium", "reasons": ["Drought tolerant", "Short growing season"], "warnings": [], "growing_tip": "Excellent rotation crop with legumes."},
            {"crop_name": "Canola",     "local_name": "Rapeseed",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 220, "water_need": "Medium", "estimated_yield": "2-4 tons/ha",   "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Good oil crop", "Cold resistant"], "warnings": [], "growing_tip": "Scout for flea beetles early."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["High yield", "Versatile market demand"], "warnings": ["Store in cool dry conditions"], "growing_tip": "Mound soil over plants as they grow."},
            {"crop_name": "Garlic",     "local_name": "Garlic",     "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "8-12 tons/ha",  "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Long shelf life", "High demand", "Low water needs"], "warnings": [], "growing_tip": "Plant cloves 5cm deep, tip up."},
            {"crop_name": "Onion",      "local_name": "Onion",      "suitability_score": 68, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 150, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Overwintered onion", "Early harvest next spring"], "warnings": [], "growing_tip": "Overwintering varieties — Japanese sets recommended."},
        ],
        "Winter": [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Cold season staple", "High demand"], "warnings": [], "growing_tip": "Vernalization improves yield in cold climates."},
            {"crop_name": "Rye",        "local_name": "Rye",        "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Sep - Oct",         "market_demand": "Medium", "reasons": ["Very cold hardy", "Low input crop", "Good cover crop"], "warnings": [], "growing_tip": "Can be planted later than wheat."},
            {"crop_name": "Spinach",    "local_name": "Spinach",    "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 40,  "water_need": "Medium", "estimated_yield": "10-15 tons/ha", "planting_window": "Oct - Feb",         "market_demand": "High",   "reasons": ["Cold tolerant leafy green", "Fast crop"], "warnings": [], "growing_tip": "Multiple cuts possible — harvest outer leaves."},
            {"crop_name": "Garlic",     "local_name": "Garlic",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "8-12 tons/ha",  "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Long shelf life", "High demand", "Low water needs"], "warnings": [], "growing_tip": "Plant cloves 5cm deep, tip up."},
            {"crop_name": "Kale",       "local_name": "Kale",       "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "15-25 tons/ha", "planting_window": "Jul - Sep",         "market_demand": "High",   "reasons": ["Cold hardy", "Growing health food market"], "warnings": [], "growing_tip": "Tastes sweeter after frost — harvest Nov-Feb."},
            {"crop_name": "Leek",       "local_name": "Leek",       "suitability_score": 68, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 150, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Winter vegetable", "High market demand"], "warnings": [], "growing_tip": "Blanch stems by earthing up as plants grow."},
        ],
    }

    return (
        _SEASON_DEFAULTS.get(season)
        or _SEASON_DEFAULTS.get("Summer")
    )
