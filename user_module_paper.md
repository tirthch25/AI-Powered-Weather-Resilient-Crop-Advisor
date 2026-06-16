# User Module Paper
## AI Powered Weather Resilient Crop Advisor — v3.1

**Document Type:** User Module Paper  
**Project:** AI Powered Weather Resilient Crop Advisor  
**Version:** 3.1  
**Platform:** Web Application (FastAPI + HTML/CSS/JS + SSE)  
**Repository:** https://github.com/tirthch25/AI-Powered-Weather-Resilient-Crop-Advisor  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Agent Modules (v3.1)](#3-agent-modules-v31)
   - 3.1 Location Agent
   - 3.2 LLM Location Agent
   - 3.3 Data Gathering Agent
   - 3.4 Climate Signal Intelligence Service
   - 3.5 Crop Agent
   - 3.6 Web Search Agent
4. [Service Modules](#4-service-modules)
   - 4.1 LLM Chat
   - 4.2 LLM Explainer
   - 4.3 LLM Filter
   - 4.4 Recommender (Legacy v2.x)
   - 4.5 Risk Assessment
   - 4.6 Pest & Disease Warning
   - 4.7 Planting Calendar
5. [API Layer](#5-api-layer)
   - 5.1 Request Models
   - 5.2 All Endpoints
   - 5.3 Streaming Response Protocol
6. [Frontend & UI Engine](#6-frontend--ui-engine)
   - 6.1 Dashboard Structure
   - 6.2 SSE Client
   - 6.3 Climate Intelligence Panel
7. [Data Layer](#7-data-layer)
   - 7.1 World Locations
   - 7.2 Crop Knowledge Database
   - 7.3 Climate Zone Tables
   - 7.4 Market Price Templates
   - 7.5 Legacy Data (v2.x)
8. [Machine Learning Pipeline](#8-machine-learning-pipeline)
   - 8.1 Random Forest Suitability Model
   - 8.2 LSTM Weather Forecasting
   - 8.3 XGBoost Weather Model
9. [LLM Integration Details](#9-llm-integration-details)
   - 9.1 Provider Priority Chain
   - 9.2 Gemini Key Rotation
   - 9.3 Google Search Grounding
   - 9.4 Model Fallback Chain
   - 9.5 In-Memory Caching
10. [Climate Signal Intelligence — Deep Dive](#10-climate-signal-intelligence--deep-dive)
    - 10.1 ENSO Detection
    - 10.2 9 Threat Assessors
    - 10.3 ENSO Zone Mapping
    - 10.4 Forecast Adjustment Formula
    - 10.5 Cyclone Basin Tracking
11. [Crop Agent — Deep Dive](#11-crop-agent--deep-dive)
    - 11.1 Country Crop Hints (50+ Countries)
    - 11.2 Validation Layer
    - 11.3 Fallback Tables by Climate Zone
    - 11.4 Prompt Engineering
12. [Data Flow Diagram (v3.1)](#12-data-flow-diagram-v31)
13. [Global API Reference](#13-global-api-reference)
14. [Technology Stack & Dependencies](#14-technology-stack--dependencies)
15. [System Limitations & Future Scope](#15-system-limitations--future-scope)

---

## 1. Project Overview

The **AI Powered Weather Resilient Crop Advisor v3.1** is a globally scalable, agentic agricultural advisory web application. It combines live satellite-quality weather data, real-time ENSO climate signals, comprehensive 9-dimensional climate threat assessment, and LLM agents with Google Search Grounding to replicate the advice of an experienced agronomist for any farming region in the world.

### 1.1 Evolution from Prior Versions

| Version | Scope | Key Capability |
|---------|-------|---------------|
| **v2.x** | India-only, 640 districts | Static rule-based engine + basic ML models |
| **v3.0** | Global, 50+ countries | Multi-agent LLM pipeline, Open-Meteo live weather |
| **v3.1** | Global + Climate Intelligence | 9-dimensional climate assessment, Search Grounding, Web Search Agent |

### 1.2 Goals

| Goal | Implementation |
|------|---------------|
| **Global Scale** | Dynamically resolve any global farm location via static DB or LLM geocoding |
| **Agentic AI** | 6-agent pipeline with specialized roles: Location, LLM-Location, Data, Climate, Crop, Web-Search |
| **Real-Time Context** | Open-Meteo live weather + NOAA ENSO + Gemini Search Grounding for current advisories |
| **Climate Resilience** | 9 climate threat assessors + ENSO-adjusted 6-month forecasts |
| **Conversational UX** | Continuous SSE streaming for analysis + context-aware LLM farmer chat |
| **Graceful Degradation** | 5-tier fallback: Search Grounding → Plain LLM → Ollama → Rule-based → Zone defaults |

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph L1["🖥️  PRESENTATION (Web Browser)"]
        UI["<b>Global Dashboard</b><br/>Inputs: Country, State, District, Irrigation, Soil<br/>Outputs: Streaming JSON via SSE<br/>Components: Weather Card, Climate Panel,<br/>Soil Card, Forecast Chart, Market Prices,<br/>Crop Rankings, Farmer Chat"]
    end

    subgraph L2["🔌  API GATEWAY (FastAPI / Uvicorn)"]
        API["<b>POST /api/analyze/stream</b> (SSE)<br/><b>POST /chat/stream</b> (SSE)<br/><b>GET /climate-signals</b><br/><b>GET /api/countries|states|districts</b><br/><b>GET /health</b>"]
    end

    subgraph L3["🧠  AGENTIC INTELLIGENCE (v3.1)"]
        LA["📍 <b>Location Agent</b><br/>world_locations.json<br/>50+ Countries, 250+ States<br/>170+ Districts"]
        LLA["🗺️ <b>LLM Location Agent</b><br/>Gemini/Ollama geocoder<br/>for unmapped rural locations"]
        DGA["🌦️ <b>Data Gathering Agent</b><br/>• Open-Meteo (live wx, 30-min cache)<br/>• 6-month forecast (zone + live anchor)<br/>• ENSO adjustment (climate_signals.py)<br/>• Soil + Market (Search Grounding)"]
        CSI["🌐 <b>Climate Signal Intelligence</b><br/>• NOAA ONI (ENSO phase)<br/>• Heat Stress assessor<br/>• Drought Index<br/>• Frost Risk<br/>• Flood Detection<br/>• Cyclone Basin tracking<br/>• Wildfire Risk<br/>• Gemini Search Grounding"]
        CA["🌱 <b>Crop Agent</b><br/>Cache → Gemini+Search → Gemini<br/>→ Ollama+Search → Ollama<br/>→ Zone Fallback<br/>Country Crop Hint Validation"]
        WSA["🔍 <b>Web Search Agent</b><br/>DuckDuckGo tool for Ollama<br/>when Gemini not available"]
    end

    subgraph L4["💬  CHAT & EXPLAIN"]
        CHAT["LLM Farmer Chat<br/>(llm_chat.py)<br/>SSE token stream"]
        EXP["LLM Explainer<br/>(llm_explainer.py)<br/>Crop explanation"]
    end

    subgraph L5["💾  DATA & ML"]
        DB["crop_knowledge.json<br/>world_locations.json<br/>regions.json (legacy)"]
        ML["Random Forest (crop)<br/>LSTM (weather, legacy)<br/>XGBoost (weather, legacy)"]
    end

    L1 -->|"SSE / HTTP"| L2
    L2 --> L3
    L3 -->|"LLM calls"| L4
    L3 --> L5
```

---

## 3. Agent Modules (v3.1)

**Location:** `agri_crop_recommendation/src/agents/`

### 3.1 Location Agent (`location_agent.py`)

**Purpose:** Primary geographic resolver using static world_locations.json database.

**Coverage:**
- 50+ countries mapped with ISO codes
- 250+ states/provinces with regional data
- 170+ districts with exact latitude/longitude

**Key Functions:**

| Function | Description |
|----------|-------------|
| `resolve_location(country, state, district)` | Returns `(lat, lon, state_code)` from static DB |
| `get_countries()` | Returns sorted list of all supported countries |
| `get_states(country)` | Returns states for a given country |
| `get_districts(country, state)` | Returns districts for a given state |

**Fallback Logic:**
1. Exact district match → return district lat/lon
2. State capital match → return state capital coordinates
3. Country centroid → return country-level coordinates
4. None found → delegate to **LLM Location Agent**

**Data File:** `data/reference/world_locations.json`

```json
{
  "india": {
    "states": {
      "maharashtra": {
        "lat": 19.7515, "lon": 75.7139,
        "districts": {
          "pune": {"lat": 18.5204, "lon": 73.8567},
          "nashik": {"lat": 19.9975, "lon": 73.7898}
        }
      }
    }
  }
}
```

---

### 3.2 LLM Location Agent (`llm_location_agent.py`)

**Purpose:** Geocodes any global location (including rural, unmapped districts) using LLM inference.

**When It's Called:** Triggered by `location_agent.py` when a district is not found in `world_locations.json`.

**Strategy:**
1. **Gemini** (preferred) — model fallback chain: `gemini-2.5-flash-lite → gemini-2.0-flash-lite → ...`
2. **Ollama** (local fallback) — `llama3.2` or configured model
3. **Geographic estimate** — latitude-based zone estimate as final fallback

**Key Functions:**

| Function | Description |
|----------|-------------|
| `resolve_location_llm(country, state, district)` | Returns `(lat, lon, climate_zone, crop_notes)` via LLM |
| `_call_gemini_location(prompt)` | Tries all 4 Gemini keys with model fallback |
| `_call_ollama_location(prompt)` | Plain Ollama call for geocoding |

**Prompt Strategy:** Asks the LLM to return JSON with `{"lat": float, "lon": float, "climate_zone": str, "crop_notes": str, "region_type": str}`.

**Output Fields:**
- `lat`, `lon` — Decimal coordinates
- `climate_zone` — Tropical/Subtropical/Arid/Temperate/Mediterranean/Continental
- `crop_notes` — Brief note on what crops are grown there
- `region_type` — Rural/Urban/Agricultural

---

### 3.3 Data Gathering Agent (`data_gathering_agent.py`)

**Purpose:** Assembles all real-world data needed for crop recommendations: live weather, 6-month forecast, ENSO adjustment, soil data, market prices.

**Size:** 845 lines — the largest agent module.

#### Weather Fetching

**Source:** Open-Meteo API (free, no key)  
**URL:** `https://api.open-meteo.com/v1/forecast`  
**Cache:** 30-minute in-memory per `(lat, lon)` rounded to 2 decimal places  

**Data fetched per request:**
```
daily: temperature_2m_max, temperature_2m_min, precipitation_sum,
       windspeed_10m_max, uv_index_max, relativehumidity_2m_mean
past_days: 7, forecast_days: 1
```

**Output fields:**
| Field | Description |
|-------|-------------|
| `temperature_c` | Average of max+min for latest day |
| `temp_max_c` | Maximum temperature |
| `temp_min_c` | Minimum temperature |
| `humidity_pct` | Mean relative humidity |
| `rainfall_7d_mm` | Sum of last 7 days' precipitation |
| `wind_kmh` | Max wind speed |
| `uv_index` | UV index (capped at 11) |
| `feels_like_c` | Heat index approximation |
| `soil_temp_c` | Estimated soil temp (temp_avg - 2°C) |

#### 6-Month Forecast Generation

**Strategy:** Zone-based climatology anchored to live temperature.

**Steps:**
1. Determine climate zone for location (India: `history.py`; World: `_COUNTRY_TO_ZONE` table)
2. Load monthly averages for that zone (`_ZONE_CLIMATE` table)
3. Compute live temperature offset: `offset = live_temp - zone_temp_for_current_month`
4. Apply offset to all 7 forecast months: `adjusted_temp = zone_temp + offset`

**Climate Zones in `_ZONE_CLIMATE`:**
```
Tropical, Subtropical, Arid, Mediterranean, Temperate, Continental,
Temperate_Americas, Tropical_Americas, Subtropical_S, Arid_Oceania
```

**Country → Zone Mapping (`_COUNTRY_TO_ZONE`):**
50+ countries mapped. Latitude-based fallback for unmapped countries:
- `|lat| < 15°` → Tropical
- `|lat| < 30°` → Subtropical  
- `|lat| < 45°` → Temperate
- `|lat| < 60°` → Continental

#### ENSO Adjustment (Step 3 of pipeline)

After building the baseline 6-month forecast, the agent calls `get_climate_signals()` from `climate_signals.py` and then `apply_enso_to_forecast()` to adjust all monthly values.

#### LLM Enrichment (Search-Grounded)

**Priority:**
1. **Gemini + Google Search Grounding** — Real current market prices and active advisories
2. **Gemini plain** — Static LLM knowledge about soil and prices
3. **Zone defaults** — `_ZONE_SOIL_DEFAULTS` + `_ZONE_MARKET_TEMPLATES` tables

**What LLM returns:**
```json
{
  "soil": {"type": "Clay-Loam", "ph": 6.5, "organic_matter": "Medium", "drainage": "Good"},
  "market_prices": {"Wheat": "₹2,400/quintal", "Rice": "₹3,200/quintal"},
  "district_summary": "Pune is a major agricultural district...",
  "climate_zone": "Subtropical"
}
```

#### Currency-Aware Market Prices

`_COUNTRY_CURRENCY` maps 30+ countries to their currency symbols (₹, $, €, £, R$, ¥, etc.).  
`_ZONE_MARKET_TEMPLATES` provides zone-default price templates with `{cur}` placeholders substituted at runtime.

#### Main Entry Point

```python
def gather_location_data(
    country: str, state: str, district: str,
    lat: float, lon: float,
    month: Optional[int] = None,
    state_code: Optional[str] = None,
    llm_climate_zone: Optional[str] = None,
    llm_crop_notes: Optional[str] = None,
    location_source: str = "llm",
) -> dict:
```

**Returns:**
```python
{
    "current": {...},                    # Live weather
    "forecast_6month": [...],            # ENSO-adjusted 7-month forecast
    "forecast_6month_baseline": [...],   # Pre-ENSO baseline
    "soil": {...},                       # LLM or zone-default soil
    "season": "Kharif",                 # Detected season
    "climate_zone": "Subtropical",      # Zone label
    "market_prices": {...},              # LLM or currency-aware defaults
    "district_summary": "...",          # LLM 1-sentence description
    "climate_signal": {...},             # Full climate_signals output
    "location_source": "database",      # "database" or "llm"
}
```

---

### 3.4 Climate Signal Intelligence Service (`climate_signals.py`)

**Purpose:** Comprehensive 9-dimensional climate threat assessment for any global location.

**Size:** 778 lines — covers all climate threats affecting agriculture.

**Data Sources:**
- NOAA CPC ONI text file (free, fetched every 6 hours)
- Live weather passed from Data Gathering Agent
- Gemini Search Grounding for real-time regional advisories

#### 9 Threat Dimensions

**1. ENSO (El Niño/La Niña)**  
Source: NOAA CPC `oni.ascii.txt`  
Detection: `_oni_to_phase_and_strength(oni: float)`  
Phases: `El Nino` (ONI ≥ 0.5), `El Nino Watch` (0.3–0.5), `Neutral`, `La Nina Watch` (−0.3 to −0.5), `La Nina` (≤ −0.5)  
Strengths: Weak / Moderate / Strong / Developing  

**2. Heat Stress**  
Function: `_assess_heat_stress(current_temp, climate_zone)`  
Zone-specific thresholds: Temperate (30/35/40°C), Tropical (36/41/45°C), Arid (38/43/47°C)  
Levels: Moderate / Severe / Extreme  

**3. Drought Index**  
Function: `_assess_drought(rainfall_7d, climate_zone)`  
Compares 7-day rainfall to zone norm. Deficit >80% → Severe; >50% → Moderate  
Also detects Excess rainfall (>2.5× zone norm → waterlogging alert)  

**4. Frost Risk**  
Function: `_assess_frost(current_temp, climate_zone)`  
Near-Frost: ≤4°C, Frost: ≤0°C  

**5. Flood / Excess Rainfall**  
Detected inside drought assessor when `rainfall_7d > norm × 2.5`  

**6. Cyclone/Typhoon/Hurricane Basin**  
Function: `_get_cyclone_context(location_str, country)`  
7 basins tracked: North Atlantic (Hurricane), Eastern Pacific (Hurricane), Western Pacific (Typhoon), North Indian Bay of Bengal (Cyclone), North Indian Arabian Sea (Cyclone), South Indian Ocean (Cyclone), South Pacific (Cyclone/Typhoon)  
Active season detection per basin.  

**7. Wildfire Risk**  
Function: `_assess_wildfire(current_temp, rainfall_7d, climate_zone)`  
High-risk zones: Mediterranean, Arid, Temperate_Americas, Subtropical_S, Arid_Oceania  
Triggered when: `temp ≥ 35°C AND rainfall_7d < 5mm` (High) or `temp ≥ 30°C AND rainfall_7d < 10mm` (Moderate)  

**8. Soil Moisture Stress**  
Inferred from drought/excess rainfall assessors.  

**9. Climate Change Trend**  
Gemini Search Grounding retrieves current regional climate advisories and long-term trend data.  

#### ENSO Zone Impact Table (`_ENSO_IMPACTS`)

```python
_ENSO_IMPACTS = {
    "El Nino": {
        "India":          (-0.20, +0.5),   # rainfall factor, temp offset
        "Subtropical":    (-0.15, +0.4),
        "Tropical":       (-0.10, +0.3),
        "China":          (+0.25, +0.8),   # China GETS more rain in El Nino
        "Southeast_Asia": (-0.10, +0.5),
        "Arid_Oceania":   (-0.20, +1.0),   # Severe drought in Australia
        ...
    },
    "La Nina": { ... },
    "El Nino Watch": { ... },  # ~50% of El Nino impact
    "La Nina Watch": { ... },
    "Neutral": {},             # No adjustments
}
```

#### Country → ENSO Zone Key Mapping (`_COUNTRY_TO_ZONE_KEY`)

```python
{
    "china":       "South_China",
    "japan":       "East_Asia",
    "south korea": "East_Asia",
    "thailand":    "Southeast_Asia",
    "indonesia":   "Southeast_Asia",
    "india":       "India",
    "australia":   "Arid_Oceania",
}
```

**South China Regions:** Guangdong, Guangxi, Hainan, Fujian, Hong Kong, Macau → automatically mapped to `South_China` zone key regardless of country name.

#### Gemini Search-Grounded Climate Analysis

The function `_gemini_comprehensive_climate()` sends a structured prompt requesting:
1. Current drought index or rainfall anomaly
2. Active heat waves, cold snaps, extreme weather
3. Government crop advisories or agricultural warnings
4. Climate change trends affecting agriculture
5. Active pest/disease outbreaks linked to climate

Returns JSON with: `summary`, `enso_impact`, `heat_stress_risk`, `drought_risk`, `flood_risk`, `frost_risk`, `cyclone_risk`, `wildfire_risk`, `climate_change_trend`, `crop_risks[]`, `immediate_actions[]`, `seasonal_outlook`, `alert_level`, `rainfall_outlook`, `temp_outlook`.

#### Caching

- **Cache key:** `(zone_key, country_lc, round(current_temp), round(rainfall_7d))`
- **TTL:** 6 hours — ENSO data updates monthly, but Search Grounding is real-time

#### Forecast Adjustment

```python
def apply_enso_to_forecast(forecast_6month: list, climate_signals: dict) -> list:
    # Applies rainfall_factor and temp_offset_c to all 7 forecast months
    m["rainfall_mm"] = m["rainfall_mm"] * rainfall_factor
    m["temp_avg"]    = m["temp_avg"] + temp_offset_c
    m["temp_max"]    = m["temp_max"] + temp_offset_c
    m["temp_min"]    = m["temp_min"] + temp_offset_c
    m["soil_temp_c"] = m["soil_temp_c"] + temp_offset_c
    m["enso_adjusted"] = True
```

---

### 3.5 Crop Agent (`crop_agent.py`)

**Purpose:** AI-powered crop ranking for any global location.

**Size:** 794 lines — the most complex agent module.

#### Pipeline (in priority order)

| Priority | Method | Description |
|----------|--------|-------------|
| 1 | **In-memory cache** | Returns instantly if same `(country, state, district, season, climate, irrigation)` was requested within 1 hour |
| 2 | **Gemini + Google Search Grounding** | Real-time crop advisories, current pest alerts, live market info |
| 3 | **Gemini plain** (4-key rotation, model chain) | Static LLM knowledge, full prompt with country hints |
| 4 | **Ollama + Web Search Tool** | Local LLM with DuckDuckGo tool-calling |
| 5 | **Ollama plain** | Local LLM, no search |
| 6 | **Geography-aware zone fallback** | Instant, no API, returns hardcoded but geographically accurate crops |

#### Country Crop Hints (`_COUNTRY_CROP_HINTS`)

50+ countries with local-language crop names to guide the LLM:

```python
"germany": "Winterweizen (Winter Wheat), Winterraps (Canola), Zuckerrübe (Sugar Beet), Mais (Maize), Kartoffel (Potato)...",
"brazil":  "Soja (Soybean), Milho (Maize), Cana-de-açúcar (Sugarcane), Café (Coffee)...",
"india":   "Chawal/Dhan (Rice), Gehun (Wheat), Kapas (Cotton), Ganna (Sugarcane)...",
"nigeria": "Rice, Maize, Sorghum, Cassava, Yam, Cowpea, Groundnut...",
...
```

Crops included from: Western Europe, Northern Europe, Central/Eastern Europe, Russia/Central Asia, North America, Central/South America, Middle East/North Africa, South Asia, East/Southeast Asia, Africa, Oceania.

#### Geographic Validation (`_validate_crops`)

Detects when Gemini has hallucinated wrong geography (e.g., returns Hindi crop names for Germany):

```python
_HINDI_CROP_NAMES = {"chawal", "gehun", "makka", "chana", "sarson", "bajra", "jowar", ...}
_SOUTH_ASIAN_COUNTRIES = {"india", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan"}

# If ≥2 crops have Hindi local names for a non-South-Asian country → reject, use zone fallback
```

#### Prompt Structure

The LLM prompt (`_build_prompt()`) includes:
- Country, state, district, season, climate zone, hemisphere (Northern/Southern)
- Country-specific crop hints embedded directly
- Live weather: temperature, humidity, 7-day rainfall
- 3-month forecast summary
- Soil type, pH, organic matter, drainage
- Local market prices
- District summary (1 sentence)
- Irrigation level and planning days

**Requested output:** JSON array of 6 crops with fields:
`crop_name`, `local_name`, `suitability_score` (0-100), `season_fit`, `risk_level`, `duration_days`, `water_need`, `estimated_yield`, `planting_window`, `market_demand`, `reasons[]`, `warnings[]`, `growing_tip`

#### Zone Fallback Tables (`_fallback_crops`)

Geography-aware fallback crop tables indexed by climate zone and hemisphere (never India-only names):
- **Tropical** — Rice, Maize, Cassava, Sweet Potato, Plantain, Groundnut
- **Arid/Semi-Arid** — Sorghum, Millet, Sesame, Cowpea, Dates, Cotton
- **Mediterranean** — Wheat, Olive, Grape, Tomato, Sunflower, Barley
- **Temperate** — European-specific (Sugar Beet, Winter Wheat, Canola, Potato) vs Americas (Corn, Soybean, Cotton)
- **Continental** — Wheat, Sunflower, Soybean, Sugar Beet, Canola
- **Subtropical (South Asia)** — Kharif/Rabi seasonal tables

#### Caching

- **Cache key:** `(country, state, district, season, climate, irrigation)`
- **TTL:** 1 hour (configurable via `_CROP_CACHE_TTL`)

---

### 3.6 Web Search Agent (`web_search_agent.py`)

**Purpose:** Enables Ollama to perform web searches via DuckDuckGo tool-calling.

**When Used:** Called by `crop_agent.py` (step 4 of pipeline) and `data_gathering_agent.py` when Gemini is not configured.

**Mechanism:**
1. Defines a `web_search` tool schema for Ollama's tool-calling API
2. Sends the prompt with available tools to Ollama
3. When Ollama calls the tool, executes DuckDuckGo search
4. Returns search results back to Ollama for final response

**Key Function:**
```python
def call_ollama_with_search(prompt: str, location: str = "", timeout: int = 45) -> Optional[str]:
```

---

## 4. Service Modules

**Location:** `agri_crop_recommendation/src/services/`

### 4.1 LLM Farmer Chat (`llm_chat.py`)

**Purpose:** Context-aware AI farming Q&A with real-time token streaming.

**Features:**
- **Context injection:** Farm-specific data (location, weather, ENSO phase, crop recommendations) is prepended to every LLM prompt
- **SSE streaming:** Tokens streamed character-by-character via FastAPI `StreamingResponse`
- **Multi-provider:** Tries Gemini (with key rotation) first, then Ollama

**Input context fields:**
```python
{
    "country": str, "state": str, "district": str,
    "weather": dict,           # live weather
    "soil": dict,              # soil data
    "enso_phase": str,         # El Nino/La Nina/Neutral
    "crop_recommendations": list,  # top crops from crop agent
    "season": str,
    "climate_zone": str,
}
```

### 4.2 LLM Explainer (`llm_explainer.py`)

**Purpose:** Generates short, farmer-friendly natural language explanations for why each top crop was recommended.

**Output includes:**
- Why this crop suits the specific climate and soil
- Key risk warnings (optionally multilingual)
- Best practices for the region

### 4.3 LLM Filter (`llm_filter.py`)

**Purpose:** Pre-filters the crop knowledge database to candidate crops before passing to the Crop Agent, reducing prompt length and improving LLM accuracy.

### 4.4 Recommender Engine (`recommender.py`)

**Purpose:** Legacy v2.x rule-based crop recommendation engine. Used by the `/recommend` endpoint.

**Scoring Formula:**

| Factor | Weight | Computation |
|--------|--------|-------------|
| Temperature Compatibility | 25% | Optimal range = 100; decay curve to survival limits; hard cutoff beyond limits |
| Water Availability | 25% | Expected rainfall + irrigation vs. crop water requirement; ENSO-adjusted |
| Soil Compatibility | 15% | Texture, pH, drainage, organic matter matching |
| Regional Suitability | 15% | District-specific modifiers from regions.json |
| Seasonal Adjustment | 10% | Traditional planting season match |
| Drought Tolerance Bonus | 10% | Score boost during El Niño dry spells |

**ML Blend:** If Random Forest model is loaded, final score = `0.6 × ML_score + 0.4 × rule_score`.

### 4.5 Risk Assessment Engine (`risk.py`)

**Purpose:** Rule-based drought and temperature stress assessment. Still used in legacy `/recommend` pipeline.

**Checks:**
- Soil moisture deficit
- Temperature range stress for each crop
- Combined drought + heat stress scenarios

### 4.6 Pest & Disease Warning System (`pests.py`)

**Purpose:** Rule-based pest and disease alerts based on weather thresholds.

**Examples:**
- High humidity + high temperature → Fungal disease pressure
- Low rainfall + hot temperatures → Aphid/mite risk
- Post-rain warm periods → Late blight in potatoes

### 4.7 Planting Calendar (`calendar.py`)

**Purpose:** Generates rule-based growth phase timelines for recommended crops.

**Phases:** Soil Preparation → Sowing → Germination → Vegetative → Flowering → Maturity → Harvest

---

## 5. API Layer

**Location:** `agri_crop_recommendation/src/api/`

### 5.1 Request Models (`models.py`)

**`AnalyzeRequest`** (POST `/api/analyze/stream`):
```python
class AnalyzeRequest(BaseModel):
    country:     str
    state:       str
    district:    str
    irrigation:  str = "Limited"          # None / Limited / Full
    planning_days: int = 90               # 30 / 60 / 90 / 180
    soil_type:   Optional[str] = None     # Clay / Loam / Sandy / etc.
    soil_ph:     Optional[float] = None   # 4.5 – 8.5
    month:       Optional[int] = None     # 1–12 (defaults to current month)
```

**`ChatRequest`** (POST `/chat` and `/chat/stream`):
```python
class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None  # weather, soil, crops, ENSO, etc.
    history: Optional[list] = None  # previous chat turns
```

### 5.2 All Endpoints (`app.py`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | None | Serves the HTML dashboard |
| `GET` | `/api/countries` | None | Returns sorted country list |
| `GET` | `/api/states/{country}` | None | Returns states for a country |
| `GET` | `/api/districts/{country}/{state}` | None | Returns districts for a state |
| `POST` | `/api/analyze/stream` | None | **Main SSE streaming analysis** |
| `GET` | `/climate-signals` | None | 9-dimensional climate assessment |
| `POST` | `/chat` | None | Single-turn farmer chat |
| `POST` | `/chat/stream` | None | Streaming farmer chat (SSE) |
| `GET` | `/weather/now/{region_id}` | None | Live temperature from Open-Meteo |
| `POST` | `/recommend` | None | Legacy batch JSON recommendation |
| `GET` | `/health` | None | System health and provider info |
| `GET` | `/docs` | None | Auto-generated Swagger UI |
| `GET` | `/redoc` | None | ReDoc API documentation |

### 5.3 Streaming Response Protocol

The `/api/analyze/stream` endpoint uses **Server-Sent Events (SSE)** to stream progress and results:

**SSE Event Format:**
```
data: {"type": "progress", "step": "weather", "message": "Fetching live weather..."}

data: {"type": "weather", "data": {"temperature_c": 32.4, "humidity_pct": 68, ...}}

data: {"type": "climate_signals", "data": {"enso_phase": "Neutral", "threats": {...}, ...}}

data: {"type": "soil", "data": {"type": "Clay-Loam", "ph": 6.5, ...}}

data: {"type": "market", "data": {"Wheat": "₹2,400/quintal", ...}}

data: {"type": "forecast", "data": [{"month": "Jul 2026", "temp_avg": 28, ...}, ...]}

data: {"type": "crops", "data": [{"crop_name": "Rice", "suitability_score": 88, ...}, ...]}

data: {"type": "summary", "data": "Based on the current El Niño conditions..."}

data: {"type": "done"}
```

**Client-side handling** (`app.js`):
```javascript
const evtSource = new EventSource('/api/analyze/stream', {method: 'POST', body: ...});
evtSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'crops') renderCropCards(payload.data);
    if (payload.type === 'climate_signals') renderClimatePanel(payload.data);
    // etc.
};
```

---

## 6. Frontend & UI Engine

**Location:** `agri_crop_recommendation/templates/index.html`, `static/js/app.js`, `static/css/style.css`

### 6.1 Dashboard Structure

```
┌─────────────────────────────────────────────────────┐
│  🌾 AI Crop Advisor     [Country] [State] [District] │
│  Irrigation: [None/Limited/Full]  Days: [30/60/90]   │
│  Soil: [type] pH: [value]   [Analyze with AI Agent]  │
├─────────────────────────────────────────────────────┤
│  Progress Strip (SSE progress events)                │
├─────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ 🌡️ Weather   │  │ 🏔️ Soil      │                 │
│  │ Temp, Rain   │  │ Type, pH     │                 │
│  │ Wind, UV     │  │ Drainage     │                 │
│  └──────────────┘  └──────────────┘                 │
├─────────────────────────────────────────────────────┤
│  🌐 Climate Intelligence Panel                       │
│  ENSO Badge + ONI | Threat Cards | Alert Level       │
│  AI Summary | Rainfall/Temp Outlook | Actions        │
├─────────────────────────────────────────────────────┤
│  📈 6-Month Forecast Chart (Chart.js — dual axis)    │
│  Temperature curve + Rainfall bars (ENSO-adjusted)   │
├─────────────────────────────────────────────────────┤
│  💰 Market Prices Card                               │
├─────────────────────────────────────────────────────┤
│  🌱 Crop Rankings (6 cards, sorted by suitability)   │
│  Each card: Score, Risk, Season Fit, Duration,       │
│  Water Need, Yield, Planting Window, Reasons         │
├─────────────────────────────────────────────────────┤
│  💬 AI Farmer Chat (SSE streaming Q&A)               │
└─────────────────────────────────────────────────────┘
```

### 6.2 SSE Client (`app.js`)

**Key Functions:**

| Function | Description |
|----------|-------------|
| `loadCountries()` | Populates country dropdown on page load |
| `loadStates(country)` | Cascading state dropdown |
| `loadDistricts(country, state)` | Cascading district dropdown |
| `startAnalysis()` | Initiates SSE stream on form submit |
| `renderWeatherCard(data)` | Builds live weather display |
| `renderClimatePanel(data)` | Builds 9-threat Climate Intelligence Panel |
| `renderForecastChart(data)` | Renders Chart.js dual-axis chart |
| `renderCropCards(crops)` | Generates suitability score cards |
| `sendChatMessage()` | Initiates SSE chat stream |

### 6.3 Climate Intelligence Panel (`renderClimatePanel`)

The Climate Intelligence Panel is rendered dynamically from the `climate_signals` SSE event:

```javascript
function renderClimatePanel(signals) {
    // ENSO phase badge with color coding
    // ONI value display
    // Threat badges: heat, drought, frost, wildfire, cyclone
    // Alert level banner (None/Advisory/Watch/Warning/Emergency)
    // AI summary text (3 sentences from Gemini)
    // Rainfall outlook badge
    // Temp outlook badge
    // Crop risks list
    // Immediate actions list
    // Forecast adjustment description
    // Data freshness and source attribution
}
```

---

## 7. Data Layer

### 7.1 World Locations (`data/reference/world_locations.json`)

**Coverage:** 50+ countries, 250+ states, 170+ districts  
**Format:** Hierarchical JSON with lat/lon, ISO codes, and metadata  
**Usage:** Primary resolver in `location_agent.py`

### 7.2 Crop Knowledge Database (`data/reference/crop_knowledge.json`)

**Purpose:** Core LLM context for crop filtering (`llm_filter.py`) and fallback rules  
**Contents:** 50+ crops with:
- Temperature ranges (optimal, minimum, maximum)
- Water requirements
- Suitable soil types and pH ranges
- Growing seasons
- Duration (days)
- Regional suitability modifiers
- Known pests and diseases

### 7.3 Climate Zone Tables (`data_gathering_agent.py` inline)

Monthly climate data for each zone — 12 months × 5 parameters:

```python
_ZONE_CLIMATE = {
    "Tropical": {
        1: {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 45, "hum": 82},
        ...
    },
    "Temperate": {...},
    "Continental": {...},
    ...
}
```

### 7.4 Market Price Templates (`data_gathering_agent.py` inline)

Currency-aware templates for 6+ zones. `{cur}` is substituted with the country's currency symbol at runtime.

### 7.5 Legacy Data (v2.x)

| File | Description |
|------|-------------|
| `data/reference/regions.json` | India 640-district database with state code mapping |
| `data/reference/regional_crops.json` | India regional crop catalog |
| `src/weather/history.py` | India zone-based historical climate data (monthly averages) |

---

## 8. Machine Learning Pipeline

### 8.1 Random Forest Suitability Model

**File:** `src/ml/predictor.py`  
**Library:** scikit-learn 1.8.0  
**Type:** Random Forest Regressor  
**Training:** `scripts/train_model.py` on simulated crop suitability data  
**Saved to:** `models/crop_suitability/`

**Features used:**
- Temperature (avg, max, min)
- Rainfall (monthly)
- Humidity
- Soil type (encoded)
- Soil pH
- Season (encoded)
- Crop type (encoded)

**Score blending (when loaded):**
```
final_score = 0.6 × ML_score + 0.4 × rule_based_score
```

### 8.2 LSTM Weather Forecasting (Legacy)

**File:** `src/ml/lstm_weather.py`  
**Library:** PyTorch 2.9.0  
**Purpose:** Medium-range weather forecasting from historical Parquet data  
**Status:** Superseded by Open-Meteo live data + zone climatology in v3.0+  
**Saved to:** `models/weather_lstm/`

### 8.3 XGBoost Weather Model (Legacy)

**Library:** XGBoost 3.2.0  
**Purpose:** Alternative weather forecasting  
**Status:** Maintained for backward compatibility  
**Saved to:** `models/weather_xgboost/`

---

## 9. LLM Integration Details

### 9.1 Provider Priority Chain

All 3 LLM-using agents follow the same priority chain:

```
1. Gemini + Google Search Grounding (search-capable models)
   → Real-time data from the live web
2. Gemini plain (all models, 4-key rotation)
   → Static LLM knowledge
3. Ollama + DuckDuckGo Web Search Tool
   → Local model with search capability
4. Ollama plain
   → Local model, fully offline
5. Zone-based rule defaults
   → No API, instant, geographically aware
```

### 9.2 Gemini Key Rotation

```python
GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]

# On 429 RESOURCE_EXHAUSTED: rotate to next key
for api_key in GEMINI_KEYS:
    for model in _GEMINI_MODELS:
        try: ...
        except Exception as e:
            if "429" in str(e): continue  # try next key/model
```

### 9.3 Google Search Grounding

Enabled via `google-genai` SDK:

```python
from google.genai import types as _gt

resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=_gt.GenerateContentConfig(
        tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
    ),
)
```

**Supported models:** `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.0-flash-001`

### 9.4 Model Fallback Chain

```python
_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",      # Fastest, most cost-efficient
    "gemini-2.0-flash-lite",      # Good balance
    "gemini-2.0-flash-lite-001",  # Stable snapshot
    "gemini-flash-lite-latest",   # Latest lite
    "gemini-2.0-flash",           # Full capability
    "gemini-2.5-flash",           # Most capable
]

_SEARCH_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
]
```

### 9.5 In-Memory Caching

| Cache | Key Fields | TTL |
|-------|-----------|-----|
| Weather | `(lat, lon)` rounded to 2dp | 30 minutes |
| Climate Signals | `(zone, country, temp, rainfall)` | 6 hours |
| Crop Recommendations | `(country, state, district, season, climate, irrigation)` | 1 hour |

---

## 10. Climate Signal Intelligence — Deep Dive

_(See Section 3.4 for detailed coverage. Key formulas repeated here for reference.)_

### 10.4 Forecast Adjustment Formula

```
adjusted_rainfall_mm[month] = baseline_rainfall_mm[month] × (1 + rain_factor)
adjusted_temp_avg[month]    = baseline_temp_avg[month] + temp_offset_c

Examples:
  El Niño in India:          rain_factor=-0.20, temp_offset=+0.5
  La Niña in India:          rain_factor=+0.20, temp_offset=-0.3
  El Niño in South China:    rain_factor=+0.25, temp_offset=+0.8  (typhoon risk)
  El Niño in Arid Oceania:   rain_factor=-0.20, temp_offset=+1.0  (severe drought)
```

---

## 11. Crop Agent — Deep Dive

_(See Section 3.5 for detailed coverage.)_

### 11.3 Fallback Tables by Climate Zone

When all LLM calls fail, crops are returned from hardcoded zone-specific tables:

| Zone | Crops Returned |
|------|---------------|
| Tropical | Rice, Maize, Cassava, Sweet Potato, Plantain, Groundnut |
| Arid/Semi-Arid | Sorghum, Millet, Sesame, Cowpea, Dates, Cotton |
| Mediterranean | Wheat, Olive, Grape, Tomato, Sunflower, Barley |
| Temperate (Europe) | Sugar Beet, Winter Wheat, Canola, Potato, Barley (season-dependent) |
| Temperate (Americas) | Corn, Soybean, Winter Wheat, Cotton, Sorghum, Sunflower |
| Continental | Wheat, Sunflower, Soybean, Sugar Beet, Canola, Barley |

---

## 12. Data Flow Diagram (v3.1)

```
FARMER INPUT (Web Form: Country, State, District, Irrigation, Soil, Days)
        │
        ▼
POST /api/analyze/stream (FastAPI SSE endpoint)
        │
        ├──[1]── Location Agent (location_agent.py)
        │        → world_locations.json lookup
        │        → If not found: LLM Location Agent (llm_location_agent.py)
        │        → Returns: lat, lon, state_code
        │
        ├──[2]── Data Gathering Agent (data_gathering_agent.py)
        │        ├─ Open-Meteo API → current weather (30-min cache)
        │        ├─ Zone climatology → 6-month baseline forecast
        │        ├─ Climate Signals (climate_signals.py)
        │        │   ├─ NOAA CPC ONI → ENSO phase
        │        │   ├─ 8 local threat assessors (heat, drought, frost, etc.)
        │        │   └─ Gemini Search Grounding → regional advisories
        │        ├─ apply_enso_to_forecast() → adjusted 6-month
        │        └─ LLM Enrichment (Gemini Search → Gemini → zone defaults)
        │            → soil type, market prices, district summary
        │
        ├──[3]── Crop Agent (crop_agent.py)
        │        ├─ Cache check
        │        ├─ Gemini + Search Grounding → real-time crop advisories
        │        ├─ Gemini plain (4 keys × 6 models)
        │        ├─ Ollama + DuckDuckGo Web Search (web_search_agent.py)
        │        ├─ Ollama plain
        │        └─ Geography-aware zone fallback
        │
        └──[4]── Stream Output (FastAPI StreamingResponse)
                 │
                 ▼ SSE events:
                 weather → climate_signals → soil → market →
                 forecast → crops → summary → done
                 │
                 ▼
        REAL-TIME STREAMING DASHBOARD (app.js)
        Renders: Weather Card, Climate Panel, Forecast Chart,
                 Market Card, Crop Rankings
```

---

## 13. Global API Reference

### Location APIs

```http
GET /api/countries
→ ["Australia", "Brazil", "Canada", "China", "Germany", "India", ...]

GET /api/states/india
→ ["Andhra Pradesh", "Bihar", "Gujarat", "Maharashtra", "Punjab", ...]

GET /api/districts/india/maharashtra
→ ["Ahmednagar", "Aurangabad", "Nagpur", "Nashik", "Pune", "Solapur", ...]
```

### Core Analysis API

```http
POST /api/analyze/stream
Content-Type: application/json

{
  "country": "India",
  "state": "Maharashtra",
  "district": "Pune",
  "irrigation": "Limited",
  "planning_days": 90,
  "soil_type": "Loam",
  "soil_ph": 6.8,
  "month": 6
}

→ SSE stream (see Section 5.3 for event format)
```

### Climate Signals API

```http
GET /climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical

→ {
    "climate_signals": {
      "enso_phase": "Neutral",
      "enso_strength": "Neutral",
      "oni_value": 0.12,
      "phase_label": "Neutral (Normal)",
      "threats": {
        "heat_stress": null,
        "drought": null,
        "frost": null,
        "wildfire": null,
        "cyclone": {"storm_type": "Cyclone", "in_active_season": false, ...}
      },
      "ai_interpretation": {
        "summary": "...",
        "crop_risks": ["...", "...", "..."],
        "immediate_actions": ["...", "..."],
        "alert_level": "None",
        "rainfall_outlook": "Near Normal",
        "temp_outlook": "Near Normal"
      },
      "forecast_adjustments": {
        "rainfall_factor": 1.0,
        "temp_offset_c": 0.0,
        "description": "No ENSO adjustment (Neutral conditions)"
      },
      "fetched_at": "2026-06-16T13:40:00",
      "source": "NOAA CPC + Gemini Search Grounding + Live Weather",
      "data_freshness": "ENSO: monthly; live threats: real-time"
    }
  }
```

### Farmer Chat API

```http
POST /chat/stream
Content-Type: application/json

{
  "message": "How does the current El Niño affect my wheat crop?",
  "context": {
    "country": "India",
    "state": "Maharashtra",
    "district": "Pune",
    "enso_phase": "El Nino",
    "weather": {...},
    "crop_recommendations": [...]
  }
}

→ SSE stream of LLM tokens (text/event-stream)
```

### Health Check

```http
GET /health

→ {
    "status": "healthy",
    "version": "3.1",
    "regions_loaded": true,
    "ml_models": {"crop_suitability": true, "weather_lstm": false, "xgboost": false},
    "llm_available": true,
    "llm_provider": "ollama",
    "ollama_running": true,
    "ollama_model": "llama3.2",
    "gemini_keys": 2,
    "timestamp": "2026-06-16T13:40:00"
  }
```

---

## 14. Technology Stack & Dependencies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Backend Framework** | FastAPI | 0.128.7 | High-performance async API + SSE streaming |
| **ASGI Server** | Uvicorn | 0.40.0 | Production-grade ASGI server |
| **Primary Local LLM** | Ollama | 0.6.2 | LLaMA 3.2 / Gemma local inference |
| **Cloud LLM** | Google Gemini | via google-genai 1.69 | Search Grounding + cloud fallback |
| **HTTP Client** | requests | 2.32.5 | Open-Meteo + NOAA API calls |
| **ML — Suitability** | scikit-learn | 1.8.0 | Random Forest crop regressor |
| **ML — Weather (LSTM)** | PyTorch | 2.9.0 | LSTM sequence model |
| **ML — Weather (XGBoost)** | XGBoost | 3.2.0 | Gradient boosting weather model |
| **ML — Jobs** | joblib | 1.5.3 | Model serialization |
| **Data Processing** | pandas | 2.3.3 | DataFrame operations |
| **Data Processing** | numpy | 2.3.5 | Numerical computing |
| **Parquet I/O** | pyarrow | 22.0.0 | Legacy Parquet data files |
| **Visualization** | matplotlib | 3.10.8 | Backend charts (legacy) |
| **Visualization** | seaborn | 0.13.2 | Statistical visualization (legacy) |
| **HTML Templating** | Jinja2 | 3.1.6 | Server-side HTML rendering |
| **Multipart Forms** | python-multipart | 0.0.22 | File upload handling |
| **Environment** | python-dotenv | 1.2.1 | .env file loading |
| **Frontend Charting** | Chart.js | (CDN) | Temperature/rainfall dual-axis chart |
| **Streaming Protocol** | Server-Sent Events | (native) | Real-time text generation to UI |

---

## 15. System Limitations & Future Scope

### 15.1 Current Limitations

| Limitation | Description |
|-----------|-------------|
| **Weather API Dependency** | Requires internet for Open-Meteo live weather. Graceful fallback to zone estimates exists. |
| **Crop Coverage** | Primarily covers annual crops (15–270 days). Long-cycle perennials (e.g., timber, rubber plantations) have limited detail. |
| **Soil Testing** | Relies on LLM-deduced or zone-default soil profiles. No integration with real IoT soil sensors. |
| **Market Prices** | Search-grounded prices are best-effort. Wholesale commodity exchange APIs not integrated. |
| **District Coverage** | world_locations.json covers ~170 districts explicitly. Rural districts fall back to LLM geocoding. |
| **Language** | UI is English-only. LLM warnings can be requested in other languages via chat. |
| **NOAA Dependency** | ENSO data unavailable if NOAA CPC is unreachable (firewall/outage). Falls back to Neutral. |

### 15.2 Future Scope

| Feature | Description |
|---------|-------------|
| **RAG Farmer Chat** | Index `crop_knowledge.json` into ChromaDB/FAISS vector store for retrieval-augmented, strictly grounded Q&A |
| **Live Market APIs** | AgMarkNet (India), USDA NASS (USA), CME Group commodity prices integration |
| **Farmer Profiles** | Persistent user accounts to track crop cycles, seasonal history, and personalized recommendations |
| **Multi-Language UI** | Hindi, Tamil, Telugu, Marathi, Swahili, Spanish, French interfaces |
| **Mobile App** | React Native or Flutter app with offline-capable rule engine |
| **Satellite NDVI** | NASA/ESA satellite vegetation index integration for crop health monitoring |
| **IoT Sensor Integration** | Real soil NPK, moisture, EC sensor data from Raspberry Pi / Arduino nodes |
| **Blockchain Traceability** | On-chain crop provenance for premium market access |
| **Extended Crop DB** | Add 200+ more crops including perennials, exotic varieties, and region-specific landraces |
| **HPC Training** | Use CDAC PARAM supercomputer to train larger LSTM models on 10+ years of regional data |

---

*End of User Module Paper — AI Powered Weather Resilient Crop Advisor v3.1*  
*Prepared by Tirth Chankeshwara | HPC Group | CDAC-Pune | June 2026*  
*Repository: https://github.com/tirthch25/AI-Powered-Weather-Resilient-Crop-Advisor*
