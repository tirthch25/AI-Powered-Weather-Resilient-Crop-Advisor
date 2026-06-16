# 🌾 AI Powered Weather Resilient Crop Advisor v3.1

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128%2B-009688?style=for-the-badge&logo=fastapi)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2%20Local-8A2BE2?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google)
![NOAA](https://img.shields.io/badge/NOAA-Climate%20Signals-0057B7?style=for-the-badge)
![HTML5](https://img.shields.io/badge/HTML5-Vanilla_JS-E34F26?style=for-the-badge&logo=html5)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-Powered, Weather-Resilient Crop Advisor powered by LLaMA & Gemini Search-Grounded Agents — covering 50+ countries, 250+ states, and 170+ districts, with real-time ENSO / El Niño / La Niña climate intelligence and 9-dimensional climate threat assessment.**

[Overview](#-overview) • [Agent Architecture](#-agent-architecture) • [Climate Intelligence](#-climate-signal-intelligence-v31) • [Features](#-features) • [Installation](#-installation) • [API Endpoints](#-api-endpoints) • [User Module](#-user-module)

</div>

---

## 🧭 Overview

The **AI Powered Weather Resilient Crop Advisor** is an end-to-end, global agricultural intelligence platform. By combining live weather data, machine-learning-blended climate forecasts, real-time ENSO signals, comprehensive climate threat assessment, and large language model (LLM) reasoning with **Google Search Grounding**, the system replicates the advice of a skilled agronomist for virtually any farming region in the world.

A user selects their country, state, and district. The system instantly runs a **5-agent pipeline**:
1. **Location Agent** — Resolves exact latitude/longitude for any global location
2. **Data Gathering Agent** — Fetches live Open-Meteo weather + 6-month forecast + ENSO adjustment
3. **Climate Signal Intelligence** — Comprehensive 9-dimensional climate threat assessment (ENSO, drought, heat, frost, cyclone, wildfire, flood, soil moisture, climate change trend)
4. **Crop Agent** — AI-powered ranking of 50+ crops with Google Search Grounding for real-time advisories
5. **LLM Chat / Explainer** — Context-aware farmer Q&A with streaming SSE responses

The platform uses **LLaMA 3.2 (via Ollama)** locally, with **Google Gemini 2.5 Flash** as an automatic cloud fallback and **Google Search Grounding** for real-time crop advisories and market prices.

---

## 💡 Project Idea

Most crop advisory tools rely on generic, static lookup tables or expensive IoT sensors. This platform takes a radically different approach — acting as an **AI-powered climate-aware agronomist**. It combines **publicly available free data sources** with **Machine Learning**, **LLM Agents + Web Search**, and **real-time climate signals** to replicate expert advice dynamically, anywhere in the world.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Global Precision** | Location Agent dynamically maps 170+ global districts. Open-Meteo fetches real-time, exact lat/lon weather. LLM Location Agent handles unmapped rural locations. |
| **Climate Intelligence** | NOAA CPC ONI data (free) reveals El Niño/La Niña phase; 9 independent climate threat assessors (ENSO, heat, drought, frost, flood, cyclone, wildfire, soil moisture, climate change) run in parallel. |
| **Search-Grounded AI** | Gemini + Google Search Grounding provides real-time crop advisories, current market prices, and active pest/disease alerts for any location. |
| **Agentic Workflow** | Multi-agent system (Location, LLM-Location, Data, Climate, Crop, Web-Search, Chat) orchestrates intelligence behind a single `/api/analyze/stream` call. |
| **Graceful Degradation** | Remove LLM → rule-based scoring continues. Remove Internet → zone-based modeling. Remove NOAA → Neutral phase assumed. Zero single points of failure. |
| **Free & Private** | LLaMA 3.2 runs locally via Ollama. NOAA data is public. Open-Meteo requires no API key. |

---

## 🤖 Agent Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                                   │
│          Web Browser ←→ index.html + app.js + style.css                     │
│   Streaming Dashboard · Climate Intelligence Panel · Farmer Chat             │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │  HTTP / REST / SSE Streaming (FastAPI)
┌─────────────────────────────▼────────────────────────────────────────────────┐
│                           API LAYER                                           │
│  GET /api/countries · GET /api/states · GET /api/districts                   │
│  POST /api/analyze/stream · POST /chat · GET /weather/now                    │
│  GET /climate-signals · POST /chat/stream · GET /health                      │
└──┬──────────────────────────────────┬──────────────────┬─────────────────────┘
   │                                  │                  │
┌──▼──────────────────┐  ┌────────────▼──────────┐  ┌───▼──────────────────────┐
│  Location Agent     │  │  Data Gathering Agent │  │  Climate Signal Service  │
│  world_locations.   │  │  • Open-Meteo live wx │  │  • NOAA ONI (ENSO)       │
│  json               │  │  • 6-month forecast   │  │  • Heat Stress           │
│  50+ Countries      │  │  • ENSO adjustment    │  │  • Drought Index         │
│  250+ States        │  │  • Soil + Market LLM  │  │  • Frost Assessment      │
│  170+ Districts     │  │  • Search grounding   │  │  • Flood Risk            │
│                     │  │                       │  │  • Cyclone Basin         │
│  LLM Location Agent │  │  Web Search Agent     │  │  • Wildfire Risk         │
│  (unmapped rural)   │  │  (DuckDuckGo tool)    │  │  • Gemini Search AI      │
└──┬──────────────────┘  └────────────┬──────────┘  └──────────────────────────┘
   │                                  │
┌──▼──────────────────────────────────▼──────────────────────────────────────┐
│                          CROP AGENT (LLM)                                   │
│  Priority: Cache → Gemini+Search → Gemini Plain → Ollama+Search → Ollama   │
│  Fallback: Geography-aware zone table (instant, no API)                    │
│  Validation: Country-specific crop hints to prevent geographic hallucination│
└──┬─────────────────────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────────────────────────────┐
│                             LLM LAYER                                        │
│  Primary:  LLaMA 3.2 via Ollama (local, free, private)                     │
│  Fallback: Google Gemini (4-key rotation, model fallback chain)             │
│  Search:   Google Search Grounding (real-time advisories + market prices)   │
│  Models:   gemini-2.5-flash-lite → gemini-2.0-flash-lite → gemini-2.0-flash│
└──┬──────────────────────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│  Crop DB (50+ crops) · World Locations · Zone Climate Normals               │
│  Open-Meteo Live API (Free) · NOAA CPC ONI (Free)                          │
│  Legacy: India 640 districts · Parquet climate files · ML Models           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Climate Signal Intelligence (v3.1)

Version 3.1 adds a full **9-Dimensional Climate Intelligence** layer. The system goes far beyond just ENSO — it now assesses every major climate threat that affects agriculture.

### 9 Climate Threat Dimensions

| # | Threat | Data Source | How It Works |
|---|--------|-------------|--------------|
| 1 | **ENSO (El Niño/La Niña)** | NOAA CPC ONI (free) | ONI value → phase detection → zone-specific rainfall/temp adjustment |
| 2 | **Drought Index** | Live Open-Meteo 7-day rainfall | Rainfall deficit % vs zone norm → Moderate/Severe alert |
| 3 | **Heat Stress** | Live temperature | Zone-specific thresholds → Moderate/Severe/Extreme alert |
| 4 | **Frost / Cold Stress** | Live min temperature | Near-frost (<4°C) and frost (<0°C) detection |
| 5 | **Flood / Excess Rainfall** | Live 7-day rainfall | >2.5× zone norm triggers waterlogging alert |
| 6 | **Cyclone / Typhoon / Hurricane** | Basin season calendar | 7 global basins tracked: Atlantic, Pacific, Indian, South Pacific |
| 7 | **Wildfire Risk** | Temperature + drought | High-risk zones (Mediterranean, Arid, Oceania) — hot + dry trigger |
| 8 | **Soil Moisture Stress** | Derived from rainfall deficit/surplus | Inferred from live rainfall vs zone expectations |
| 9 | **Climate Change Trend** | Gemini Search Grounding | Real-time regional advisories from agricultural agencies |

### ENSO Impact Reference

| ENSO Phase | India / South Asia | East Asia / China | Oceania |
|------------|--------------------|--------------------|---------|
| 🔴 **El Niño** | Weaker SW monsoon, −20% rainfall, +0.5°C | Flood risk (typhoon intensification), +25% rainfall, +0.8°C | Severe drought, −20% rainfall, +1.0°C |
| 🔵 **La Niña** | Stronger SW monsoon, +20% rainfall, −0.3°C | Drier conditions, −10% rainfall | Above-normal rainfall, +0.25 factor |
| 🟢 **Neutral** | Near-normal conditions | Standard seasonal | Near-normal |
| 🟡 **Developing Watch** | 50% of full phase impact | 50% of full phase impact | 50% of full phase impact |

### ENSO Adjustment Formula

```text
adjusted_rainfall = baseline_rainfall × (1 + rainfall_factor)
adjusted_temp     = baseline_temp + temp_offset_c

Example — El Niño in India:
  rainfall_factor = -0.20  →  rainfall × 0.80
  temp_offset_c   = +0.50  →  temp + 0.5°C
```

### New `/climate-signals` Endpoint

```http
GET /climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical
```

Returns full 9-dimensional threat assessment:
- `enso_phase` — El Nino / La Nina / Neutral / Watch
- `oni_value` — Oceanic Niño Index (°C)
- `phase_label` — Human-readable label with strength
- `threats` — Object with heat_stress, drought, frost, wildfire, cyclone sub-objects
- `ai_interpretation` — Gemini Search-grounded summary with crop_risks, immediate_actions, alert_level
- `forecast_adjustments` — `rainfall_factor` and `temp_offset_c` applied to 6-month forecast

---

## 🔄 Recommendation Pipeline (Step by Step)

```text
Farmer Input (Country → State → District + Planning Horizon)
        │
        ▼
[1] Location Agent (location_agent.py)
        Resolves district exact lat/lon from world_locations.json (170+ districts).
        If not found, delegates to LLM Location Agent (llm_location_agent.py)
        which uses Gemini / Ollama to geocode any global rural location.
        │
        ▼
[2] Data Gathering Agent — Live Weather (data_gathering_agent.py)
        Fetches real-time weather from Open-Meteo (temp, rainfall, wind, UV, humidity).
        30-minute in-memory cache per lat/lon to avoid repeated API calls.
        │
        ▼
[3] Data Gathering Agent — Climatology Forecast
        Builds 6-month forecast from zone-based climatology (India: history.py;
        World: inline zone table). Anchors to live temperature for district accuracy.
        │
        ▼
[4] Climate Signal Intelligence (climate_signals.py) — NEW IN v3.1
        Fetches NOAA ENSO/ONI index (free, no key). Runs 9 independent threat
        assessors. Gemini Search Grounding retrieves real-time climate advisories.
        Applies research-based rainfall & temperature adjustments to 6-month forecast.
        │
        ▼
[5] Data Gathering Agent — LLM Enrichment
        Gemini Search Grounding (preferred) → fetches REAL current market prices,
        soil type, and agricultural advisories. Falls back to plain Gemini, then
        zone-based defaults.
        │
        ▼
[6] Crop Agent — AI Scoring Engine (crop_agent.py)
        Pipeline: Cache → Gemini+Search → Gemini Plain → Ollama+Search → Ollama → Zone Fallback
        Country-specific crop hint lists prevent geographic hallucination.
        Validates output; rejects Hindi crop names for non-South-Asian countries.
        │
        ▼
[7] Streaming Response
        Streams final ranked recommendations, weather, climate intelligence, and
        LLM-generated summary to the UI in real time via Server-Sent Events (SSE).
```

---

## 🦙 LLM Integration (LLaMA 3.2 + Gemini Fallback)

| LLM Component | File | Purpose |
|---------------|------|---------|
| **Location Geocoding** | `llm_location_agent.py` | Geocodes unmapped rural districts globally |
| **Soil & Market Enrichment** | `data_gathering_agent.py` | Real-time soil types and market prices via Search Grounding |
| **Climate Intelligence** | `climate_signals.py` | 9-threat comprehensive assessment via Search Grounding |
| **Crop Ranking** | `crop_agent.py` | Real-time crop advisories, suitability scoring, pest warnings |
| **Streaming Explainer** | `crop_agent.py + llm_explainer.py` | Real-time LLM summary of why crops were recommended |
| **Farmer Chat** | `llm_chat.py` | Context-aware Q&A remembering live weather, ENSO, and region specifics |
| **Web Search Tool** | `web_search_agent.py` | DuckDuckGo tool-calling for Ollama when Search Grounding unavailable |

### Provider Priority Chain (all components)

```
1. Gemini + Google Search Grounding  ← Real-time advisories, current prices
2. Gemini Plain (4-key rotation)     ← Fallback if Search Grounding unavailable
3. Ollama + Web Search Tool          ← Local model with DuckDuckGo search
4. Ollama Plain                      ← Local model, no internet needed
5. Zone-based Rule Engine            ← Final fallback, instant, no API
```

### Gemini Key Rotation

The system supports up to **4 Gemini API keys** (env vars: `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`, `GEMINI_API_KEY_4`) rotated automatically on `429 RESOURCE_EXHAUSTED` errors. Having 3 keys effectively triples the free daily quota.

### Gemini Model Fallback Chain

```
gemini-2.5-flash-lite → gemini-2.0-flash-lite → gemini-2.0-flash-lite-001
  → gemini-flash-lite-latest → gemini-2.0-flash → gemini-2.5-flash
```

Search-grounded models: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.0-flash-001`

---

## 🌱 Crop Database (50+ Crops)

| Category | Example Crops |
|----------|---------------|
| **Millets** | Bajra, Jowar, Ragi, Foxtail Millet, Sorghum |
| **Pulses & Beans** | Moong, Urad, Cowpea, Soybean, Chickpea, Pigeon Pea |
| **Grains & Cereals** | Wheat, Rice, Maize, Barley, Oats |
| **Vegetables** | Tomato, Brinjal, Okra, Cucumber, Ridge Gourd, Spinach |
| **Fruits** | Apple, Mango, Banana, Citrus, Grapes |
| **Cash Crops** | Cotton, Sugarcane, Tea, Coffee, Canola, Sunflower |

### Country-Specific Crop Hints

The Crop Agent includes **region-specific crop hint lists** for 50+ countries (in local language where applicable). Examples:
- 🇩🇪 Germany: `Winterweizen (Winter Wheat), Winterraps (Canola), Zuckerrübe (Sugar Beet)...`
- 🇧🇷 Brazil: `Soja (Soybean), Milho (Maize), Cana-de-açúcar (Sugarcane)...`
- 🇮🇳 India: `Chawal/Dhan (Rice), Gehun (Wheat), Kapas (Cotton), Ganna (Sugarcane)...`

If Gemini returns incorrect crops for a country (e.g., Hindi names for a European location), the validation layer **rejects the result** and uses the zone fallback table instead.

---

## 🧪 Suitability Scoring Engine

| Factor | Weight | How It's Computed |
|--------|--------|-------------------|
| **Temperature Compatibility** | 25% | Optimal range (100) → linear decay to survival limits (60) → 8°C grace margin (20) → beyond limits (0) |
| **Water Availability** | 25% | Expected rainfall + irrigation vs. crop water requirement. **ENSO-adjusted in v3.1** |
| **Soil Compatibility** | 15% | Texture, pH, drainage, and organic matter matching |
| **Regional Suitability** | 15% | District-specific suitability modifiers |
| **Seasonal Adjustment** | 10% | Whether the crop is traditionally planted in the detected season |
| **Drought Tolerance Bonus** | 10% | Score boost for drought-resistant crops during expected dry spells (elevated in El Niño) |

*When the Random Forest model is trained and available, the final score becomes a 60:40 blend of ML prediction and rule-based engine.*

---

## ✨ Features

### 🌍 Global Location Support
- **50+ countries**, **250+ states/provinces**, and **170+ districts** in `world_locations.json`
- **LLM Location Agent** geocodes rural/unmapped locations on-the-fly via Gemini/Ollama
- Fallback to state capital coordinates if district is unknown

### 🌐 Climate Signal Intelligence (v3.1)
- Real-time **El Niño / La Niña** status from NOAA CPC (free, no API key)
- **9-dimensional threat assessment**: ENSO, heat, drought, frost, flood, cyclone, wildfire, soil moisture, climate change
- 6-month forecast **automatically adjusted** based on ENSO phase and climate zone
- Gemini Search Grounding provides **real-time regional advisories**
- **Climate Intelligence Panel** in the dashboard with phase badge, ONI value, threat cards, alert level, and AI advice

### 🧠 Graceful AI Degradation
- **Tier 1**: Gemini + Google Search Grounding — real-time crop advisories and market prices
- **Tier 2**: Ollama runs LLaMA 3.2 locally — free, private, offline with web search tool
- **Tier 3**: Plain LLM (no search) — general crop and climate knowledge
- **Tier 4**: Static rule-based scoring if no LLM is available
- **ENSO Tier**: Falls back to Neutral (no adjustment) if NOAA is unreachable

### 🌡️ Real-Time & Forecasted Weather
- **Live Open-Meteo integration** (free, no key) for today's weather
- **30-minute in-memory cache** per lat/lon for efficient repeated calls
- **6-Month forward-looking climate modeling** with ENSO adjustment layer
- Temperature, rainfall, humidity, wind, UV index, soil temperature

### 💬 Interactive Streaming Chat
- **Server-Sent Events (SSE)** for real-time token streaming
- Chat is **context-aware**: knows live weather, ENSO phase, soil, and current crop recommendations
- LLaMA/Gemini-powered, with graceful degradation

### 🔍 Web Search Agent
- `web_search_agent.py` implements DuckDuckGo search as an **Ollama tool-calling** interface
- Enables Ollama to get real-time crop advisories when Gemini is not configured

### 🤖 In-Memory Caching
- **Crop recommendations**: 1-hour cache per `(country, state, district, season, climate, irrigation)`
- **Weather**: 30-minute cache per lat/lon
- **Climate signals**: 6-hour cache per zone/country/temp/rainfall key

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, FastAPI 0.128+, Uvicorn 0.40+ |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript (ES6+), Chart.js |
| **Machine Learning** | Scikit-learn 1.8 (Random Forest), XGBoost 3.2, PyTorch 2.9 (LSTM) |
| **Agent Framework** | Custom Python Agents (5 specialized agents) |
| **Primary LLM** | **LLaMA 3.2** via **Ollama 0.6.2** (local, free, private) |
| **Fallback LLM** | **Google Gemini** via `google-genai 1.69` SDK (4-key rotation) |
| **Web Search** | Google Search Grounding (Gemini) + DuckDuckGo (Ollama tool) |
| **Live Weather** | Open-Meteo API (Free, no key required) |
| **Climate Signals** | NOAA CPC ONI (Free, no key required) |
| **Data Streaming** | Server-Sent Events (SSE) via FastAPI `StreamingResponse` |
| **Data Formats** | JSON, Parquet (legacy historical data) |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 2 GB | 6 GB (Python packages + LLaMA model) |
| **Ollama Model** | `gemma3:2b` (Lighter) | `llama3.2` (Recommended) |
| **Internet** | Required for live weather & NOAA | Stable broadband |
| **Browser** | Any modern browser | Chrome / Edge (latest) |

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- [Ollama](https://ollama.com/download) *(recommended — free local LLM)*
- A free **[Google Gemini API key](https://aistudio.google.com/app/apikey)** *(optional — used as fallback and for Search Grounding)*

> **NOAA Climate Data requires no API key** — it is fetched automatically.

### 1. Clone the Repository
```bash
git clone https://github.com/tirthch25/AI-Powered-Weather-Resilient-Crop-Advisor.git
cd AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama (Local LLM — Recommended)
```bash
# Download and install from https://ollama.com/download, then:
ollama pull llama3.2        # ~2 GB download, one-time
ollama serve                # Start the local server (auto-starts on Windows)
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Linux / macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env`:
```env
# Primary LLM provider
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Gemini API Keys (optional but recommended for Search Grounding)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_API_KEY_2=          # Optional: second key for higher quota
GEMINI_API_KEY_3=          # Optional: third key for higher quota
GEMINI_API_KEY_4=          # Optional: fourth key for higher quota

# NOAA climate data — no key needed, fetched automatically
```

### 6. Start the Platform
```bash
# Windows (one-command setup)
.\setup.bat

# Or run directly:
python run_website.py
```

### 7. Open in Browser
```
http://localhost:8000          ← Web Interface
http://localhost:8000/docs     ← Interactive Swagger API Docs
http://localhost:8000/health   ← System status & LLM provider info
http://localhost:8000/climate-signals?country=india  ← ENSO + 9-threat status
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/countries` | List of all supported countries |
| `GET` | `/api/states/{country}` | States/provinces for a given country |
| `GET` | `/api/districts/{country}/{state}` | Districts/cities for a state |
| `POST` | `/api/analyze/stream` | **Streaming** AI crop recommendation (SSE) |
| `GET` | `/climate-signals` | **9-dimensional** climate threat assessment + ENSO status |
| `POST` | `/chat` | Interactive Farmer Q&A Chat (single response) |
| `POST` | `/chat/stream` | Streaming Farmer Chat (SSE tokens) |
| `GET` | `/weather/now/{region_id}` | Live real-time temperature from Open-Meteo |
| `POST` | `/recommend` | Legacy full-batch JSON recommendation engine (v2.x) |
| `GET` | `/health` | API health, ML status, Ollama status, LLM provider |

### Example: Analyze Stream Request

```json
POST /api/analyze/stream
{
  "country": "India",
  "state": "Maharashtra",
  "district": "Pune",
  "irrigation": "Limited",
  "planning_days": 90,
  "soil_type": "Loam",
  "soil_ph": 6.8
}
```

Response: SSE stream of JSON progress events with weather, climate signals, soil, market prices, and ranked crop recommendations.

---

## 📁 Project Structure

```text
AI-Powered-Weather-Resilient-Crop-Advisor/
├── README.md                      # ← You are here
├── SETUP_GUIDE.md                 # Detailed setup & troubleshooting guide
├── user_module_paper.md           # Full technical user module documentation
├── HPC_SUPERCOMPUTER_GUIDE.md     # HPC/SLURM deployment guide
├── setup.bat                      # One-command Windows setup script
├── setup.sh                       # One-command Linux/macOS setup script
├── setup_project.py               # Python-based project setup utility
├── run_job.slurm                  # SLURM job script for HPC deployment
└── agri_crop_recommendation/
    ├── main.py                    # Alternative entrypoint
    ├── run_website.py             # Primary server startup script
    ├── requirements.txt           # Python dependencies (pinned)
    ├── .env.example               # Environment variable template (with docs)
    ├── data/
    │   └── reference/
    │       ├── world_locations.json    # 50+ countries, 250+ states, 170+ districts
    │       ├── crop_knowledge.json     # Crop database (50+ crops, LLM context)
    │       ├── regions.json            # Legacy: India 640-district database
    │       └── regional_crops.json    # Legacy: India regional crop data
    ├── models/
    │   ├── crop_suitability/           # Scikit-learn Random Forest models
    │   ├── weather_lstm/               # PyTorch LSTM weather forecast models
    │   └── weather_xgboost/            # XGBoost weather models
    ├── scripts/
    │   ├── test_api.py                 # API smoke test suite
    │   ├── test_chatbot.py             # Chatbot integration tests
    │   └── verify_models.py            # ML model verification
    ├── src/
    │   ├── agents/
    │   │   ├── location_agent.py       # World location resolver (world_locations.json)
    │   │   ├── llm_location_agent.py   # LLM geocoder for unmapped locations
    │   │   ├── data_gathering_agent.py # Weather + forecast + ENSO + soil + market
    │   │   ├── crop_agent.py           # AI crop ranking with Search Grounding
    │   │   └── web_search_agent.py     # DuckDuckGo tool for Ollama search
    │   ├── api/
    │   │   ├── app.py                  # FastAPI routes and SSE streaming logic
    │   │   └── models.py               # Pydantic request/response models
    │   ├── crops/                      # Crop database and soil definitions
    │   ├── ml/
    │   │   ├── predictor.py            # Random Forest suitability predictor
    │   │   └── lstm_weather.py         # LSTM weather forecasting
    │   ├── services/
    │   │   ├── climate_signals.py      # 9-dimensional climate threat service (NOAA + AI)
    │   │   ├── recommender.py          # Rule-based scoring + ML blend engine
    │   │   ├── llm_chat.py             # LLM Farmer Chat (SSE streaming)
    │   │   ├── llm_explainer.py        # Crop explanation generator
    │   │   ├── llm_filter.py           # LLM-based crop filter
    │   │   ├── risk.py                 # Drought + temperature risk assessment
    │   │   ├── pests.py                # Pest & disease warning system
    │   │   └── calendar.py             # Planting calendar generator
    │   ├── utils/                      # Region manager, season utilities
    │   └── weather/
    │       ├── history.py              # India zone-based climate history
    │       └── fetcher.py              # Open-Meteo API wrapper
    ├── static/
    │   ├── css/style.css               # Full dashboard CSS (Climate Panel styles)
    │   └── js/app.js                   # SSE client + renderClimatePanel()
    └── templates/
        └── index.html                  # Dashboard with Climate Intelligence Panel
```

---

## 🙏 Acknowledgements

- **[Open-Meteo](https://open-meteo.com/)** — Free, robust weather API (no key required).
- **[NOAA CPC](https://www.cpc.ncep.noaa.gov/)** — Free Oceanic Niño Index (ONI) data for ENSO monitoring.
- **[Ollama](https://ollama.com/)** — Local LLM inference made effortless.
- **[Meta LLaMA](https://llama.meta.com/)** — Open-weights LLaMA 3.2 model.
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance Python web framework.
- **[Google Gemini](https://deepmind.google/technologies/gemini/)** — Cloud LLM fallback with Search Grounding.
- **[DuckDuckGo](https://duckduckgo.com/)** — Privacy-respecting web search for Ollama tool-calling.
- **[Scikit-learn](https://scikit-learn.org/)** — Random Forest suitability model.
- **[CDAC Pune](https://www.cdac.in/)** — Compute infrastructure and HPC support.

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Global Agricultural Intelligence · v3.1</strong><br/>
  <em>Empowering agriculture through AI Agents, Real-Time Data, and Climate Science</em><br/><br/>
  <em>Developed by Tirth Chankeshwara | CDAC-Pune | June 2026</em>
</div>
