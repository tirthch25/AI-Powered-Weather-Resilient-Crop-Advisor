# 🌾 AI Powered Weather Resilient Crop Advisor v3.1

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2%20Local-8A2BE2?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![NOAA](https://img.shields.io/badge/NOAA-Climate%20Signals-0057B7?style=for-the-badge)
![HTML5](https://img.shields.io/badge/HTML5-Vanilla_JS-E34F26?style=for-the-badge&logo=html5)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-Powered Weather-Resilient Crop Advisor v3.1 powered by LLaMA & Web Search Agents — covering 50+ countries, 250+ states, and 170+ districts, now with real-time ENSO / El Niño / La Niña climate intelligence.**

[Overview](#-overview) • [Agent Architecture](#-agent-architecture) • [Climate Intelligence](#-climate-signal-intelligence-v31-new) • [Features](#-features) • [Installation](#-installation) • [API Endpoints](#-api-endpoints)

</div>

---

## 🧭 Overview

The **AI Powered Weather Resilient Crop Advisor** is an end-to-end, global agricultural intelligence platform. By combining live weather data, machine-learning-blended climate forecasts, real-time ENSO signals, and large language model (LLM) reasoning, the system replicates the advice of a skilled agronomist for virtually any farming region in the world.

A user selects their country, state, and district. The **Data Gathering Agent** instantly pulls:
1. Real-time weather from Open-Meteo
2. Historical climate-zone forecasts anchored to live temperature
3. **Current global ENSO status from NOAA CPC** (El Niño / La Niña / Neutral — free, no key)
4. Soil data and local market prices from the LLM

Finally, the **Crop Agent** ranks over 50 crops and streams a personalized, climate-aware analysis to the UI.

The platform relies on **LLaMA 3.2 (via Ollama)** running entirely locally, with Google Gemini as an automatic cloud fallback.

---

## 💡 Project Idea

Most crop advisory tools rely on generic, static lookup tables or expensive IoT sensors. This platform takes a radically different approach by acting as an **AI-powered climate-aware agronomist**. It combines **publicly available free data sources** with **Machine Learning**, **LLM Agents**, and **real-time ENSO climate signals** to replicate expert advice dynamically, anywhere in the world.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Global Precision** | Location Agent dynamically maps 170+ global districts. Open-Meteo fetches real-time, exact lat/lon weather. |
| **Climate Intelligence** | NOAA CPC ONI data (free) reveals El Niño/La Niña phase; AI agent interprets the impact for the farmer's location. |
| **Agentic Workflow** | Multi-agent system (Location, Data, Crop, Climate) orchestrates intelligence gathering behind a single `/api/analyze/stream` call. |
| **Graceful Degradation** | Remove LLM → rule-based scoring continues. Remove Internet → zone-based modeling. Remove NOAA → Neutral phase assumed. Zero single points of failure. |
| **Free & Private** | LLaMA 3.2 runs locally via Ollama. NOAA data is public. Open-Meteo requires no API key. |

---

## 🤖 Agent Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│       Web Browser ←→ index.html + app.js + style.css               │
│  Streaming Dashboard · Climate Intelligence Panel · Farmer Chat     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTP / REST / SSE Streaming (FastAPI)
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API LAYER                                     │
│  GET /api/countries · GET /api/states · GET /api/districts          │
│  POST /api/analyze/stream · POST /chat · GET /weather/now           │
│  GET /climate-signals  ← NEW                                        │
└──┬───────────────────────────┬──────────────────┬───────────────────┘
   │                           │                  │
┌──▼──────────────────┐ ┌──────▼──────────┐ ┌────▼────────────────────┐
│ Location Agent      │ │ Data Gathering  │ │ Climate Signal Service  │
│ world_locations.json│ │ Agent           │ │ (climate_signals.py)    │
│ 50+ Countries       │ │ Weather + Zone  │ │ NOAA ONI → ENSO Phase   │
│                     │ │ + ENSO Adjust   │ │ + AI Location Impact    │
└──┬──────────────────┘ └──────┬──────────┘ └─────────────────────────┘
   │                           │
┌──▼───────────────────────────▼────────────────────────────┐
│                       LLM LAYER                            │
│  Primary:  LLaMA 3.2 via Ollama (local, free, private)    │
│  Fallback: Google Gemini (if Ollama not running)          │
│  Uses: Soil, Market Prices, Crop Ranking, Chat, ENSO      │
└──┬────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  Crop DB (50+ crops) · World Locations · Zone Climate Normals       │
│  Open-Meteo Live API (Free) · NOAA CPC ONI (Free)                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 Climate Signal Intelligence (v3.1 — NEW)

Version 3.1 adds a full **Climate Signal Intelligence** layer. The system automatically fetches the current global ENSO state from NOAA CPC every 6 hours and applies research-based adjustments to the 6-month forecast and crop recommendations.

### How It Works (No API Key Required)

```text
NOAA CPC (Free)
  └── ONI text file → Current Oceanic Niño Index value
        │
        ▼
  Phase Detection
  El Niño (ONI ≥ +0.5) | La Niña (ONI ≤ −0.5) | Neutral
        │
        ▼
  Research-Based Forecast Adjustment (per climate zone)
  India El Niño → rainfall ×0.80, temp +0.5°C
  India La Niña → rainfall ×1.20, temp −0.3°C
        │
        ▼
  Gemini/LLaMA Interpretation
  → Plain-language impact summary for the farmer's location
  → Crop risks list, seasonal opportunity, alert level
        │
        ▼
  6-Month Forecast (ENSO-adjusted)
  + Climate Intelligence Panel in dashboard UI
```

### ENSO Impact Reference

| ENSO Phase | India / South Asia | Impact on Crops |
|------------|-------------------|-----------------|
| 🔴 **El Niño** | Weaker SW monsoon, below-normal rainfall, +0.5°C warmer | Drought stress risk for Kharif crops; favour drought-tolerant varieties |
| 🔵 **La Niña** | Stronger SW monsoon, above-normal rainfall, −0.3°C cooler | Flood/waterlogging risk; fungal disease pressure rises |
| 🟢 **Neutral** | Near-normal conditions | Standard seasonal recommendations apply |

### New `/climate-signals` Endpoint

```http
GET /climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical
```

Returns:
- `enso_phase` — El Nino / La Nina / Neutral
- `oni_value` — Oceanic Niño Index (°C)
- `phase_label` — Human-readable label with strength
- `forecast_adjustments` — `rainfall_factor` and `temp_offset_c` applied to forecast
- `ai_interpretation` — AI-generated summary, crop risks, opportunity, alert level

*Data: NOAA CPC (free, no API key). Interpretation: existing Gemini/LLaMA agent.*

---

## 🔄 Recommendation Pipeline (Step by Step)

```text
Farmer Input (Country → State → District + Planning Horizon)
        │
        ▼
[1] Location Agent
        Resolves the district's exact latitude and longitude using world_locations.json.
        │
        ▼
[2] Data Gathering Agent — Live Weather
        Fetches real-time weather from Open-Meteo (temp, rainfall, wind, UV).
        │
        ▼
[3] Data Gathering Agent — Climatology Forecast
        Builds a 6-month forecast from historical climate zone data,
        anchored to the live temperature for district-level accuracy.
        │
        ▼
[4] Climate Signal Intelligence (NEW)
        Fetches NOAA ENSO/ONI index (free). Applies research-based rainfall &
        temperature adjustments to the 6-month forecast. AI generates a plain-
        language impact summary for the farmer's specific location.
        │
        ▼
[5] Data Gathering Agent — LLM Enrichment
        Invokes Gemini/LLaMA to assess soil profiles and market prices.
        Falls back to zone defaults if LLM is down.
        │
        ▼
[6] Crop Agent — Scoring Engine
        Evaluates 50+ crops using a 6-factor suitability score (Temp, Water,
        Soil, Regional, Season, Drought). Optionally blends with a Random Forest
        ML model (0.6 ML + 0.4 Rule-based).
        │
        ▼
[7] Risk & Pest Assessment
        Flags drought risk, temperature stress, and pest warnings based on live
        weather and ENSO-adjusted forecasts.
        │
        ▼
[8] Streaming Response
        Streams the final ranked recommendations and a generated summary to the
        UI in real time via Server-Sent Events.
```

---

## 🦙 LLM Integration (LLaMA 3.2 + Gemini Fallback)

| LLM Component | File | Purpose |
|---------------|------|---------|
| **Soil & Market Enrichment** | `data_gathering_agent.py` | Dynamically assesses local soil types and market prices. |
| **ENSO Interpretation** | `climate_signals.py` | Translates global ENSO phase into a location-specific farming impact. |
| **Streaming Explainer** | `crop_agent.py` | Generates a real-time summary of why specific crops were recommended. |
| **Farmer Chat** | `llm_chat.py` | Context-aware Q&A bot remembering live weather, ENSO, and region specifics. |

**Provider Priority Chain:**
1. **LLaMA 3.2 via Ollama** (Local, free, private) ← *Default*
2. **Google Gemini** (Automatic fallback if Ollama is unreachable)
3. **Graceful Degradation** (Rule-based defaults kick in if both are unavailable)

---

## 🌱 Crop Database (50+ Crops)

| Category | Example Crops |
|----------|---------------|
| **Millets** | Bajra, Jowar, Ragi, Foxtail Millet |
| **Pulses & Beans** | Moong, Urad, Cowpea, Soybean |
| **Grains & Cereals** | Wheat, Rice, Maize, Barley |
| **Vegetables** | Tomato, Brinjal, Okra, Cucumber, Ridge Gourd |
| **Fruits** | Apple, Mango, Banana, Citrus |
| **Cash Crops** | Cotton, Sugarcane, Tea, Coffee |

---

## 🧪 Suitability Scoring Engine

| Factor | Weight | How it's computed |
|--------|--------|-------------------|
| **Temperature Compatibility** | 25% | Optimal range (100) → linear decay to survival limits (60) → 8°C grace margin (20) → beyond limits (0). |
| **Water Availability** | 25% | Expected rainfall + irrigation vs. crop water requirement. **ENSO-adjusted in v3.1.** |
| **Soil Compatibility** | 15% | Texture, pH, drainage, and organic matter matching. |
| **Regional Suitability** | 15% | District-specific suitability modifiers. |
| **Seasonal Adjustment** | 10% | Whether the crop is traditionally planted in the detected season. |
| **Drought Tolerance Bonus** | 10% | Score boost for drought-resistant crops during expected dry spells (elevated in El Niño). |

*(Note: If the Random Forest model is trained and available, the final score becomes a 60:40 blend of ML prediction and this rule-based engine).*

---

## ✨ Features

### 🌍 Global Location Support
- Supports **50+ countries**, **250+ states/provinces**, and **170+ districts**.
- Includes accurate fallback mechanisms to state capitals or regional centers.

### 🌐 Climate Signal Intelligence (v3.1 New)
- Real-time **El Niño / La Niña** status from NOAA CPC (free, no API key).
- 6-month forecast **automatically adjusted** based on ENSO phase and climate zone.
- AI agent generates a **plain-language impact summary** with crop risks and opportunities.
- **Climate Intelligence Panel** in the dashboard shows phase badge, alert level, rainfall/temp outlook, and AI-generated advice.

### 🧠 Graceful AI Degradation
- **Tier 1**: Ollama runs LLaMA 3.2 locally — free, private, offline.
- **Tier 2**: Seamlessly falls back to Google Gemini if Ollama isn't running.
- **Tier 3**: Static rule-based scoring if no LLM is available.
- **ENSO Tier**: Falls back to Neutral (no adjustment) if NOAA is unreachable.

### 🌡️ Real-Time & Forecasted Weather
- Live Open-Meteo integration for accurate today-weather.
- 6-Month forward-looking climate modeling with ENSO adjustment layer.

### 💬 Interactive Streaming Chat
- Real-time token streaming for both the main crop analysis and the Farmer Chat.
- Chat is aware of live weather, ENSO phase, and current crop recommendations.

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript |
| **Machine Learning** | Scikit-learn (Random Forest Suitability Model) |
| **Agent Framework** | Custom Python Agents |
| **Primary LLM** | **LLaMA 3.2** via **Ollama** (local, free, private) |
| **Fallback LLM** | Google Gemini (`gemini-2.0-flash-lite`) |
| **Live Weather** | Open-Meteo API (Free, no key) |
| **Climate Signals** | NOAA CPC ONI (Free, no key) — **NEW in v3.1** |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 600 MB | 3 GB (to hold the LLaMA model) |
| **Ollama Model** | `gemma3:2b` (Lighter) | `llama3.2` |
| **Internet** | Required for live weather & NOAA | Stable broadband |

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- [Ollama](https://ollama.com/download) *(recommended — free local LLM)*
- A free **[Google Gemini API key](https://aistudio.google.com/app/apikey)** *(optional — used as fallback)*

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

### 4. Set Up Ollama (Local LLM)
```powershell
# Download and install from https://ollama.com/download, then:
ollama pull llama3.2        # ~2 GB download, one-time
ollama serve                # Start the local server
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```env
# Primary LLM provider
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Optional fallback
GEMINI_API_KEY=your_gemini_api_key_here

# NOAA climate data — no key needed, fetched automatically
```

### 6. Start the Platform
```bash
# Windows
.\setup.bat

# Or run directly:
python run_website.py
```

### 7. Open in Browser
```
http://localhost:8000          ← Web Interface
http://localhost:8000/docs     ← Interactive Swagger API Docs
http://localhost:8000/health   ← System status
http://localhost:8000/climate-signals?country=india  ← ENSO Status (new)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/countries` | List of supported countries |
| `GET` | `/api/states/{country}` | States/provinces for a given country |
| `GET` | `/api/districts/{country}/{state}` | Districts/cities |
| `POST` | `/api/analyze/stream` | **Streaming** crop recommendation via LLM Agent |
| `GET` | `/climate-signals` | **NEW** — Current ENSO status + AI location impact |
| `POST` | `/chat` | Interactive Farmer Q&A Chat |
| `POST` | `/chat/stream` | Streaming Farmer Chat (SSE) |
| `GET` | `/weather/now/{region_id}` | Live real-time temperature from Open-Meteo |
| `POST` | `/recommend` | Legacy full-batch JSON recommendation engine |
| `GET` | `/health` | API health, ML status, and active LLM provider |

---

## 📁 Project Structure

```text
agri_crop_recommendation/
├── data/                       # Reference data (locations, crops)
├── models/                     # Saved ML models
├── scripts/                    # Utility scripts (training, scraping)
├── src/
│   ├── agents/                 # LLaMA Agents (Location, Data Gathering, Crop)
│   │   └── data_gathering_agent.py  # Now includes ENSO adjustment step
│   ├── api/                    # FastAPI endpoints (app.py, models.py)
│   ├── crops/                  # Crop database and soil definitions
│   ├── ml/                     # ML pipelines and predictors
│   ├── services/
│   │   ├── recommender.py      # Score blending engine
│   │   └── climate_signals.py  # NEW — NOAA ENSO fetch + AI interpretation
│   ├── utils/                  # Region manager, seasons
│   └── weather/                # Weather fetcher, historical climatology
├── static/                     # CSS, JS, Images
│   ├── css/style.css           # Includes Climate Intelligence Panel styles
│   └── js/app.js               # Includes renderClimatePanel() function
├── templates/
│   └── index.html              # Dashboard with Climate Intelligence Panel
├── .env.example                # Example environment variables
├── requirements.txt            # Python dependencies
└── run_website.py              # Server startup script
```

---

## 🙏 Acknowledgements

- **[Open-Meteo](https://open-meteo.com/)** — Free, robust weather API (no key required).
- **[NOAA CPC](https://www.cpc.ncep.noaa.gov/)** — Free Oceanic Niño Index (ONI) data for ENSO monitoring.
- **[Ollama](https://ollama.com/)** — Local LLM inference made effortless.
- **[Meta LLaMA](https://llama.meta.com/)** — Open-weights LLaMA 3.2 model.
- **[FastAPI](https://fastapi.tiangolo.com/)** — High-performance Python web framework.
- **[Google Gemini](https://deepmind.google/technologies/gemini/)** — Cloud LLM fallback engine.

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Global Agricultural Intelligence · v3.1</strong><br/>
  <em>Empowering agriculture through AI Agents, Real-Time Data, and Climate Science</em>
</div>
