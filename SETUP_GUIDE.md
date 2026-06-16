# AI Powered Weather Resilient Crop Advisor v3.1 — Complete Setup Guide

This guide takes the project from a fresh clone to a fully running local global dashboard. It covers every component, configuration option, LLM setup, climate intelligence feature, verification steps, and all common failure modes.

---

## Table of Contents

1. [What's New in v3.1](#whats-new-in-v31)
2. [System Requirements](#system-requirements)
3. [Setup Map](#setup-map)
4. [Option A: One-Command Setup](#option-a-one-command-setup)
5. [Option B: Manual Step-by-Step Setup](#option-b-manual-step-by-step-setup)
6. [Environment File Reference](#environment-file-reference)
7. [LLM Setup (Ollama + Gemini)](#llm-setup-ollama--gemini)
8. [Google Search Grounding](#google-search-grounding)
9. [Verifying All Components](#verifying-all-components)
10. [Using the Web Application](#using-the-web-application)
11. [API Smoke Tests](#api-smoke-tests)
12. [Model and Data Checks](#model-and-data-checks)
13. [Troubleshooting](#troubleshooting)
14. [Common Commands](#common-commands)
15. [What To Do After Setup](#what-to-do-after-setup)

---

## What's New in v3.1

v3.1 adds two major layers on top of the v3.0 Global Dashboard:

### Climate Signal Intelligence (9 Threat Dimensions)

| Feature | Detail |
|---------|--------|
| **ENSO / El Niño Detection** | Automatically fetches NOAA CPC Oceanic Niño Index (ONI) every 6 hours |
| **No New API Key** | NOAA data is free and public — no registration |
| **9 Threat Assessors** | ENSO, Heat Stress, Drought Index, Frost, Flood, Cyclone Basin, Wildfire, Soil Moisture, Climate Change Trend |
| **Forecast Adjustment** | 6-month rainfall and temperature forecasts adjusted per ENSO phase and climate zone |
| **AI Climate Summary** | Gemini Search Grounding retrieves real-time regional advisories |
| **Climate Intelligence Panel** | New dashboard panel with threat badges, alert level, and AI advice |
| **New API Endpoint** | `GET /climate-signals` |

### Search-Grounded AI Agents (all three agents upgraded)

| Component | What Changed |
|-----------|-------------|
| **Crop Agent** | Now tries Gemini + Google Search first for real-time crop advisories and market prices |
| **Data Gathering Agent** | Search Grounding for real current market prices, not generic estimates |
| **Climate Signals** | Search Grounding for active government crop advisories and regional climate alerts |
| **Web Search Agent** | New DuckDuckGo tool-calling agent for Ollama when Gemini is not configured |

---

## System Requirements

| Component | Minimum | Recommended | Why It Matters |
|-----------|---------|-------------|----------------|
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 13+, Ubuntu 22.04+ | Modern Python and package compatibility |
| Python | 3.8+ | 3.10 or 3.11 | FastAPI, ML libraries, and Pydantic v2 |
| RAM | 4 GB | 8 GB+ | Local LLM (LLaMA 3.2 uses ~3 GB) |
| Storage | 2 GB | 6 GB+ | Python packages (~1 GB) + LLaMA model (~2 GB) |
| Internet | Required for weather & NOAA | Stable broadband | Open-Meteo + NOAA + Gemini Search |
| Browser | Chrome/Edge/Firefox (current) | Chrome or Edge | SSE streaming + Climate Panel |
| Ollama / Gemini | Highly Recommended | Both configured | Full agentic experience |

> **Note on LLMs:** v3.1 relies heavily on LLMs for crop filtering, climate interpretation, and ENSO analysis. Without any LLM, the system degrades to rule-based zone defaults.

> **Note on Search Grounding:** Google Search Grounding requires a Gemini API key with a model that supports it (`gemini-2.0-flash` or higher). Without it, the system falls back to static LLM knowledge.

---

## Setup Map

```text
Install Python 3.8+
  → Clone repository
  → cd agri_crop_recommendation
  → Create virtual environment (.venv)
  → Activate virtual environment
  → pip install -r requirements.txt
  → Copy .env.example → .env
  → Set LLM_PROVIDER, GEMINI_API_KEY in .env
  → (Recommended) Install Ollama and pull llama3.2
  → Verify: python run_website.py
  → Open http://localhost:8000
  → Test: curl http://localhost:8000/health
  → Test: curl "http://localhost:8000/climate-signals?country=india"
```

---

## Option A: One-Command Setup

Run from the **repository root** (not inside `agri_crop_recommendation/`).

### Windows

```powershell
.\setup.bat
```

### macOS / Linux

```bash
bash setup.sh
```

The setup script:
1. Checks Python version (requires 3.8+)
2. Creates `agri_crop_recommendation/.venv`
3. Installs all packages from `requirements.txt`
4. Copies `.env.example` → `.env` if `.env` doesn't exist
5. Checks if Ollama is installed and running
6. Verifies key data files (`world_locations.json`, `crop_knowledge.json`)
7. Optionally starts the server

---

## Option B: Manual Step-by-Step Setup

### Step 1: Navigate to the app directory

```bash
cd agri_crop_recommendation
```

All subsequent commands run from inside `agri_crop_recommendation/`.

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows PowerShell:**
```powershell
.venv\Scripts\activate
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

### Step 4: Upgrade pip and Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key packages installed:
- `fastapi==0.128.7` + `uvicorn==0.40.0` — API server
- `ollama==0.6.2` — Local LLM client
- `google-genai==1.69.0` — Gemini AI with Search Grounding
- `scikit-learn==1.8.0` — ML models
- `torch==2.9.0` + `xgboost==3.2.0` — LSTM & XGBoost legacy models
- `requests==2.32.5` — HTTP client for Open-Meteo + NOAA
- `pandas==2.3.3` + `pyarrow==22.0.0` — Data processing
- `python-dotenv==1.2.1` — Environment variable loading
- `jinja2==3.1.6` — HTML templating

### Step 5: Configure Environment

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env` and fill in your values (see [Environment File Reference](#environment-file-reference) below).

### Step 6: Start the Server

```bash
python run_website.py
```

### Step 7: Open the App

```
http://localhost:8000
```

---

## Environment File Reference

All configuration lives in `agri_crop_recommendation/.env`. Copy from `.env.example` to get started.

### Full Reference Table

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `ollama` | Primary LLM: `ollama` or `gemini` |
| `OLLAMA_MODEL` | If Ollama | `llama3.2` | Ollama model name (see options below) |
| `OLLAMA_BASE_URL` | If Ollama | `http://localhost:11434` | Ollama API URL |
| `GEMINI_API_KEY` | Recommended | _(blank)_ | Primary Gemini API key |
| `GEMINI_API_KEY_2` | Optional | _(blank)_ | Second Gemini key (auto-rotated on 429) |
| `GEMINI_API_KEY_3` | Optional | _(blank)_ | Third Gemini key |
| `GEMINI_API_KEY_4` | Optional | _(blank)_ | Fourth Gemini key |

### Configuration Modes

| Mode | `.env` Setup | Best For |
|------|-------------|----------|
| **Local-first** (recommended) | `LLM_PROVIDER=ollama` | Private local inference + Gemini fallback |
| **Cloud + Search** | `LLM_PROVIDER=ollama` + `GEMINI_API_KEY=...` | Best quality: Search Grounding for real-time data |
| **Cloud-only** | `LLM_PROVIDER=gemini` + `GEMINI_API_KEY=...` | Machines without GPU / can't run local model |
| **No LLM (Legacy)** | Leave keys blank, don't start Ollama | Falls back to v2.x rule-based engine |

### Example: Recommended Setup

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

GEMINI_API_KEY=AIzaSy...your_key_here
GEMINI_API_KEY_2=AIzaSy...second_key   # optional, triples quota
GEMINI_API_KEY_3=                        # leave blank if only 1 key
GEMINI_API_KEY_4=
```

---

## LLM Setup (Ollama + Gemini)

### Ollama Setup (Local LLM)

Install from:
```
https://ollama.com/download
```

**Windows:** Run the `.exe` installer — Ollama is added to PATH and starts as a background service automatically.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:** Download the `.dmg`, open it — Ollama runs as a menu-bar app.

#### Pull a Model

```bash
ollama pull llama3.2      # Recommended — ~2 GB download
```

#### Start Ollama (if not auto-started)

```bash
ollama serve
```

#### Verify Ollama is Running

```bash
ollama list                # Shows downloaded models
curl http://localhost:11434/api/tags   # JSON response confirms server is up
```

#### Model Options

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2` | ~2 GB | ✅ Recommended default — fast, good reasoning |
| `gemma3:2b` | ~1.5 GB | Lightest option for low-RAM machines |
| `llama3.1` | ~5 GB | Better reasoning, needs 8 GB+ RAM |
| `mistral` | ~4 GB | Good multilingual performance |

After changing `OLLAMA_MODEL` in `.env`, pull the new model:
```bash
ollama pull gemma3:2b
```

### Gemini API Key Setup

1. Open → [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"** → **"Create API key in new project"**
4. Copy the key (starts with `AIzaSy...`)
5. Paste into `.env` as `GEMINI_API_KEY=AIzaSy...`

**Free Tier Limits (as of 2025):**
- `gemini-2.5-flash-lite` → 1,500 requests/day, 15 requests/minute per key
- 3 keys → effectively 4,500 requests/day free

**Getting a Second Key:**  
Repeat steps 1–4 and use a different Google project. Paste as `GEMINI_API_KEY_2`.

---

## Google Search Grounding

**What it does:** Allows Gemini to search the live web before answering. Enables:
- **Real-time crop market prices** (not generic estimates)
- **Active pest/disease alerts** for your region
- **Current government agricultural advisories**
- **Live climate anomaly reports**

**Requirements:**
- A Gemini API key in `.env` (`GEMINI_API_KEY`)
- A model that supports it: `gemini-2.0-flash`, `gemini-2.5-flash`, or `gemini-2.0-flash-001`

**Which components use it:**
1. `crop_agent.py` — Real-time crop advisories and current prices (highest priority)
2. `data_gathering_agent.py` — Real current market prices and soil info
3. `climate_signals.py` — Active government crop advisories and climate alerts

**Fallback:** If Search Grounding is unavailable, falls through to plain Gemini → Ollama → zone defaults.

---

## Verifying All Components

### 1. Server Health Check

```bash
curl http://localhost:8000/health
```

Expected response includes:
```json
{
  "status": "healthy",
  "version": "3.1",
  "regions_loaded": true,
  "ml_models": {"crop_suitability": true, "weather_lstm": false},
  "llm_available": true,
  "llm_provider": "ollama",
  "ollama_running": true,
  "gemini_keys": 2,
  "timestamp": "..."
}
```

### 2. Climate Signal Feature (NOAA + AI)

```bash
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical"
```

Expected response shape:
```json
{
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
      "cyclone": {"storm_type": "Cyclone", "in_active_season": false}
    },
    "ai_interpretation": {
      "summary": "...",
      "crop_risks": ["..."],
      "immediate_actions": ["..."],
      "alert_level": "None",
      "rainfall_outlook": "Near Normal",
      "temp_outlook": "Near Normal"
    },
    "forecast_adjustments": {
      "rainfall_factor": 1.0,
      "temp_offset_c": 0.0,
      "description": "No ENSO adjustment (Neutral conditions)"
    },
    "source": "NOAA CPC + Gemini Search Grounding + Live Weather"
  }
}
```

If NOAA is unreachable, `oni_value` will be `0.0` and `enso_phase` will be `"Neutral"` — no error is raised.

### 3. Location API

```bash
curl http://localhost:8000/api/countries
curl http://localhost:8000/api/states/india
curl http://localhost:8000/api/districts/india/maharashtra
```

### 4. Ollama Direct Test

```bash
ollama run llama3.2 "What crops grow well in Maharashtra in June?"
```

### 5. Browser Checks

```
http://localhost:8000                        ← Web Interface + Climate Panel
http://localhost:8000/docs                   ← Swagger Interactive API Docs
http://localhost:8000/health                 ← System status
http://localhost:8000/climate-signals?country=india  ← ENSO + 9-threat status
```

---

## Using the Web Application

### Step-by-Step User Flow

1. **Select a country** from the dropdown (e.g., India, USA, Brazil, Germany)
2. **Select a state** — populates dynamically from world_locations.json
3. **Select a district** — maps to exact lat/lon for weather fetch
4. **Set irrigation** — None / Limited / Full (affects crop scoring)
5. *(Optional)* **Provide soil texture and pH** — override AI-detected values
6. **Set planning period** in days (30 / 60 / 90 / 180)
7. **Click "Analyze with AI Agent"**
8. **Watch the streaming progress** — see each agent step in real time via SSE
9. **Review results:**
   - 🌡️ **Live Weather Card** — Today's temp, humidity, rainfall, wind, UV
   - 🏔️ **Soil Card** — AI-detected soil type, pH, organic matter, drainage
   - 🌐 **Climate Intelligence Panel** — ENSO badge + 9-threat assessment
   - 📈 **6-Month Forecast** — ENSO-adjusted monthly temp and rainfall chart
   - 💰 **Market Prices** — Real-time (search-grounded) or zone-default prices
   - 🌱 **Crop Rankings** — Top 6 crops sorted by AI suitability score
10. **Ask a follow-up** in the Farmer Chat below the results

### Climate Intelligence Panel Breakdown

The **Climate Intelligence Panel** appears automatically between the Soil Card and the 6-Month Forecast. It shows:

| Element | Description |
|---------|-------------|
| ENSO Phase Badge | 🔴 El Niño / 🔵 La Niña / 🟢 Neutral + ONI value |
| ⚠️ Alert Bar | Warning level: None / Advisory / Watch / Warning / Emergency |
| Threat Cards | Heat Stress · Drought · Frost · Wildfire · Cyclone (if active) |
| AI Summary | 3-sentence Gemini Search-grounded analysis for your location |
| Rainfall Outlook | Below Normal / Near Normal / Above Normal |
| Temp Outlook | Below Normal / Near Normal / Above Normal |
| Crop Risks | 5 specific risks for current ENSO phase at your location |
| Immediate Actions | 3 farming actions to take this week |
| Forecast Adjustment | The rainfall factor and temp offset applied to the 6-month chart |
| Data Source | NOAA CPC + Gemini Search + Live Weather |

### Good First Test

```
Country: India
State: Maharashtra
District: Pune
Irrigation: Limited
Planning period: 90 days
Soil: Loam, pH 6.8
```

### International Test (Demonstrates Global Coverage)

```
Country: Germany
State: Bavaria
District: Munich
Irrigation: Full
Planning period: 150 days
```

---

## API Smoke Tests

With the server running at `http://localhost:8000`:

```bash
# Basic health
curl http://localhost:8000/health

# List countries
curl http://localhost:8000/api/countries

# States for India
curl http://localhost:8000/api/states/india

# Districts for Maharashtra
curl http://localhost:8000/api/districts/india/maharashtra

# Climate signals (v3.1)
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune"

# Climate signals for a European location
curl "http://localhost:8000/climate-signals?country=germany&state=Bavaria&district=Munich&climate_zone=Temperate"
```

**Main Streaming Endpoint:**

```bash
curl -X POST http://localhost:8000/api/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"country":"India","state":"Maharashtra","district":"Pune","irrigation":"Limited","planning_days":90}'
```

**Legacy v2.x Fallback (Windows PowerShell):**

```powershell
curl -X POST http://localhost:8000/recommend `
  -H "Content-Type: application/json" `
  -d "{`"region_id`":`"MH_PUNE`",`"irrigation`":`"Limited`",`"planning_days`":90}"
```

---

## Model and Data Checks

Run from inside `agri_crop_recommendation/`:

```bash
# v3.1 API Tests
python scripts/test_api.py

# Chatbot Integration Test
python scripts/test_chatbot.py

# Legacy ML Model Verification
python scripts/verify_models.py
```

### Critical v3.1 Data Files

```text
data/reference/
├── world_locations.json  (Legacy reference — superseded by LLM agent in v3.1)
└── crop_knowledge.json   (Core LLM Context Database, 50+ crops)

src/agents/
├── location_agent.py       (195 UN countries ISO list; states + districts 100% LLM-driven)
├── llm_location_agent.py   (Gemini/Ollama: states, districts, coords — 24h cache)
└── data_gathering_agent.py  (Updated — Step 3 now includes ENSO adjustment)

src/services/
└── climate_signals.py      (9-dimensional climate threat assessment — NOAA + Gemini Search)

src/agents/ (also important)
├── crop_agent.py           (AI crop ranking — Search Grounding + country crop hints)
└── web_search_agent.py     (DuckDuckGo tool for Ollama)
```

### Legacy Files (Maintained for Fallback)

```text
models/
├── crop_suitability/       ← Random Forest ML (v2.x)
├── weather_lstm/           ← LSTM weather forecast (v2.x)
└── weather_xgboost/        ← XGBoost weather (v2.x)

data/reference/
├── regions.json            ← India 640-district database (v2.x)
└── regional_crops.json     ← India regional crop catalog (v2.x)
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `python` not found | Python not on PATH | Install Python 3.8+ and tick **"Add Python to PATH"** |
| PowerShell activation blocked | Execution policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails on torch | Old pip or network issue | `python -m pip install --upgrade pip`, then retry |
| `ModuleNotFoundError: src` | Running from wrong directory | Must run from inside `agri_crop_recommendation/` |
| Chat / Recommendations unavailable | Ollama stopped or no Gemini key | Run `ollama serve` or add `GEMINI_API_KEY` to `.env` |
| Ollama model missing | Model was not pulled | `ollama pull llama3.2` |
| `429 RESOURCE_EXHAUSTED` | Gemini quota exceeded | Add `GEMINI_API_KEY_2` and `GEMINI_API_KEY_3` to `.env` |
| Search Grounding not working | Model doesn't support it | Use `gemini-2.0-flash` or `gemini-2.5-flash` in search-capable slot |
| Climate Panel not showing | NOAA unreachable or JS error | Check browser console. NOAA fetch silently falls back to Neutral. |
| `oni_value` always 0.0 | NOAA unreachable (firewall) | Normal fallback — system assumes Neutral, no error raised |
| Weather data missing | Open-Meteo unreachable | Retry later; zone-based estimates are used automatically |
| Port 8000 in use | Another server running | Stop it or use an alternate port (see below) |
| `world_locations.json` not found | Wrong working directory | Run from `agri_crop_recommendation/` |
| Hindi crop names in non-India result | LLM geographic hallucination | The validation layer auto-detects and uses zone fallback instead |

### Running on a Different Port

```bash
python -c "import uvicorn; uvicorn.run('src.api.app:app', host='0.0.0.0', port=8080)"
```

Then open `http://localhost:8080`.

### Checking Log Output

The server logs each agent step. Look for:
```
[DataAgent] Live weather: 32.4°C, rain7d=15.2mm, humidity=68%
[ClimateSignals] El Nino ONI=1.2 | heat=None drought=None frost=None cyclone=Cyclone wildfire=None
[CropAgent] Gemini search-grounded OK (gemini-2.0-flash, key ...abc123) → 6 crops
[DataAgent] Gemini search-grounded enrich OK (gemini-2.0-flash, key ...abc123)
```

---

## Common Commands

```bash
# Start the server
python run_website.py

# Run all API checks
python scripts/test_api.py

# Test the chatbot
python scripts/test_chatbot.py

# Test ENSO endpoint
curl "http://localhost:8000/climate-signals?country=india"

# Check Ollama models
ollama list

# Pull the default model
ollama pull llama3.2

# Check server health
curl http://localhost:8000/health

# Test streaming analyze (basic)
curl -X POST http://localhost:8000/api/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"country":"India","state":"Maharashtra","district":"Pune","irrigation":"Limited","planning_days":90}'
```

---

## What To Do After Setup

### Explore Global Coverage
- Try two districts in entirely different countries (e.g., **Pune, India** vs **Munich, Germany**) and compare recommendations.
- Test an African location (e.g., **Lagos, Nigeria**) or Southeast Asia (e.g., **Bangkok, Thailand**).

### Explore Climate Intelligence
- Check the **Climate Intelligence Panel** — observe whether ENSO adjustments change the 6-month rainfall values vs the baseline.
- During an active El Niño year, switch irrigation from `None` to `Full` and observe how drought-tolerant crops rise in ranking.
- Look for cyclone/typhoon alerts for coastal regions (e.g., Odisha, India or Philippines).

### Test Sensitivity
- Override soil pH and texture and observe how different soil types change the suitability score ranking.
- Change planning days from 30 to 180 and observe how the crop list changes.

### Explore the AI Chat
- Ask: *"How does the current El Niño affect my wheat crop in Pune?"*
- Ask: *"What is the best fertilizer schedule for cotton in this climate?"*
- Ask: *"When should I plant rabi crops given the current forecast?"*

### Use the API Docs
- Open `/docs` and explore the `/climate-signals`, `/api/analyze/stream`, and `/chat/stream` request models.
- Use the **"Try it out"** feature in Swagger to test any endpoint interactively.

### Train the ML Model (Optional)
```bash
cd agri_crop_recommendation
python scripts/train_model.py
```
This trains the Scikit-learn Random Forest on local crop suitability data. Once trained, final recommendations blend ML (60%) + rule-based engine (40%).

---

*End of Setup Guide — AI Powered Weather Resilient Crop Advisor v3.1*  
*Prepared by Tirth Chankeshwara | CDAC-Pune | June 2026*
