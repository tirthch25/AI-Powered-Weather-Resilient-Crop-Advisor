# AI Powered Weather Resilient Crop Advisor v4.0 — Complete Setup Guide

This guide takes the project from a fresh clone to a fully running local global dashboard. It covers every component, configuration option, LLM setup, data pipeline, verification steps, and all common failure modes.

---

## Table of Contents

1. [What's New in v4.0](#whats-new-in-v40)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Option A: One-Command Setup](#option-a-one-command-setup)
5. [Option B: Manual Step-by-Step Setup](#option-b-manual-step-by-step-setup)
6. [Environment File Reference](#environment-file-reference)
7. [LLM Setup (Ollama + Gemini)](#llm-setup-ollama--gemini)
8. [Google Search Grounding](#google-search-grounding)
9. [Verifying All Components](#verifying-all-components)
10. [Using the Web Application](#using-the-web-application)
11. [API Smoke Tests](#api-smoke-tests)
12. [Troubleshooting](#troubleshooting)
13. [Common Commands Reference](#common-commands-reference)

---

## What's New in v4.0

v4.0 is a complete **static data elimination** release. Every value shown to the farmer comes from a real API or live LLM reasoning — no hardcoded zone tables, no fallback crop lists, no templated market prices.

### Removed Static Data

| What was removed | Why | What replaced it |
|-----------------|-----|-----------------|
| `_ZONE_CLIMATE` (120-cell static climate table) | Gave identical weather to all of Brazil, all of Australia, etc. | **Open-Meteo Archive API** — real 2-year monthly averages per lat/lon |
| `_COUNTRY_TO_ZONE` (30 country→zone mappings) | Too coarse — California ≠ Florida | Gemini LLM + latitude-based fallback |
| `_ZONE_SOIL_DEFAULTS` (static soil per climate zone) | Every tropical location got the same soil | Gemini `_llm_enrich_fast()` + search-grounded upgrade |
| `_ZONE_MARKET_TEMPLATES` (templated market prices) | Showed fake prices with `{cur}` placeholders | Gemini Search Grounding — real prices or honest empty |
| `_COUNTRY_CURRENCY` (30-country currency map) | Had to be manually updated | LLM returns correct local currency automatically |
| `_COUNTRY_CROP_HINTS` (60+ country static hints) | Large static list, risked stale data | Dynamic prompt with LLM regional context |
| `_fallback_crops()` (600+ line crop tables) | Served misleading "zombie" crops when LLM failed | `_llm_simple_fallback()` → empty list on failure |
| Zone-based weather when Open-Meteo fails | Showed `25.0°C` for every location | `_llm_estimate_current_weather()` from coordinates |

### New Components

| Component | Description |
|-----------|-------------|
| **`_fetch_openmeteo_monthly_climatology()`** | Fetches 2-year real historical monthly data from Open-Meteo Archive API per lat/lon |
| **`_llm_generate_forecast()`** | Gemini generates 6-month forecast when archive API unavailable |
| **`_llm_estimate_current_weather()`** | Gemini estimates conditions from coordinates when live API fails |
| **`_llm_enrich_fast()`** | Fast single Gemini call (< 8s) for soil/market in parallel with weather fetch |
| **`/api/enrich-soil`** | Background endpoint called after render for richer search-grounded soil/market data |
| **Background soil update** | `_bgEnrichSoil()` in JS updates the soil card smoothly after analysis completes |

### Bugs Fixed

| Bug | Fix |
|-----|-----|
| Soil card always showing `Unknown` | `llm_future.result(timeout=3)` was too short; now uses `_llm_enrich_fast` (< 8s) with `timeout=20` |
| 7-month forecast (should be 6) | Fixed `_build_forecast_6month()` to iterate exactly 6 months |
| `25.0°C` hardcoded in streaming path | Removed; live weather temperature used |
| `null` values showing as `0.0°C` | `animateCount()` now shows `—` for null/undefined |
| `feels_like_c: null` showing `Feels ?°C` | Now shows `Current` label when null |
| ENSO step missing from streaming path | `climate_signal` now computed and included in `gathered` dict |
| `lon` not passed to `_build_forecast_6month()` | Fixed — Archive API requires both lat and lon |

---

## System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 13+, Ubuntu 22.04+ | Modern Python/package compatibility |
| Python | 3.8+ | 3.10 or 3.11 | FastAPI, Pydantic v2, ML libraries |
| RAM | 4 GB | 8 GB+ | LLaMA 3.2 uses ~3 GB |
| Storage | 2 GB | 6 GB+ | Packages (~1 GB) + LLaMA model (~2 GB) |
| Internet | **Required** | Stable broadband | Open-Meteo + Archive API + NOAA + Gemini |
| Browser | Chrome/Edge/Firefox (current) | Chrome or Edge | SSE streaming, fadeIn animations |
| Gemini API key | Strongly recommended | 2-3 keys | Soil, market prices, crop ranking |

> **v4.0 requires internet for almost all features** — weather, archive climatology, NOAA, and LLM enrichment all hit live APIs. An offline mode is not available.

> **Without a Gemini key**: soil shows `🤖 Analyzing...`, market prices show "data being gathered", crop recommendations rely solely on Ollama.

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/tirthch25/AI-Powered-Weather-Resilient-Crop-Advisor.git
cd AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
Copy-Item .env.example .env     # Windows
# cp .env.example .env          # macOS/Linux
# → Edit .env and add GEMINI_API_KEY=AIzaSy...

# 5. (Optional but recommended) Pull Ollama model
ollama pull llama3.2

# 6. Run the server
python run_website.py

# 7. Open browser
# http://localhost:8000
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

The setup script checks Python version, creates `.venv`, installs all packages, copies `.env.example` → `.env`, checks Ollama, and optionally starts the server.

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
- `scikit-learn==1.8.0` + `torch==2.9.0` + `xgboost==3.2.0` — ML models (legacy)
- `requests==2.32.5` — HTTP client for Open-Meteo, Archive API, NOAA
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

Edit `.env` with your values (see [Environment File Reference](#environment-file-reference) below).

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

All configuration lives in `agri_crop_recommendation/.env`.

### Full Reference Table

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `ollama` | Primary LLM: `ollama` or `gemini` |
| `OLLAMA_MODEL` | If Ollama | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | If Ollama | `http://localhost:11434` | Ollama API URL |
| `GEMINI_API_KEY` | Strongly recommended | _(blank)_ | Primary Gemini key — needed for soil, market, crops |
| `GEMINI_API_KEY_2` | Optional | _(blank)_ | Second key (auto-rotated on 429) |
| `GEMINI_API_KEY_3` | Optional | _(blank)_ | Third key |
| `GEMINI_API_KEY_4` | Optional | _(blank)_ | Fourth key |

### Configuration Modes

| Mode | `.env` Setup | Result |
|------|-------------|--------|
| **Best (recommended)** | `LLM_PROVIDER=ollama` + `GEMINI_API_KEY=...` | Full features: fast enrichment + search-grounded upgrades + local LLM chat |
| **Cloud-only** | `LLM_PROVIDER=gemini` + `GEMINI_API_KEY=...` | Machines without GPU; no local model |
| **Local-only** | `LLM_PROVIDER=ollama`, no Gemini key | Soil/market may show `Analyzing...`; crop ranking via Ollama |
| **No LLM** | Both blank, Ollama not running | Weather and forecast work; soil/market/crops unavailable |

### Example: Recommended Setup

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

GEMINI_API_KEY=AIzaSy...your_primary_key
GEMINI_API_KEY_2=AIzaSy...second_key
GEMINI_API_KEY_3=
GEMINI_API_KEY_4=
```

---

## LLM Setup (Ollama + Gemini)

### Ollama Setup (Local LLM)

Install from: `https://ollama.com/download`

**Windows:** Run the `.exe` installer — Ollama is added to PATH and starts automatically.

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:** Download the `.dmg`, open it — Ollama runs as a menu-bar app.

#### Pull a Model

```bash
ollama pull llama3.2      # Recommended — ~2 GB download
```

#### Verify Ollama is Running

```bash
ollama list
curl http://localhost:11434/api/tags
```

#### Model Options

| Model | Size | Use Case |
|-------|------|----------|
| `llama3.2` | ~2 GB | ✅ Recommended default |
| `gemma3:2b` | ~1.5 GB | Lightest for low-RAM machines |
| `llama3.1` | ~5 GB | Better reasoning, needs 8 GB+ RAM |
| `mistral` | ~4 GB | Good multilingual performance |

### Gemini API Key Setup

1. Open → [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"** → **"Create API key in new project"**
4. Copy the key (starts with `AIzaSy...`)
5. Paste into `.env` as `GEMINI_API_KEY=AIzaSy...`

**Free Tier Limits (as of 2025):**
- `gemini-2.0-flash-lite` → 1,500 requests/day, 15 req/min per key
- 3 keys → effectively 4,500 requests/day free

> **v4.0 relies on Gemini significantly more than v3.x** — soil enrichment, market prices, and crop ranking all use Gemini calls. Having at least one key is strongly recommended.

---

## Google Search Grounding

Search Grounding lets Gemini search the live web before answering. It enables:
- **Real-time crop market prices** in local currency
- **Active pest/disease alerts** for your region
- **Current government agricultural advisories**
- **Live climate anomaly reports**

**Requirements:**
- `GEMINI_API_KEY` set in `.env`
- A model that supports it: `gemini-2.0-flash`, `gemini-2.5-flash`, or `gemini-2.0-flash-001`

**Which components use it:**
1. `crop_agent.py` — Real-time crop advisories (step 2 of crop pipeline)
2. `/api/enrich-soil` — Background soil + market upgrade after dashboard renders
3. `climate_signals.py` — Active government crop advisories

**Fallback:** Search Grounding unavailable → plain Gemini → Ollama → empty (no static defaults).

---

## Verifying All Components

### 1. Server Health Check

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "version": "4.0",
  "llm_available": true,
  "llm_provider": "ollama",
  "gemini_keys": 2
}
```

### 2. Live Weather Test (Open-Meteo)

```bash
python -c "
from src.agents.data_gathering_agent import _fetch_openmeteo_current
result = _fetch_openmeteo_current(18.52, 73.86)
print(result)
"
```

Expected: `{'temperature_c': 31.5, 'humidity_pct': 68, ...}` — real live values.

### 3. Archive Climatology Test (Open-Meteo Archive API)

```bash
python -c "
from src.agents.data_gathering_agent import _fetch_openmeteo_monthly_climatology, _build_forecast_6month
import datetime
clim = _fetch_openmeteo_monthly_climatology(18.52, 73.86)
print('Months:', len(clim))
fc = _build_forecast_6month('India', 'MH', 18.52, 73.86, 31.5, datetime.datetime.now().month)
print('Forecast months:', len(fc))
"
```

Expected: `Months: 12`, `Forecast months: 6`

### 4. Fast LLM Enrichment Test

```bash
python -c "
from src.agents.data_gathering_agent import _llm_enrich_fast
result = _llm_enrich_fast('Pune, Maharashtra, India', 'India', 31.5, 6)
if result:
    print('Soil type:', result.get('soil', {}).get('type'))
    print('Market prices:', list(result.get('market_prices', {}).keys())[:3])
else:
    print('Enrichment failed — check GEMINI_API_KEY')
"
```

### 5. Background Enrich-Soil Endpoint

```bash
curl "http://localhost:8000/api/enrich-soil?district=Pune&state=Maharashtra&country=India&temp=31.5&month=6"
```

Expected:
```json
{
  "soil": {"type": "Clay-Loam", "ph": 6.5, "organic_matter": "Medium", "drainage": "Good"},
  "market_prices": {"Rice": "₹2,400/quintal", "Wheat": "₹2,100/quintal"},
  "district_summary": "Pune is a major agricultural district...",
  "climate_zone": "Subtropical"
}
```

### 6. Climate Signals (NOAA + AI)

```bash
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical"
```

### 7. Location APIs

```bash
curl http://localhost:8000/api/countries
curl http://localhost:8000/api/states/india
curl http://localhost:8000/api/districts/india/maharashtra
```

### 8. Streaming Analysis (Full Pipeline Test)

```bash
curl -X POST http://localhost:8000/api/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"country_code":"IN","country_name":"India","state_code":"MH","state_name":"Maharashtra","district":"Pune","lat":18.52,"lon":73.86,"irrigation":"Limited","planning_days":90}'
```

Watch for SSE events: `step 1` → `step 2` (live weather) → `step 3` (archive forecast) → `step 4` (soil/market) → `step 5` (crops) → `done`

---

## Using the Web Application

### Step-by-Step User Flow

1. **Select a country** — all 195 UN countries available
2. **Select a state** — AI generates the full state list for that country
3. **Select a district** — AI generates districts; coordinates are embedded in the response
4. **Set irrigation** — None / Limited / Full (affects crop scoring)
5. *(Optional)* **Provide soil texture and pH** — overrides AI-detected values
6. **Set planning period** in days (30 / 60 / 90 / 180)
7. **Click "Analyze with AI Agent"** — starts the 5-step streaming analysis
8. **Watch the streaming progress** — 5 steps with real-time messages:
   - Step 1: Location resolved (lat/lon)
   - Step 2: Live weather fetched (real temperature, humidity, rainfall)
   - Step 3: 6-month forecast built (from real archive data)
   - Step 4: AI soil and market analysis collected
   - Step 5: Crop ranking completed
9. **Review the dashboard:**
   - 🌡️ **Metric Cards** — Temp, humidity, rainfall, soil temp, wind, UV (null = `—`)
   - 🪨 **Soil Profile** — AI-detected type, pH, drainage (shows `🤖 Analyzing...` until ready)
   - 🌐 **Climate Intelligence Panel** — ENSO badge + 9-threat assessment
   - 📅 **6-Month Forecast Table** — Real archive-based temperature and rainfall
   - 💰 **Market Prices** — Real-time (search-grounded) or empty
   - 🌱 **Crop Rankings** — AI-ranked by suitability with reasons

> **Note on soil card:** After the dashboard loads, a **background request** fires to `/api/enrich-soil` using search-grounded Gemini. If the fast parallel enrichment returned `🤖 Analyzing...`, this background call will update the card within 10-20 seconds with richer data — no page refresh needed.

### Good First Tests

**India test:**
```
Country: India | State: Maharashtra | District: Pune
Irrigation: Limited | Planning: 90 days
```

**International test (global coverage):**
```
Country: Germany | State: Bavaria | District: Munich
Irrigation: Full | Planning: 150 days
```

**Africa test:**
```
Country: Nigeria | State: Kano | District: Kano Municipal
Irrigation: None | Planning: 60 days
```

---

## API Smoke Tests

```bash
# Health
curl http://localhost:8000/health

# Countries list
curl http://localhost:8000/api/countries

# States for India (LLM-generated)
curl http://localhost:8000/api/states/india

# Districts for Maharashtra (LLM-generated)
curl http://localhost:8000/api/districts/india/maharashtra

# Background soil enrichment
curl "http://localhost:8000/api/enrich-soil?district=Pune&state=Maharashtra&country=India&temp=31&month=6"

# Climate signals
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune"

# Climate signals (Europe)
curl "http://localhost:8000/climate-signals?country=germany&state=Bavaria&district=Munich&climate_zone=Temperate"

# Streaming analysis (full pipeline)
curl -X POST http://localhost:8000/api/analyze/stream \
  -H "Content-Type: application/json" \
  -d '{"country_code":"IN","country_name":"India","state_code":"MH","state_name":"Maharashtra","district":"Pune","lat":18.52,"lon":73.86,"irrigation":"Limited","planning_days":90}'
```

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `python` not found | Python not on PATH | Install Python 3.8+ and tick **"Add Python to PATH"** |
| PowerShell activation blocked | Execution policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails on torch | Old pip | `python -m pip install --upgrade pip`, then retry |
| `ModuleNotFoundError: src` | Wrong directory | Must run from inside `agri_crop_recommendation/` |
| Soil shows `🤖 Analyzing...` after 30s | No Gemini key or quota exceeded | Add `GEMINI_API_KEY` or `GEMINI_API_KEY_2` to `.env` |
| Market prices empty | LLM enrichment failed | Add Gemini key — market prices require Gemini Search Grounding |
| Crop recommendations empty | All LLM providers failed | Check Ollama is running: `ollama serve` + check Gemini key |
| `429 RESOURCE_EXHAUSTED` | Gemini quota exceeded | Add `GEMINI_API_KEY_2` and `GEMINI_API_KEY_3` to `.env` |
| Forecast shows 0 months | Open-Meteo Archive API down + no Gemini | Both data sources unavailable; retry later |
| Weather shows `—` everywhere | Open-Meteo current API failed + no Gemini | Retry later or add Gemini key for LLM weather estimate |
| `oni_value` always 0.0 | NOAA unreachable (firewall) | Normal fallback — Neutral phase assumed, no error raised |
| Port 8000 in use | Another server running | Stop it or run on alternate port (see below) |
| Ollama model missing | Model not pulled | `ollama pull llama3.2` |
| Search Grounding not working | Model doesn't support it | Requires `gemini-2.0-flash` or `gemini-2.5-flash` |
| `world_locations.json` not found | Wrong working directory | Run from `agri_crop_recommendation/` |

### Running on a Different Port

```bash
python -c "import uvicorn; uvicorn.run('src.api.app:app', host='0.0.0.0', port=8080)"
```

Then open `http://localhost:8080`.

### Checking Log Output

The server logs each agent step. Key messages to look for:

```
[DataAgent] Live weather OK for (18.52, 73.86): 31.5°C, 68% humidity, 15.2mm rain
[DataAgent] Climatology fetched for 18.52,73.86: 12 months
[DataAgent] Forecast built from Open-Meteo Archive: 6 months
[DataAgent] Fast enrich OK (gemini-2.0-flash-lite, key ...abc123)
[CropAgent] Gemini search-grounded OK (gemini-2.0-flash, key ...abc123) → 6 crops
[ClimateSignals] El Nino ONI=1.2 | heat=None drought=None frost=None cyclone=Cyclone
```

Warning messages that indicate degraded mode:

```
[DataAgent] Open-Meteo current API failed: <error>  ← Falls back to LLM estimate
[DataAgent] Open-Meteo Archive API failed: <error>  ← Falls back to LLM forecast
[DataAgent] Fast enrich failed for Pune, Maharashtra, India  ← Soil = Analyzing...
[CropAgent] All LLM providers failed — returning empty  ← No crops shown
```

---

## Common Commands Reference

```bash
# Start the server
python run_website.py

# Test data pipeline
python -c "
from src.agents.data_gathering_agent import _fetch_openmeteo_monthly_climatology
import datetime
c = _fetch_openmeteo_monthly_climatology(18.52, 73.86)
print('Archive months:', len(c) if c else 'FAILED')
"

# Test fast LLM enrichment
python -c "
from src.agents.data_gathering_agent import _llm_enrich_fast
r = _llm_enrich_fast('Pune, Maharashtra, India', 'India', 31, 6)
print('Soil:', r.get('soil') if r else 'FAILED')
"

# Test import health
python -c "
from src.agents.data_gathering_agent import gather_location_data, _llm_enrich_fast
from src.agents.crop_agent import recommend_crops_agent
from src.api.app import app
print('All imports OK')
"

# Check Ollama
ollama list
ollama pull llama3.2

# Health endpoint
curl http://localhost:8000/health

# Background soil enrichment
curl "http://localhost:8000/api/enrich-soil?district=Pune&state=Maharashtra&country=India&temp=31&month=6"

# Climate signals
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune"

# Full streaming analysis (PowerShell)
Invoke-WebRequest -Uri "http://localhost:8000/api/analyze/stream" -Method POST `
  -ContentType "application/json" `
  -Body '{"country_code":"IN","country_name":"India","state_code":"MH","state_name":"Maharashtra","district":"Pune","lat":18.52,"lon":73.86,"irrigation":"Limited","planning_days":90}'
```

---

*End of Setup Guide — AI Powered Weather Resilient Crop Advisor v4.0*  
*Prepared by Tirth Chankeshwara | CDAC-Pune | June 2026*
