# AI Powered Weather Resilient Crop Advisor v3.1 — Setup Guide

This guide takes the project from a fresh clone to a running local global dashboard. It explains the LLM setup, new Climate Signal Intelligence feature, verification steps, common failure modes, and what each setup choice changes.

## Setup Map

```text
Install Python
  -> Clone or open repository
  -> Create virtual environment
  -> Install dependencies
  -> Configure .env
  -> Recommended: install Ollama and pull llama3.2
  -> (Optional) Verify NOAA climate data is reachable
  -> Verify models and API
  -> Start web app
```

## What's New in v3.1

v3.1 adds **Climate Signal Intelligence** on top of the v3.0 Global Dashboard:

| Feature | Detail |
| --- | --- |
| **ENSO / El Niño Detection** | Automatically fetches the NOAA CPC Oceanic Niño Index (ONI) every 6 hours |
| **No New API Key** | NOAA data is free and public — no registration needed |
| **Forecast Adjustment** | 6-month rainfall and temperature forecasts are adjusted per ENSO phase and climate zone |
| **AI Climate Summary** | Existing Gemini/LLaMA agent interprets the signal for the farmer's specific location |
| **Climate Intelligence Panel** | New dashboard panel showing ENSO phase badge, alert level, rainfall/temp outlook, crop risks, and AI advice |
| **New API Endpoint** | `GET /climate-signals` — callable independently |

---

## Requirements

| Component | Minimum | Recommended | Why It Matters |
| --- | --- | --- | --- |
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 13+, Ubuntu 22.04+ | Modern Python and package support |
| Python | 3.8+ | 3.10 or 3.11 | FastAPI, ML, and Pydantic compatibility |
| RAM | 4 GB | 8 GB+ | Local LLM and model workflows need memory |
| Storage | 2 GB | 6 GB+ | Python packages, trained models, Ollama model |
| Internet | Required for first setup | Stable broadband | Open-Meteo live weather + NOAA ENSO data |
| Browser | Current Chrome, Edge, Firefox, or Safari | Current Chrome or Edge | Dashboard and SSE streaming responses |
| Ollama / Gemini | Highly Recommended | Required for v3.1 | Core agentic recommendations + ENSO interpretation |

> **Note on LLMs:** v3.1 relies on LLMs for core crop filtering, risk assessment, and ENSO interpretation. While the system degrades gracefully without LLMs, the full climate-aware experience requires either Ollama or a Gemini key.

> **Note on NOAA:** If NOAA is unreachable (offline, firewall, etc.) the Climate Signal step is silently skipped and the system assumes Neutral conditions — no error is raised.

---

## Option A: One-Command Setup

Run this from the repository root.

### Windows

```powershell
.\setup.bat
```

### macOS / Linux

```bash
bash setup.sh
```

The setup script checks Python, creates `agri_crop_recommendation/.venv`, installs packages, prepares `.env`, checks Ollama, verifies important files, and can start the server.

---

## Option B: Manual Setup

```bash
cd agri_crop_recommendation
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the app:

```bash
python run_website.py
```

Open:

```text
http://localhost:8000
```

---

## Environment File

Create `.env` inside `agri_crop_recommendation/`:

```bash
cp .env.example .env
```

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Recommended values:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

GEMINI_API_KEY=

# NOAA climate data is fetched automatically — no key required
```

Use these modes:

| Mode | `.env` Setup | Best For |
| --- | --- | --- |
| Local-first | `LLM_PROVIDER=ollama` | Private local chat, ENSO interpretation, and fast enrichment |
| Cloud fallback | `LLM_PROVIDER=ollama` plus `GEMINI_API_KEY` | Resilience if Ollama is stopped or slow |
| Cloud-only | `LLM_PROVIDER=gemini` plus `GEMINI_API_KEY` | Machines that cannot run a local model |
| No LLM (Legacy) | Leave keys blank and do not run Ollama | Falls back to v2.x rule-based engine; ENSO fetch still runs but AI interpretation is minimal |

---

## Ollama Setup

Install Ollama from:

```text
https://ollama.com/download
```

Pull the default model:

```bash
ollama pull llama3.2
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Check installed models:

```bash
ollama list
```

Try the model:

```bash
ollama run llama3.2
```

Model options:

| Model | Typical Use |
| --- | --- |
| `llama3.2` | Recommended default balance |
| `gemma3:2b` | Lighter local model for low-resource machines |
| `llama3.1` | Larger model with stronger responses |

After changing `OLLAMA_MODEL`, pull that model and restart the app.

---

## Verifying the Climate Signal Feature

After starting the server, test the NOAA endpoint directly:

```bash
# Quick ENSO status check
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune&climate_zone=Subtropical"
```

Expected response shape:

```json
{
  "climate_signals": {
    "enso_phase": "Neutral",
    "enso_strength": "Neutral",
    "oni_value": 0.12,
    "phase_label": "🟢 Neutral (Normal)",
    "ai_interpretation": {
      "summary": "...",
      "crop_risks": ["..."],
      "opportunity": "...",
      "alert_level": "None",
      "rainfall_outlook": "Near Normal",
      "temp_outlook": "Near Normal"
    },
    "forecast_adjustments": {
      "rainfall_factor": 1.0,
      "temp_offset_c": 0.0,
      "description": "No adjustment applied (Neutral conditions)"
    },
    "source": "NOAA CPC (free, no API key)"
  }
}
```

If NOAA is unreachable the response will still succeed but `oni_value` will be `0.0` and `enso_phase` will be `"Neutral"`.

---

## Run And Verify

Start the server:

```bash
python run_website.py
```

Health check:

```bash
curl http://localhost:8000/health
```

Browser checks:

```text
http://localhost:8000                       ← Web Interface + Climate Panel
http://localhost:8000/docs                  ← Interactive Swagger API Docs
http://localhost:8000/health                ← System status
http://localhost:8000/climate-signals?country=india  ← ENSO status
```

Expected health response includes:

- `status`
- `version`
- `regions_loaded`
- `ml_models`
- `llm_available`
- `llm_provider`
- `ollama_running`
- `timestamp`

---

## Using The Web App

1. Select a country (e.g., India, USA, Brazil).
2. Select a state or province.
3. Select a district or region.
4. Choose irrigation.
5. Optionally provide soil texture and pH.
6. Set the planning period.
7. Click **Analyze with AI Agent**.
8. Watch the streaming SSE progress steps.
9. Review weather, soil, **Climate Intelligence Panel**, forecast, market, and crop ranking cards.
10. Ask a follow-up question in the AI farmer chat.

The **Climate Intelligence Panel** appears automatically between the Soil card and the 6-Month Forecast. It shows:
- 🔴 El Niño / 🔵 La Niña / 🟢 Neutral phase badge with ONI value
- ⚠️ Alert bar (if conditions are active)
- AI-generated 2-sentence impact summary for your location
- Rainfall outlook, Temperature outlook, Forecast adjustment description
- Crop risks specific to the ENSO phase
- 💡 One farming action recommendation
- Data timestamp and source attribution

Good first test:

```text
Country: India
State: Maharashtra
District: Pune
Irrigation: Limited
Planning period: 90
Soil: Loam, pH 6.8
```

---

## API Smoke Tests

With the server running:

```bash
curl http://localhost:8000/api/countries
curl http://localhost:8000/health

# New in v3.1 — Climate Signals
curl "http://localhost:8000/climate-signals?country=india&state=Maharashtra&district=Pune"
```

**v3.1 Main Global Endpoint:**

```text
POST /api/analyze/stream
```

Streams Server-Sent Events with live weather, ENSO-adjusted forecast, and ranked crop recommendations.

**Climate Signals Endpoint (NEW):**

```text
GET /climate-signals?country=india&state=<state>&district=<district>&climate_zone=Subtropical
```

**Legacy v2.x Fallback Endpoint:**

```powershell
curl -X POST http://localhost:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"region_id\":\"MH_PUNE\",\"irrigation\":\"Limited\",\"planning_days\":90}"
```

---

## Model And Data Checks

Run these from `agri_crop_recommendation/`:

```bash
# v3.1 Tests
python scripts/test_api.py
python scripts/test_chatbot.py

# Legacy ML Test
python scripts/verify_models.py
```

Important v3.1 Core files:

```text
data/reference/
|-- world_locations.json  (Global coordinate resolver)
`-- crop_knowledge.json   (Core LLM Context Database)

src/services/
`-- climate_signals.py    (NEW — NOAA ENSO fetch + AI interpretation)

src/agents/
`-- data_gathering_agent.py  (Updated — Step 3 now includes ENSO adjustment)
```

Legacy files (Maintained for Fallback / API consumers):

```text
models/                   (v2.x Legacy Machine Learning)
|-- crop_suitability/
|-- weather_lstm/
`-- weather_xgboost/

data/reference/
|-- regions.json          (v2.x India-only static database)
`-- regional_crops.json
```

---

## Troubleshooting

| Problem | Cause | Fix |
| --- | --- | --- |
| `python` not found | Python not on PATH | Install Python 3.8+ and enable `Add Python to PATH` |
| PowerShell activation blocked | Execution policy | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails | Old pip or interrupted install | `python -m pip install --upgrade pip`, then retry |
| `ModuleNotFoundError: src` | Command run from wrong folder | Run from `agri_crop_recommendation/` |
| Chat / Recommendations unavailable | Ollama stopped or no Gemini key | Start `ollama serve` or add `GEMINI_API_KEY` |
| Ollama model missing | Model was not pulled | `ollama pull llama3.2` |
| Climate panel not showing | NOAA unreachable or JS error | Check browser console; NOAA fetch is best-effort and silently falls back |
| Weather unavailable | Open-Meteo API/network issue | Retry later; fallback zone estimates will be used |
| Port 8000 in use | Another server is running | Stop it or run uvicorn on another port |

Run on another port:

```bash
python -c "import uvicorn; uvicorn.run('src.api.app:app', host='0.0.0.0', port=8080)"
```

Then open `http://localhost:8080`.

---

## Common Commands

```bash
# Start app
python run_website.py

# Run API checks
python scripts/test_api.py

# Test chat integration
python scripts/test_chatbot.py

# Test ENSO endpoint
curl "http://localhost:8000/climate-signals?country=india"

# Check installed Ollama models
ollama list

# Pull default LLM
ollama pull llama3.2
```

---

## What To Do After Setup

- Try two districts in entirely different countries (e.g., USA vs. India) and compare recommendations.
- Check the **Climate Intelligence Panel** — observe whether ENSO adjustments change the 6-month rainfall forecast values compared to the baseline.
- During an active El Niño year, switch irrigation from `None` to `Full` and observe how drought-tolerant crops rise in ranking.
- Change irrigation from `None` to `Full` and observe crop ranking changes.
- Override soil pH and texture to test sensitivity.
- Ask the chat assistant: *"How does the current El Niño affect my wheat crop?"*
- Open `/docs` and inspect the `/climate-signals` request model.
