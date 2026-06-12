# AI Powered Weather Resilient Crop Advisor v3.0 — Setup Guide

This guide takes the project from a fresh clone to a running local global dashboard. It also explains the LLM setup, verification steps, common failure modes, and what each setup choice changes.

## Setup Map

```text
Install Python
  -> Clone or open repository
  -> Create virtual environment
  -> Install dependencies
  -> Configure .env
  -> Recommended: install Ollama and pull llama3.2
  -> Verify models and API
  -> Start web app
```

## Requirements

| Component | Minimum | Recommended | Why It Matters |
| --- | --- | --- | --- |
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 13+, Ubuntu 22.04+ | Modern Python and package support |
| Python | 3.8+ | 3.10 or 3.11 | FastAPI, ML, and Pydantic compatibility |
| RAM | 4 GB | 8 GB+ | Local LLM and model workflows need memory |
| Storage | 2 GB | 6 GB+ | Python packages, trained models, Ollama model |
| Internet | Required for first setup | Stable broadband | Dependencies, live Open-Meteo weather API |
| Browser | Current Chrome, Edge, Firefox, or Safari | Current Chrome or Edge | Dashboard and SSE streaming responses |
| Ollama / Gemini | Highly Recommended | Required for v3.0 | Core agentic recommendations rely on LLM logic |

> **Note on LLMs:** v3.0 of this platform heavily relies on LLMs for core crop filtering and risk assessment. While it can run without API keys or Ollama (falling back to legacy v2.x rules), an LLM is required to experience the full Global Agentic architecture.

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

Choose this path when you want the least manual work.

## Option B: Manual Setup

Use this path when you want to see each step.

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
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
```

Use these modes:

| Mode | `.env` Setup | Best For |
| --- | --- | --- |
| Local-first | `LLM_PROVIDER=ollama` | Private local chat and fast local enrichment |
| Cloud fallback | `LLM_PROVIDER=ollama` plus `GEMINI_API_KEY` | Resilience if Ollama is stopped or slow |
| Cloud-only | `LLM_PROVIDER=gemini` plus `GEMINI_API_KEY` | Machines that cannot run a local model |
| No LLM (Legacy) | Leave keys blank and do not run Ollama | Falls back to the legacy v2.x rule-based engine (Not Recommended for v3.0) |

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
| `llama3.1` | Larger model with stronger responses, if your machine can handle it |

After changing `OLLAMA_MODEL`, pull that model and restart the app.

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
http://localhost:8000
http://localhost:8000/docs
http://localhost:8000/health
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

## Using The Web App

1. Select a country (e.g., India, USA, Brazil).
2. Select a state or province.
3. Select a district or region.
4. Choose irrigation.
5. Optionally provide soil texture and pH.
6. Set the planning period.
7. Click `Analyze with AI Agent`.
8. Watch the streaming Server-Sent Events (SSE) progress steps.
9. Review weather, soil, forecast, market, and crop ranking cards.
10. Ask a follow-up question in the AI farmer chat.

Good first test:

```text
Country: India
State: Maharashtra
District: Pune
Irrigation: Limited
Planning period: 90
Soil: Loam, pH 6.8
```

## API Smoke Tests

With the server running:

```bash
curl http://localhost:8000/api/countries
curl http://localhost:8000/health
```

**v3.0 Main Global Endpoint:**
The web UI primarily uses the streaming agentic endpoint.
```text
POST /api/analyze/stream
```
That endpoint streams Server-Sent Events so the user sees progress while live weather and multi-agent LLM reasoning work is running.

**Legacy v2.x Fallback Endpoint:**
```powershell
curl -X POST http://localhost:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"region_id\":\"MH_PUNE\",\"irrigation\":\"Limited\",\"planning_days\":90}"
```

For macOS/Linux, use backslashes instead of `^`:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"region_id":"MH_PUNE","irrigation":"Limited","planning_days":90}'
```

## Model And Data Checks

Run these from `agri_crop_recommendation/`:

```bash
# v3.0 Tests
python scripts/test_api.py
python scripts/test_chatbot.py

# Legacy ML Test
python scripts/verify_models.py
```

Important v3.0 Core files:

```text
data/reference/
|-- world_locations.json  (Global coordinate resolver)
`-- crop_knowledge.json   (Core LLM Context Database)
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

Retraining scripts live in `scripts/`. Note that in v3.0, retraining the ML models is not strictly necessary for global coverage because the Data Gathering Agent handles forecasting via live Open-Meteo anchoring.

## Troubleshooting

| Problem | Cause | Fix |
| --- | --- | --- |
| `python` not found | Python is missing or not on PATH | Install Python 3.8+ and enable `Add Python to PATH` |
| PowerShell activation blocked | Execution policy restriction | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `pip install` fails | Old pip or interrupted install | `python -m pip install --upgrade pip`, then retry |
| `ModuleNotFoundError: src` | Command run from wrong folder | Run from `agri_crop_recommendation/` |
| `ModuleNotFoundError: pydantic` | Dependencies missing in active Python | Activate `.venv` and run `pip install -r requirements.txt` |
| Chat / Recommendations unavailable | Ollama stopped or no Gemini key | Start `ollama serve` or add `GEMINI_API_KEY` |
| Ollama model missing | Model was not pulled | `ollama pull llama3.2` |
| Weather unavailable | API/network issue | Retry later; fallback estimates may be used |
| Port 8000 in use | Another server is running | Stop it or run uvicorn on another port |

Run on another port:

```bash
python -c "import uvicorn; uvicorn.run('src.api.app:app', host='0.0.0.0', port=8080)"
```

Then open:

```text
http://localhost:8080
```

## Common Commands

```bash
# Start app
python run_website.py

# Run API checks
python scripts/test_api.py

# Test chat integration
python scripts/test_chatbot.py

# Check installed Ollama models
ollama list

# Pull default LLM
ollama pull llama3.2
```

## What To Do After Setup

- Try two districts in entirely different countries (e.g., USA vs. India) and compare recommendations.
- Change irrigation from `None` to `Full` and observe crop ranking changes.
- Override soil pH and texture to test sensitivity.
- Ask the chat assistant why the top crop is safer than the second crop.
- Open `/docs` and inspect the request models.
