# 🌾 Indian Farmer Crop Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?style=for-the-badge&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-Weather-007ACC?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2%20Local-8A2BE2?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered, season-aware crop recommendation platform for Indian farmers — covering all 640 districts across 28 states and 8 union territories.**

[Overview](#-overview) • [How It Works](#-how-it-works) • [Architecture](#-system-architecture) • [Features](#-features) • [Installation](#-installation) • [API Docs](#-api-endpoints) • [Tech Stack](#-tech-stack)

</div>

---

## 🧭 Overview

India's agriculture sector faces a critical information gap: farmers make crop selection decisions based on tradition or limited local knowledge, without access to the data-driven insights available to large agribusinesses. The result is suboptimal crop choices, water wastage, and unnecessary exposure to risk.

The **Indian Farmer Crop Recommendation System** bridges this gap with an end-to-end AI-powered advisory platform. A farmer inputs their district, soil type, and irrigation availability — the system responds with ranked crop recommendations backed by:

- **Live real-time weather** fetched per district from Open-Meteo
- **Machine learning forecasts** (LSTM + XGBoost) for the upcoming 90 days
- **District-specific crop approval** from an LLM-enriched database covering all 640 Indian agricultural districts
- **Explainable AI** — each recommendation comes with a farmer-friendly explanation in English, Hindi, or Marathi

The system uses **LLaMA 3.2 (via Ollama)** as its primary LLM — running entirely locally, free, and offline. Google Gemini is supported as an automatic fallback. All LLM features degrade gracefully when neither is configured.

---

## 💡 Project Idea

Most crop advisory tools either rely on simple lookup tables (season → crop list) or require expensive IoT sensors. This system takes a different approach: combine **publicly available free data sources** with **multiple ML models** to replicate the advice a skilled agronomist would give, at scale, for every district in India.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **District-level accuracy** | Open-Meteo API called with each district's exact lat/lon — Leh (3524m) gets 8°C, not the plains average |
| **No black-box recommendations** | Every score is a transparent 6-factor weighted sum + ML blend; farmers see why a crop was recommended |
| **Graceful degradation** | Remove weather API → zone averages kick in. Remove LLM → rule-based scoring continues. Zero single points of failure |
| **Pre-computed enrichment** | LLM runs once per district offline; results cached in `regional_crops.json`. Runtime recommendations need zero LLM calls for filtering |
| **Nationwide completeness** | Every one of India's 640 agricultural districts has a region profile, soil default, climate zone, and crop suitability data |
| **Free & private LLM** | LLaMA 3.2 runs locally via Ollama — no API costs, no data sent to cloud, no quota limits |

---

## 🏗️ System Architecture

The system follows a **layered pipeline architecture** where each stage enriches the data before passing it to the next:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│       Web Browser ←→ index.html + app.js + style.css               │
│       Bilingual UI (Hindi / English) · Chart.js visualizations      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTP / REST (FastAPI)
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API LAYER                                     │
│  POST /recommend · GET /weather/now · POST /chat/stream             │
│  GET /forecast · POST /risk-assessment · GET /pest-warnings          │
└──┬──────────────┬────────────┬────────────┬───────────┬─────────────┘
   │              │            │            │           │
┌──▼──────┐ ┌────▼────┐ ┌─────▼────┐ ┌────▼───┐ ┌────▼──────┐
│ Weather │ │  Crop   │ │  Risk    │ │  Pest  │ │ Planting  │
│ Engine  │ │  Reco.  │ │ Assess.  │ │Warning │ │ Calendar  │
│LSTM+XGB │ │Engine   │ │ Engine   │ │ System │ │           │
└──┬──────┘ └────┬────┘ └──────────┘ └────────┘ └───────────┘
   │              │
┌──▼──────────────▼──────────────────────────────────────────────────┐
│                     LLM LAYER                                       │
│  Primary:  LLaMA 3.2 via Ollama (local, free, private)             │
│  Fallback: Google Gemini (if Ollama not running)                    │
│  Uses: llm_chat.py · llm_explainer.py · llm_filter.py              │
└──┬──────────────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  Crop DB (50+ crops) · Region Data (640 districts)                  │
│  District Weather (Parquet) · regional_crops.json (LLM-enriched)   │
│  crop_knowledge.json · Zone Climate Normals (CSV)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommendation Pipeline (Step by Step)

```
Farmer Input (Region / GPS + Soil + Irrigation)
        │
        ▼
[1] Resolve Region
        RegionManager looks up the district profile from regions.json.
        If GPS coordinates are provided, finds the nearest district
        center within 150 km using the Haversine formula.
        │
        ▼
[2] Fetch Live Weather
        Open-Meteo API called with the district's exact lat/lon.
        Returns last 7 days of real weather (temp_max, temp_min, rainfall).
        Humidity enriched from zone historical averages.
        │
        ▼
[3] Detect Season
        Current date + region → Kharif (Jun–Sep) / Rabi (Oct–Feb) / Zaid (Mar–May).
        │
        ▼
[4] Agricultural Feature Engineering
        Adds: temp_avg, GDD, dry_spell_days, 7-day rolling rainfall.
        │
        ▼
[5] ML Weather Forecast (17–90 days)
        LSTM (PyTorch) + XGBoost ensemble → 7-day ahead forecast.
        Days 15–90 filled with zone climatology blended with ML trend.
        │
        ▼
[6] Regional Enrichment Gate
        Checks regional_crops.json (pre-computed by LLM) for this district.
        If found → filters to approved crops with district-specific scores.
        If not found → falls back to runtime LLaMA/Gemini LLM filter, then static zone scores.
        │
        ▼
[7] Score All Crops (6-factor rule-based + 60:40 ML blend)
        ├── Temperature Compatibility (25%)
        ├── Water Availability (25%)
        ├── Soil Compatibility (15%)
        ├── Regional Suitability (15%)
        ├── Seasonal Adjustment (10%)
        └── Drought Tolerance Bonus (10%)
        │
        ▼
[8] Risk Assessment per Crop
        Drought risk · Temperature stress · Extreme weather events
        │
        ▼
[9] Pest & Disease Warnings
        Weather-triggered alerts from crop_knowledge.json pest database
        │
        ▼
[10] Planting Calendar
        Sowing → Germination → Vegetative → Flowering → Harvest dates
        │
        ▼
[11] LLM Explainer (LLaMA 3.2 / Gemini fallback)
        Top 3 crops get a 2-sentence farmer-friendly explanation
        in English, Hindi, or Marathi
        │
        ▼
     Top 15 Crops Ranked
     Score · Risk · Pest Alerts · Calendar · AI Explanation
```

---

## ✨ Features

### 🗺️ Nationwide District Coverage
- **640 Indian Agricultural Districts** across 28 states and 8 union territories
- Region IDs follow `<STATE_CODE>_<DISTRICT>` format (e.g., `MH_PUNE`, `UP_LUCKNOW`)
- GPS coordinate lookup via Haversine nearest-neighbor (150 km radius)

### 🌡️ Accurate Real-Time Weather
- Open-Meteo API called with each district's **exact latitude/longitude**
- Live temperature displayed in the UI and injected into AI chat context

### 🤖 Machine Learning Models

| Model | Framework | Purpose | Performance |
|-------|-----------|---------|-------------|
| **LSTM Weather Forecaster** | PyTorch (2-layer, hidden=128) | 7-day ahead weather forecast | RMSE 0.4502 on 455 districts |
| **XGBoost Weather Forecaster** | XGBoost | temp_max, temp_min, rainfall | 74 lag & rolling features |
| **Random Forest Crop Suitability** | Scikit-learn | Suitability score 0–100 | 60% ML : 40% rule-based blend |

> The Random Forest model is trained on **44 million rows** of crop-region-weather-soil combinations.

### 🦙 LLM Integration (LLaMA 3.2 + Gemini Fallback)

| LLM Component | File | Purpose |
|---------------|------|---------|
| **Crop Filter** | `llm_filter.py` | Runtime filter for unenriched districts |
| **Crop Explainer** | `llm_explainer.py` | 2-sentence explanations (English / Hindi / Marathi) |
| **Farmer Chat** | `llm_chat.py` | Streaming Q&A grounded in live weather + district context |

**Provider priority chain:**
```
1. LLaMA 3.2 via Ollama  (local, free, no internet needed)  ← default
2. Google Gemini          (automatic fallback if Ollama is down)
3. Graceful degradation   (rule-based scoring continues if both unavailable)
```

> **Note:** LLM is **not used for crop scoring** — the Random Forest model handles that. LLM is only used for chat, explanations, and filtering.

### 🌱 Crop Database (50+ Crops)

| Category | Crops |
|----------|-------|
| **Millets** | Bajra, Jowar, Ragi, Foxtail Millet |
| **Pulses** | Moong, Urad, Cowpea, Guar, Soybean |
| **Oilseeds** | Sesame, Sunflower |
| **Vegetables** | Tomato, Brinjal, Okra, Bottle Gourd, Cucumber, Ridge Gourd, Bitter Gourd, Pumpkin, Green Chilli |
| **Leafy Greens** | Spinach, Fenugreek, Coriander, Amaranth, Lettuce, Mustard Greens |
| **Root Vegetables** | Carrot, Radish, Beetroot, Turnip |

### 🧪 Suitability Scoring Engine

| Factor | Weight | How it's computed |
|--------|--------|-------------------|
| Temperature Compatibility | 25% | 3-zone model: optimal (100) → survival range (60–100) → grace margin (0–20) → beyond (0) |
| Water Availability | 25% | Expected rainfall + irrigation buffer vs. crop water requirement |
| Soil Compatibility | 15% | Texture, pH, drainage, organic matter vs. crop requirements |
| Regional Suitability | 15% | Score from `regional_crops.json` or zone-based fallback |
| Seasonal Adjustment | 10% | Whether the crop is ideal for the detected season |
| Drought Tolerance Bonus | 10% | Bonus for drought-tolerant crops during predicted dry spells |

### 🛡️ Risk Assessment, 🐛 Pest Warnings, 📅 Planting Calendar
- Per-crop risk scoring (Drought 40% · Temperature Stress 35% · Extreme Events 25%)
- Weather-triggered pest/disease alerts for 50+ crops
- Sowing-to-harvest milestone timeline with phase-specific care tips

### 💬 AI Farming Chat
Streaming LLaMA 3.2 chat grounded in the farmer's district, current season, and live weather. Farmers can ask free-form questions in English. The model is warmed up at app startup — first response is instant.

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- [Ollama](https://ollama.com/download) *(recommended — free local LLM)*
- A free **[Google Gemini API key](https://aistudio.google.com/app/apikey)** *(optional — used as fallback if Ollama is not running)*

### 1. Clone the Repository
```bash
git clone https://github.com/tirthch25/Indian-Farmer-Crop-Recommendation-System.git
cd Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation
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
```powershell
# Download and install from https://ollama.com/download, then:
ollama pull llama3.2        # ~2 GB download, one-time
ollama serve                # Start the local server

# Or run the automated setup script:
.\scripts\setup_ollama.ps1
```

> **Alternative models:** `gemma3:2b` (1.6 GB, lightest) · `llama3.1` (4.7 GB, best quality)

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```env
# Primary LLM provider (ollama = local LLaMA, free & private)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Optional fallback — Gemini is used automatically if Ollama is down
GEMINI_API_KEY=your_gemini_api_key_here
```

### 6. (Recommended) Pre-Compute District Enrichment
```bash
python scripts/enrich_regional_crops.py --only-missing
```
> Runs LLaMA/Gemini once per district and caches to `data/reference/regional_crops.json`. Safe to stop and resume.

### 7. (Optional) Fetch District Weather History & Train ML Models
```bash
python scripts/fetch_district_weather.py   # Download 10yr weather for 640+ districts
python scripts/train_model.py --model all  # Train RF + LSTM + XGBoost
```

### 8. Start the Server
```bash
python run_website.py
```

### 9. Open in Browser
```
http://localhost:8000          ← Web Interface
http://localhost:8000/docs     ← Interactive Swagger API Docs
http://localhost:8000/health   ← System status (LLM provider, ML models)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/recommend` | Generate ranked crop recommendations (top 15) |
| `GET` | `/weather/now/{region_id}` | Live real-time temperature for a district |
| `GET` | `/forecast/{region_id}?days=N` | ML weather forecast (LSTM + XGBoost ensemble) |
| `POST` | `/risk-assessment` | Drought / temperature / extreme event risk per crop |
| `GET` | `/pest-warnings/{region_id}` | Weather-triggered pest & disease alerts |
| `GET` | `/planting-calendar/{crop_id}` | Sowing-to-harvest calendar with care tips |
| `POST` | `/chat/stream` | Streaming AI farming Q&A (LLaMA 3.2, context-grounded) |
| `GET` | `/regions` | All 640+ district profiles |
| `GET` | `/health` | API health + ML model status + LLM provider info |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 📁 Project Structure

```
agri_crop_recommendation/
│
├── src/
│   ├── api/
│   │   └── app.py                  # FastAPI — all REST endpoints + LLM warmup
│   ├── crops/
│   │   ├── database.py             # 50+ crop knowledge base
│   │   ├── models.py               # CropInfo dataclass
│   │   └── soil.py                 # SoilInfo model + compatibility scoring
│   ├── ml/
│   │   ├── pipeline.py             # Feature engineering
│   │   ├── predictor.py            # Random Forest crop suitability (44M rows)
│   │   ├── lstm_weather.py         # PyTorch LSTM weather forecaster
│   │   └── xgboost_weather.py      # XGBoost weather forecaster
│   ├── services/
│   │   ├── recommender.py          # Core recommendation + scoring engine
│   │   ├── risk.py                 # Risk assessment engine
│   │   ├── pests.py                # Pest & disease warning system
│   │   ├── calendar.py             # Planting calendar generator
│   │   ├── llm_filter.py           # LLaMA/Gemini runtime crop filter
│   │   ├── llm_explainer.py        # LLaMA/Gemini crop explanation generator
│   │   └── llm_chat.py             # LLaMA/Gemini streaming farming chat
│   ├── utils/
│   │   ├── regions.py              # RegionManager — 640+ districts
│   │   └── seasons.py              # Season detection & transition logic
│   └── weather/
│       ├── fetcher.py              # Open-Meteo live weather client
│       ├── forecast.py             # ML ensemble forecast (17–90 days)
│       └── history.py              # Zone-level climate normals
│
├── data/
│   ├── reference/
│   │   ├── regions.json            # 640+ district profiles
│   │   ├── regional_crops.json     # LLM-enriched district→crop approvals
│   │   └── crop_knowledge.json     # Growth phases, pest DB, planting windows
│   └── ml/training/
│       └── crop_suitability/
│           └── crop_suitability_data.csv   # 44M rows training data (4.16 GB)
│
├── models/
│   ├── crop_suitability/           # rf_model.joblib + label_encoders.joblib
│   ├── weather_lstm/               # lstm_weights.pt + metadata.json
│   └── weather_xgboost/            # temp_max/min/rainfall .joblib
│
├── scripts/
│   ├── enrich_regional_crops.py    # LLM district enrichment (run once)
│   ├── fetch_district_weather.py   # Download 10yr weather data
│   ├── train_model.py              # Train RF + XGBoost + LSTM
│   ├── setup_ollama.ps1            # One-shot Ollama install + model pull
│   ├── test_llama_integration.py   # LLaMA integration smoke tests
│   └── test_api.py                 # API smoke tests
│
├── templates/index.html            # Bilingual single-page web UI
├── static/css/style.css
├── static/js/app.js
├── .env.example                    # Environment variable template
├── run_website.py                  # Server startup script
└── requirements.txt
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript · Chart.js |
| **Deep Learning** | PyTorch — 2-layer LSTM weather forecaster |
| **Gradient Boosting** | XGBoost — weather forecasting |
| **Classical ML** | Scikit-learn — Random Forest crop suitability (44M rows) |
| **Primary LLM** | **LLaMA 3.2** via **Ollama** (local, free, private) |
| **Fallback LLM** | Google Gemini (`gemini-2.5-flash`) via `google-genai` SDK |
| **Weather** | Open-Meteo API — free, no API key, per-district lat/lon |
| **Storage** | Apache Parquet · JSON · CSV |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence** | joblib (RF/XGBoost) · torch.save (LSTM) |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10 or 3.11 |
| RAM | 4 GB | 8 GB (for LSTM/XGBoost training) |
| Storage | 600 MB | 5 GB (with full weather data + LLaMA model) |
| Ollama model | `gemma3:2b` (3 GB RAM) | `llama3.2` (4 GB RAM) |
| Internet | Required (weather) | Stable broadband |
| Gemini Key | Optional (fallback) | Free at [aistudio.google.com](https://aistudio.google.com/app/apikey) |

---

## 🙏 Acknowledgements

- **[Open-Meteo](https://open-meteo.com/)** — Free, open-source weather API
- **[Ollama](https://ollama.com/)** — Local LLM runtime for LLaMA/Gemma
- **[Meta LLaMA 3.2](https://llama.meta.com/)** — Open-source LLM powering local AI features
- **[Google Gemini](https://ai.google.dev/)** — Fallback LLM
- **C-DAC (Centre for Development of Advanced Computing), Pune** — Project incubation and research support
- Indian farmers — the end users this system was built for

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built with ❤️ for Indian Farmers</strong><br/>
  <em>Empowering agriculture through data and technology — C-DAC Pune</em>
</div>
