# 🌾 AI Powered Weather Resilient Crop Advisor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![LLaMA](https://img.shields.io/badge/LLaMA-3.2%20Local-8A2BE2?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)
![HTML5](https://img.shields.io/badge/HTML5-Vanilla_JS-E34F26?style=for-the-badge&logo=html5)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A Global Farm Intelligence Dashboard powered by LLaMA and Web Search Agents — covering 50+ countries, 250+ states, and 170+ districts natively.**

[Overview](#-overview) • [Agent Architecture](#-agent-architecture) • [Features](#-features) • [Installation](#-installation) • [API Endpoints](#-api-endpoints)

</div>

---

## 🧭 Overview

The **AI Powered Weather Resilient Crop Advisor** is an end-to-end, global agricultural intelligence platform. By combining live weather data, machine learning-blended climate forecasts, and large language model (LLM) reasoning, the system replicates the advice of a skilled agronomist for virtually any farming region in the world.

A user selects their country, state, and district, and the **Data Gathering Agent** instantly pulls real-time weather from Open-Meteo, cross-references it against historical climate zones, and uses a local LLaMA model (via Ollama) to enrich the soil data and local market prices. Finally, the **Crop Agent** ranks over 50 crops and streams a personalized, context-aware analysis directly to the UI.

The platform relies on **LLaMA 3.2 (via Ollama)** running entirely locally for free and private inference, with Google Gemini acting as an automatic cloud fallback.

---

## 💡 Project Idea

Most crop advisory tools rely on generic, static lookup tables or expensive IoT sensors. This platform takes a radically different approach by acting as a **Global Farm Intelligence Dashboard**. It combines **publicly available free data sources** with **Machine Learning** and **LLM Agents** to replicate the advice a skilled agronomist would give, dynamically, anywhere in the world.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Global Precision** | Location Agent dynamically maps 170+ global districts. Open-Meteo fetches real-time, exact lat/lon weather. |
| **Agentic Workflow** | Multi-agent system (Location, Data, Crop) orchestrates intelligence gathering behind a single `/api/analyze/stream` call. |
| **Graceful Degradation** | Remove LLM → rule-based scoring continues. Remove Internet → uses static climate zone modeling. Zero single points of failure. |
| **Free & Private LLM** | LLaMA 3.2 runs locally via Ollama — no API costs, no data sent to cloud, no quota limits. |

---

## 🤖 Agent Architecture

The system utilizes a multi-agent backend to orchestrate the intelligence gathering and analysis process:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                             │
│       Web Browser ←→ index.html + app.js + style.css               │
│       Dynamic Streaming UI · Chart.js Visualizations                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  HTTP / REST / Streaming (FastAPI)
┌──────────────────────────────▼──────────────────────────────────────┐
│                        API LAYER                                     │
│  GET /api/countries · GET /api/states · GET /api/districts          │
│  POST /api/analyze/stream · POST /chat · GET /weather/now           │
└──┬───────────────────────────┬───────────────────────────┬──────────┘
   │                           │                           │
┌──▼──────────────────┐ ┌──────▼──────────────────┐ ┌──────▼──────────┐
│ Location Agent      │ │ Data Gathering Agent    │ │ Crop Agent      │
│ (world_locations.json)│ │ (Weather + Climatology +│ │ (Ranking Engine │
│ 50+ Countries       │ │ LLM Soil/Market Enrich) │ │ + Streaming LLM)│
└──┬──────────────────┘ └──────┬──────────────────┘ └──────┬──────────┘
   │                           │                           │
┌──▼───────────────────────────▼───────────────────────────▼──────────┐
│                           LLM LAYER                                 │
│  Primary:  LLaMA 3.2 via Ollama (local, free, private)             │
│  Fallback: Google Gemini (if Ollama not running)                    │
│  Uses: Soil Enrichment, Market Prices, Crop Ranking, Farmer Chat    │
└──┬──────────────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  Crop DB (50+ crops) · World Locations · Zone Climate Normals       │
│  Open-Meteo Live API (Free, No Key)                                 │
└─────────────────────────────────────────────────────────────────────┘
```

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
[2] Data Gathering Agent (Weather)
        Fetches live real-time weather from Open-Meteo (temp_max, temp_min, rainfall, wind, UV).
        │
        ▼
[3] Data Gathering Agent (Climate & Season)
        Maps the district to a global climate zone (e.g. Mediterranean, Tropical).
        Builds a 6-month historical forecast anchored to the live temperature.
        │
        ▼
[4] Data Gathering Agent (LLM Enrichment)
        Invokes Gemini/LLaMA to dynamically assess typical soil profiles and market prices
        for the specified region. Falls back to zone defaults if LLM is down.
        │
        ▼
[5] Crop Agent (Scoring Engine)
        Evaluates 50+ crops using a 6-factor suitability score (Temp, Water, Soil, Regional, Season, Drought).
        Optionally blends with a Random Forest ML model (0.6 ML + 0.4 Rule-based).
        │
        ▼
[6] Risk & Pest Assessment
        Flags drought risk, temperature stress, and current pest warnings based on live weather.
        │
        ▼
[7] Streaming Response
        Streams the final ranked recommendations and a generated summary to the UI token-by-token.
```

---

## 🦙 LLM Integration (LLaMA 3.2 + Gemini Fallback)

| LLM Component | File | Purpose |
|---------------|------|---------|
| **Soil & Market Enrichment** | `data_gathering_agent.py` | Dynamically assesses local soil types and market crop prices for global regions. |
| **Streaming Explainer** | `crop_agent.py` | Generates a real-time summary of why specific crops were recommended. |
| **Farmer Chat** | `llm_chat.py` | Context-aware Q&A bot that remembers live weather and region specifics. |

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
| **Temperature Compatibility** | 25% | Optimal range (100) → linear decay to survival limits (60) → gentle 8°C grace margin decay (20) → beyond limits (0). |
| **Water Availability** | 25% | Expected rainfall + irrigation vs. crop water requirement. |
| **Soil Compatibility** | 15% | Texture, pH, drainage, and organic matter matching. |
| **Regional Suitability** | 15% | District-specific suitability modifiers. |
| **Seasonal Adjustment** | 10% | Whether the crop is traditionally planted in the detected season. |
| **Drought Tolerance Bonus** | 10% | Score boost for drought-resistant crops during expected dry spells. |

*(Note: If the Random Forest model is trained and available, the final score becomes a 60:40 blend of ML prediction and this rule-based engine).*

---

## ✨ Features

### 🌍 Global Location Support
- Supports **50+ countries**, **250+ states/provinces**, and **170+ districts**.
- Includes accurate fallback mechanisms to state capitals or regional centers if a specific district is missing.

### 🧠 Graceful AI Degradation
- **Tier 1 (Default)**: Ollama runs a local LLaMA model for 100% free, private, offline intelligence.
- **Tier 2 (Fallback)**: If Ollama isn't running, it seamlessly falls back to Google Gemini (requires an API key in `.env`).
- **Tier 3 (Safe Mode)**: If no LLM is available, the system uses static rule-based scoring and fallback climate profiles to ensure the farmer still gets a recommendation.

### 🌡️ Real-Time & Forecasted Weather
- Live Open-Meteo integration for accurate today-weather.
- 6-Month forward-looking climate modeling using dynamic zone anchoring (ensuring high-altitude regions get accurate cooler temperatures compared to their broad geographic zone).

### 💬 Interactive Streaming Chat
- Real-time token streaming gives a modern, "ChatGPT-like" experience for both the main crop analysis and the interactive Farmer Chat.

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.8+, FastAPI, Uvicorn |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript, Chart.js |
| **Machine Learning** | Scikit-learn (Random Forest Suitability Model) |
| **Agent Framework** | Custom Python Agents |
| **Primary LLM** | **LLaMA 3.2** via **Ollama** (local, free, private) |
| **Fallback LLM** | Google Gemini (`gemini-2.0-flash-lite`) |
| **Live Weather** | Open-Meteo API (Free, no API key required) |

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8+ | 3.10 or 3.11 |
| **RAM** | 4 GB | 8 GB |
| **Storage** | 600 MB | 3 GB (to hold the LLaMA model) |
| **Ollama Model** | `gemma3:2b` (Lighter) | `llama3.2` |
| **Internet** | Required for live weather | Stable broadband |

---

## 🚀 Installation

### Prerequisites
- Python **3.8+**
- [Ollama](https://ollama.com/download) *(recommended — free local LLM)*
- A free **[Google Gemini API key](https://aistudio.google.com/app/apikey)** *(optional — used as fallback)*

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
```

### 6. Start the Platform
You can use the bundled setup script, or run the web server directly:
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
http://localhost:8000/health   ← System status (LLM provider, agents)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/countries` | List of supported countries |
| `GET` | `/api/states/{country}` | List of states/provinces for a given country |
| `GET` | `/api/districts/{country}/{state}` | List of districts/cities |
| `POST` | `/api/analyze/stream` | **Streaming** crop recommendation via LLM Agent |
| `POST` | `/chat` | Interactive Farmer Q&A Chat |
| `GET` | `/weather/now/{region_id}` | Live real-time temperature from Open-Meteo |
| `POST` | `/recommend` | Legacy full-batch JSON recommendation engine |
| `GET` | `/health` | API health, ML model status, and active LLM provider |

---

## 📁 Project Structure

```text
agri_crop_recommendation/
├── data/                       # Reference data (locations, crops)
├── models/                     # Saved ML models
├── scripts/                    # Utility scripts (training, scraping)
├── src/
│   ├── agents/                 # LLaMA Agents (Location, Data Gathering, Crop)
│   ├── api/                    # FastAPI endpoints (app.py, models.py)
│   ├── crops/                  # Crop database and soil definitions
│   ├── ml/                     # ML pipelines and predictors
│   ├── services/               # Recommender, risk, pests, calendar
│   ├── utils/                  # Region manager, seasons
│   └── weather/                # Weather fetcher, historical climatology
├── static/                     # CSS, JS, Images
├── templates/                  # HTML templates (index.html)
├── .env.example                # Example environment variables
├── requirements.txt            # Python dependencies
└── run_website.py              # Server startup script
```

---

## 📄 License

MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Global Agricultural Intelligence</strong><br/>
  <em>Empowering agriculture through AI, Agents, and Real-Time Data</em>
</div>
