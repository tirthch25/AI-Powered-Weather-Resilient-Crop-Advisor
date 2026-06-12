import re

with open(r'd:\AI-Powered-Weather-Resilient-Crop-Advisor\user_module_paper.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update the tree structure
new_tree = '''```text
agri_crop_recommendation/
|-- main.py                     # Application entry (alternative)
|-- run_website.py              # Primary startup script (uvicorn)
|-- requirements.txt            # Python dependencies
|
|-- src/
|   |-- api/
|   |   |-- app.py              # All FastAPI routes & request handlers
|   |   |-- models.py           # API Request/Response Pydantic models
|   |-- crops/
|   |   |-- database.py         # Crop knowledge base (50+ crops)
|   |   |-- models.py           # CropInfo dataclass
|   |   |-- soil.py             # Soil compatibility calculations
|   |-- agents/                 # Multi-Agent Intelligence Layer
|   |   |-- location_agent.py   # Resolves global district coordinates
|   |   |-- data_gathering_agent.py # Aggregates live weather, soil, and market data
|   |   |-- crop_agent.py       # Orchestrates scoring and final LLM explanations
|   |-- ml/
|   |   |-- pipeline.py         # ML data pipeline + feature engineering
|   |   |-- predictor.py        # Random Forest crop suitability model
|   |   |-- lstm_weather.py     # PyTorch LSTM forecaster
|   |   |-- xgboost_weather.py  # XGBoost weather forecaster
|   |-- services/
|   |   |-- recommender.py      # Core recommendation engine
|   |   |-- risk.py             # Risk assessment engine
|   |   |-- pests.py            # Pest & disease warning system
|   |   |-- calendar.py         # Planting calendar generator
|   |   |-- llm_filter.py       # LLM regional crop filter (Ollama/Gemini)
|   |   |-- llm_explainer.py    # LLM crop explanation generator
|   |   |-- llm_chat.py         # LLM streaming farmer Q&A chat
|   |-- utils/
|   |   |-- regions.py          # Region manager & GPS lookup
|   |   |-- seasons.py          # Season detection & water adjustment
|   |-- weather/
|       |-- fetcher.py          # Live weather API client
|       |-- forecast.py         # ML ensemble forecasting logic
|       |-- history.py          # Historical climatology data
|
|-- scripts/
|   |-- fetch_district_weather.py  # Download 10yr weather data
|   |-- train_model.py             # Train all ML models
|   |-- setup_weather.py           # Initial data setup
|   |-- check_missing_districts.py # Audit global missing districts
|   |-- build_world_locations.py   # Rebuild location lookup table
|
|-- models/
|   |-- crop_suitability/          # Random Forest ML weights
|   |-- weather_lstm/              # LSTM model weights
|   |-- weather_xgboost/           # XGBoost model weights
|
|-- data/
|   |-- reference/
|   |   |-- crop_knowledge.json
|   |   |-- world_locations.json
|   |-- weather/
|       |-- zone/
|
|-- static/                     # CSS, JS, Images
|-- templates/                  # HTML templates (index.html)
```'''

# Find the Directory Structure section
start_idx = text.find('## 17. Directory Structure')
# We need to find the next section to slice properly
end_idx = text.find('## 16. Module 14', start_idx)

if start_idx != -1 and end_idx != -1:
    section_prefix = text[start_idx:text.find('```', start_idx)]
    new_text = text[:start_idx] + '## 17. Directory Structure\n\n' + new_tree + '\n\n' + text[end_idx:]
    
    # 2. Fix the module numbering (11 to 14 jump)
    new_text = new_text.replace('## 16. Module 14 — LLM Regional Crop Filter', '## 18. Module 12 — LLM Regional Crop Filter')
    new_text = new_text.replace('## 17. Module 15 — LLM Crop Explainer', '## 19. Module 13 — LLM Crop Explainer')
    new_text = new_text.replace('## 18. Module 16 — LLM Farmer Chat', '## 20. Module 14 — LLM Farmer Chat')
    
    # 3. Add Agent module
    agent_module = '''## 21. Module 15 — Multi-Agent Intelligence Workflow (New in v3.0)

### 21.1 Purpose
Orchestrates global location resolution, live data gathering, and crop recommendations through a multi-agent system. This ensures that the platform scales to 50+ countries without hardcoding location-specific behaviors.

### 21.2 Agent Architecture

1. **Location Agent (`location_agent.py`)**
   - Parses the requested Country, State, and District.
   - Looks up exact Latitude/Longitude coordinates using `world_locations.json`.
   - Implements graceful fallback to state capitals if a district is unrecognized.

2. **Data Gathering Agent (`data_gathering_agent.py`)**
   - Asynchronously gathers real-time weather from Open-Meteo.
   - Retrieves historical climate-zone expectations.
   - Injects the location profile into the primary LLM (LLaMA 3.2 / Gemini) to dynamically assess expected soil composition and local market prices.
   - Combines API data and LLM inferences into a unified `GatheredData` state.

3. **Crop Agent (`crop_agent.py`)**
   - Receives the `GatheredData` state.
   - Invokes the `recommend_crops()` scoring engine.
   - Enriches the top 3 recommended crops with farmer-friendly, LLM-generated explanations (`llm_explainer.py`).
   - Streams the final ranked recommendations back to the client UI.

'''
    
    # Insert before "## 19. System Limitations"
    limit_idx = new_text.find('## 19. System Limitations')
    new_text = new_text[:limit_idx] + agent_module + new_text[limit_idx:].replace('## 19.', '## 22.')
    
    # Update TOC
    toc_start = new_text.find('## Table of Contents')
    toc_end = new_text.find('## 1. Project Overview')
    
    if toc_start != -1 and toc_end != -1:
        new_toc = '''## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Module 1 — Web Interface (Frontend)](#3-module-1--web-interface-frontend)
4. [Module 2 — REST API Layer](#4-module-2--rest-api-layer)
5. [Module 3 — Crop Recommendation Engine](#5-module-3--crop-recommendation-engine)
6. [Module 4 — ML Weather Forecasting (LSTM + XGBoost)](#6-module-4--ml-weather-forecasting-lstm--xgboost)
7. [Module 5 — Crop Suitability ML Model (Random Forest)](#7-module-5--crop-suitability-ml-model-random-forest)
8. [Module 6 — Risk Assessment Engine](#8-module-6--risk-assessment-engine)
9. [Module 7 — Pest & Disease Warning System](#9-module-7--pest--disease-warning-system)
10. [Module 8 — Planting Calendar](#10-module-8--planting-calendar)
11. [Module 9 — Crop Knowledge Base](#11-module-9--crop-knowledge-base)
12. [Module 10 — Regional Data & Soil Information](#12-module-10--regional-data--soil-information)
13. [Module 11 — Historical Weather Data Pipeline](#13-module-11--historical-weather-data-pipeline)
14. [Data Flow Diagram](#14-data-flow-diagram)
15. [API Reference Summary](#15-api-reference-summary)
16. [Technology Stack](#16-technology-stack)
17. [Directory Structure](#17-directory-structure)
18. [Module 12 — LLM Regional Crop Filter](#18-module-12--llm-regional-crop-filter)
19. [Module 13 — LLM Crop Explainer](#19-module-13--llm-crop-explainer)
20. [Module 14 — LLM Farmer Chat](#20-module-14--llm-farmer-chat)
21. [Module 15 — Multi-Agent Intelligence Workflow](#21-module-15--multi-agent-intelligence-workflow)
22. [System Limitations & Future Scope](#22-system-limitations--future-scope)

'''
        # Replace TOC
        new_text = new_text[:toc_start] + new_toc + new_text[toc_end:]
        
        with open(r'd:\AI-Powered-Weather-Resilient-Crop-Advisor\user_module_paper.md', 'w', encoding='utf-8') as fw:
            fw.write(new_text)
        print("Updated successfully")
    else:
        print("TOC not found")
else:
    print("Directory Structure section not found")
