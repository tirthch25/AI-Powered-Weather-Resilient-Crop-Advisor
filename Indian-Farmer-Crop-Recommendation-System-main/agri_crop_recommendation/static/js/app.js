/**
 * Indian Farmer Crop Recommendation System — Frontend v2.0
 */

// ===== Global State =====
let weatherChart = null;
let _currentRegionId = null;
let _currentSeason = null;
let _chatEnabled = false;
let _liveWeatherSummary = ''; // Real-time weather fetched from Open-Meteo via /weather/now

// ===== Crop Emoji Map (50+ crops) =====
const CROP_EMOJIS = {
  // Cereals & Millets
  'BAJRA_01': '🌾', 'JOWAR_01': '🌾', 'RAGI_01': '🌾',
  'FOXTAIL_01': '🌾', 'WHEAT_01': '🌾', 'RICE_01': '🌾',
  'MAIZE_01': '🌽', 'BARLEY_01': '🌾', 'BARNYARD_01': '🌾',
  'KODO_01': '🌾', 'PROSO_01': '🌾', 'LITTLE_01': '🌾',

  // Pulses
  'MOONG_01': '🫘', 'URAD_01': '🫘', 'COWPEA_01': '🫘',
  'GUAR_01': '🌿', 'HORSEGRAM_01': '🫘', 'MOTHBEAN_01': '🫘',
  'LENTIL_01': '🫘', 'CHICKPEA_01': '🫘', 'PIGEONPEA_01': '🫘',
  'FIELDPEA_01': '🫘', 'RAJMA_01': '🫘',

  // Oilseeds
  'SESAME_01': '🌻', 'SUNFLOWER_01': '🌻', 'SOYBEAN_01': '🫘',
  'GROUNDNUT_01': '🥜', 'MUSTARD_01': '🌼', 'FLAXSEED_01': '🌿',
  'CASTOR_01': '🌿', 'SAFFLOWER_01': '🌼',

  // Vegetables (Fruiting)
  'TOMATO_01': '🍅', 'BRINJAL_01': '🍆', 'OKRA_01': '🌿',
  'BOTTLEGOURD_01': '🥒', 'CUCUMBER_01': '🥒', 'RIDGEGOURD_01': '🥒',
  'BITTERGOURD_01': '🥒', 'PUMPKIN_01': '🎃', 'WATERMELON_01': '🍉',
  'MUSKMELON_01': '🍈', 'CHILLI_01': '🌶️', 'CAPSICUM_01': '🫑',
  'FRENCHBEAN_01': '🫘', 'CLUSTERBEAN_01': '🌿', 'WINGEDBEAN_01': '🌿',

  // Root Vegetables
  'CARROT_01': '🥕', 'RADISH_01': '🌿', 'BEETROOT_01': '🌿',
  'TURNIP_01': '🌿', 'POTATO_01': '🥔', 'SWEETPOTATO_01': '🍠',
  'YARDLONGBEAN_01': '🌿',

  // Leafy Greens
  'SPINACH_01': '🥬', 'FENUGREEK_01': '🌿', 'CORIANDER_01': '🌿',
  'AMARANTH_01': '🌿', 'MUSTARDGREENS_01': '🥬', 'LETTUCE_01': '🥬',
  'SPRINGONION_01': '🧅', 'CABBAGE_01': '🥬', 'CAULIFLOWER_01': '🧄',

  // Cash crops
  'SUGARCANE_01': '🌿', 'COTTON_01': '🌿', 'TURMERIC_01': '🌿',
  'GINGER_01': '🌿', 'GARLIC_01': '🧄', 'ONION_01': '🧅',
};

const PHASE_COLORS = ['#166534', '#15803d', '#16a34a', '#22c55e', '#4ade80'];

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
  loadRegions();
  document.getElementById('recommendationForm').addEventListener('submit', handleSubmit);
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
    });
  }
});

// ===== Fetch Live Weather (Open-Meteo via backend) =====
async function fetchLiveWeather(regionId) {
  if (!regionId) return;
  try {
    const res = await fetch(`/weather/now/${encodeURIComponent(regionId)}`);
    if (!res.ok) return;
    const data = await res.json();
    _liveWeatherSummary = data.summary || '';
    // Show a small live-weather pill in the chat header so the user knows data is ready
    const notice = document.getElementById('chatWeatherPill');
    if (notice && data.temp_max !== undefined) {
      notice.textContent =
        `📡 Live: ${data.temp_max}\u00b0C high · ${data.temp_min}\u00b0C low · ${data.region_name} · ${data.date}`;
      notice.style.display = 'inline-block';
    }
  } catch (_) {
    // silently ignore — non-critical
  }
}

// ===== Load Regions =====
let _allRegions = [];

async function loadRegions() {
  try {
    const response = await fetch('/regions');
    const data = await response.json();
    _allRegions = data.regions;

    const stateSelect = document.getElementById('state');
    const states = [...new Set(_allRegions.map(r => r.state || 'Other'))].sort();
    stateSelect.innerHTML = '<option value="">Select your state…</option>';
    states.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      stateSelect.appendChild(opt);
    });

    stateSelect.addEventListener('change', () => {
      const distSelect = document.getElementById('region');
      const chosen = stateSelect.value;
      if (!chosen) {
        distSelect.innerHTML = '<option value="">Select a state first…</option>';
        distSelect.disabled = true;
        return;
      }
      const filtered = _allRegions
        .filter(r => (r.state || 'Other') === chosen)
        .sort((a, b) => a.name.localeCompare(b.name));
      distSelect.innerHTML = '<option value="">Select your district…</option>';
      filtered.forEach(region => {
        const opt = document.createElement('option');
        opt.value = region.region_id;
        opt.textContent = `${region.name} (${region.climate_zone})`;
        distSelect.appendChild(opt);
      });
      distSelect.disabled = false;

      // Pre-fetch live weather as soon as the user picks a state
      // (they'll likely pick a district next — wire that too)
      distSelect.addEventListener('change', () => {
        if (distSelect.value) fetchLiveWeather(distSelect.value);
      }, { once: false });
    });
  } catch (error) {
    console.error('Failed to load regions:', error);
    document.getElementById('state').innerHTML = '<option value="">Failed to load — refresh page</option>';
  }
}

// ===== Handle Form Submit =====
async function handleSubmit(e) {
  e.preventDefault();

  const regionId = document.getElementById('region').value;
  if (!regionId) { alert('Please select a district'); return; }

  const body = {
    region_id: regionId,
    irrigation: document.getElementById('irrigation').value,
    planning_days: parseInt(document.getElementById('planning_days').value)
  };

  const soilTexture = document.getElementById('soil_texture').value;
  const soilPh = document.getElementById('soil_ph').value;
  if (soilTexture && soilPh) {
    body.soil = {
      texture: soilTexture,
      ph: parseFloat(soilPh),
      organic_matter: document.getElementById('soil_organic').value,
      drainage: document.getElementById('soil_drainage').value
    };
  }

  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('results').classList.add('hidden');
  document.getElementById('submitBtn').disabled = true;

  try {
    const response = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Server error');
    }

    const data = await response.json();

    // Store context for chat
    _currentRegionId = data.region?.region_id || regionId;
    _currentSeason = data.season?.current || '';
    _chatEnabled = true;
    // Also refresh live weather summary from the recommendation result
    if (data.medium_range_forecast) {
      const f = data.medium_range_forecast;
      const avg = f.expected_avg_temp ?? null;
      if (avg !== null) {
        _liveWeatherSummary = `Avg temp: ${parseFloat(avg).toFixed(1)}\u00b0C (from recommendation forecast)`;
      }
    }

    renderResults(data);
  } catch (error) {
    console.error('Recommendation failed:', error);
    alert('Error: ' + error.message);
  } finally {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('submitBtn').disabled = false;
  }
}

// ===== Render All Results =====
function renderResults(data) {
  // AI Banner
  const banner = document.getElementById('ai-banner');
  if (data.llm_powered && banner) {
    banner.classList.remove('hidden');
    const txt = document.getElementById('ai-banner-text');
    if (txt) txt.textContent = data.llm_note || 'AI-powered recommendations active.';
  } else if (banner) {
    banner.classList.add('hidden');
  }

  renderOverview(data);
  renderGuidance(data.season);
  renderForecastChart(data.medium_range_forecast);
  renderCrops(data.recommended_crops, data.llm_powered);
  renderRiskAssessment(data.recommended_crops);
  renderPestWarnings(data.recommended_crops);
  renderCalendar(data.planting_calendars);

  // Build crop context string for chat (top 5 crop names)
  if (data.recommended_crops && data.recommended_crops.length > 0) {
    _chatCropCtx = data.recommended_crops
      .slice(0, 5)
      .map(c => c.crop)
      .join(', ');
    _showSuggestionChips(data.recommended_crops);
  }

  document.getElementById('results').classList.remove('hidden');

  setTimeout(() => {
    document.getElementById('overview').scrollIntoView({ behavior: 'smooth' });
  }, 200);
}

// ===== Overview Stats =====
function renderOverview(data) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  set('result-region', data.region?.name || '-');
  set('result-season', data.season?.current || '-');

  const forecast = data.medium_range_forecast || {};
  const rawTemp = forecast.expected_avg_temp ?? forecast.ml_summary?.avg_temp ?? null;
  set('result-temp', rawTemp !== null ? parseFloat(rawTemp).toFixed(1) + '°C' : '-');

  const rawRain = forecast.expected_rainfall_mm ?? forecast.ml_summary?.total_rainfall ?? null;
  set('result-rain', rawRain !== null ? Math.round(rawRain) + ' mm' : '-');

  set('result-soil', data.soil ? `${data.soil.texture} pH ${data.soil.ph}` : '-');

  const source = forecast.forecast_source || 'climatology';
  set('result-forecast-src', source);

  const badge = document.getElementById('forecast-badge');
  if (badge) {
    if (source.includes('ensemble') || source.includes('LSTM')) {
      badge.textContent = '🤖 ML Ensemble'; badge.className = 'badge badge-success';
    } else if (source.includes('XGBoost')) {
      badge.textContent = '🤖 ' + source; badge.className = 'badge badge-info';
    } else {
      badge.textContent = '📊 Climatology'; badge.className = 'badge badge-warning';
    }
  }
}

// ===== Season Guidance =====
function renderGuidance(season) {
  const guidance = document.getElementById('season-guidance');
  let text = season.guidance || `Current season: ${season.current}`;
  if (season.is_transition) text += ` Transitioning to ${season.next_season} soon.`;
  guidance.textContent = text;
}

// ===== Weather Forecast Chart =====
function renderForecastChart(forecast) {
  const canvas = document.getElementById('weatherChart');
  if (!canvas) return;
  if (weatherChart) { weatherChart.destroy(); weatherChart = null; }

  const monthly = (forecast && forecast.monthly_forecast) || [];
  const dailyPreds = (forecast && forecast.daily_predictions) || [];

  if (monthly.length === 12) { _renderMonthlyChart(canvas, monthly, forecast); return; }
  if (dailyPreds.length > 0) { _renderDailyChart(canvas, dailyPreds); return; }

  const parent = canvas.parentElement;
  if (parent) {
    const avgTemp = forecast.expected_avg_temp || '-';
    const rain = forecast.expected_rainfall_mm || '-';
    const hum = forecast.expected_humidity || '-';
    parent.innerHTML = `
      <div class="forecast-summary-card">
        <div class="forecast-summary-row">
          <div class="forecast-summary-item">
            <span class="fsi-icon">🌡️</span>
            <span class="fsi-val">${avgTemp}°C</span>
            <span class="fsi-label">Avg Temp</span>
          </div>
          <div class="forecast-summary-item">
            <span class="fsi-icon">🌧️</span>
            <span class="fsi-val">${rain} mm</span>
            <span class="fsi-label">Expected Rain</span>
          </div>
          <div class="forecast-summary-item">
            <span class="fsi-icon">💧</span>
            <span class="fsi-val">${hum}%</span>
            <span class="fsi-label">Avg Humidity</span>
          </div>
        </div>
        <p class="forecast-summary-note">📊 Seasonal summary — train ML models for daily detail</p>
      </div>`;
  }
}

function _renderMonthlyChart(canvas, monthly, forecast) {
  const currentMonth = new Date().getMonth();
  const indices = [0, 1, 2].map(offset => (currentMonth + offset) % 12);
  const window3 = indices.map(i => monthly[i]);

  const labels    = window3.map(m => m.month);
  const tempMax   = window3.map(m => m.temp_max   ?? m.temperature + 7);
  const tempMin   = window3.map(m => m.temp_min   ?? m.temperature - 7);
  const rainfall  = window3.map(m => m.rainfall);
  const humidity  = window3.map(m => m.humidity);

  const barBg     = ['rgba(86,163,43,0.85)', 'rgba(86,163,43,0.50)', 'rgba(86,163,43,0.30)'];
  const barBorder = ['#56a32b', 'rgba(86,163,43,0.6)', 'rgba(86,163,43,0.35)'];

  weatherChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Rainfall (mm)',
          data: rainfall,
          type: 'bar',
          backgroundColor: barBg,
          borderColor: barBorder,
          borderWidth: 1.5,
          borderRadius: 6,
          yAxisID: 'yRain',
          order: 3,
        },
        {
          // Invisible dataset to create fill between temp_max and temp_min
          label: '_tempFill',
          data: tempMin,
          type: 'line',
          borderColor: 'transparent',
          backgroundColor: 'rgba(249,115,22,0.15)',
          borderWidth: 0,
          fill: '+1',          // fill up to the next dataset (temp_max)
          tension: 0.3,
          pointRadius: 0,
          yAxisID: 'yTemp',
          order: 2,
          hidden: false,
        },
        {
          label: 'Max Temp (°C)',
          data: tempMax,
          type: 'line',
          borderColor: '#f97316',
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          tension: 0.3,
          fill: false,
          pointBackgroundColor: ['#f97316', 'rgba(249,115,22,0.75)', 'rgba(249,115,22,0.55)'],
          pointRadius: [8, 5, 5],
          pointHoverRadius: 10,
          yAxisID: 'yTemp',
          order: 1,
        },
        {
          label: 'Min Temp (°C)',
          data: tempMin,
          type: 'line',
          borderColor: '#60a5fa',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 4],
          tension: 0.3,
          fill: false,
          pointBackgroundColor: ['#60a5fa', 'rgba(96,165,250,0.75)', 'rgba(96,165,250,0.55)'],
          pointRadius: [6, 4, 4],
          pointHoverRadius: 8,
          yAxisID: 'yTemp',
          order: 1,
        },
        {
          label: 'Humidity (%)',
          data: humidity,
          type: 'line',
          borderColor: '#a78bfa',
          backgroundColor: 'transparent',
          borderWidth: 1.8,
          borderDash: [3, 3],
          tension: 0.3,
          fill: false,
          pointRadius: 4,
          pointHoverRadius: 7,
          yAxisID: 'yHum',
          order: 1,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: {
          labels: {
            color: '#a8c488',
            font: { family: 'Inter', size: 12 },
            usePointStyle: true,
            pointStyleWidth: 10,
            filter: (item) => item.text !== '_tempFill',  // hide fill helper
          }
        },
        tooltip: {
          backgroundColor: 'rgba(13,26,13,0.97)',
          titleColor: '#7ec850',
          bodyColor: '#d1fae5',
          borderColor: 'rgba(86,163,43,0.3)',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            title: (items) => {
              const idx = items[0].dataIndex;
              return labels[idx] + (idx === 0 ? '  ◀ This Month' : '');
            },
            label: (item) => {
              if (item.dataset.label === '_tempFill') return null;
              const units = {
                'Rainfall (mm)':  ' mm rainfall',
                'Max Temp (°C)':  '°C  (daily high)',
                'Min Temp (°C)':  '°C  (daily low)',
                'Humidity (%)':   '% humidity',
              };
              const suffix = units[item.dataset.label] || '';
              return ` ${item.dataset.label.replace(' (°C)', '').replace(' (%)', '').replace(' (mm)', '')}: ${item.parsed.y}${suffix}`;
            },
            afterBody: (items) => {
              const maxItem = items.find(i => i.dataset.label === 'Max Temp (°C)');
              const minItem = items.find(i => i.dataset.label === 'Min Temp (°C)');
              if (maxItem && minItem) {
                return [` Range: ${minItem.parsed.y}°C – ${maxItem.parsed.y}°C`];
              }
              return [];
            }
          }
        },
      },
      scales: {
        x: {
          ticks: {
            color: (ctx) => ctx.index === 0 ? '#f97316' : '#6b8f51',
            font: (ctx) => ctx.index === 0
              ? { family: 'Inter', weight: '700', size: 13 }
              : { family: 'Inter', size: 12 },
          },
          grid: { color: 'rgba(255,255,255,0.04)' }
        },
        yRain: {
          type: 'linear', position: 'right',
          title: { display: true, text: 'Rainfall (mm)', color: '#4ade80', font: { size: 11 } },
          ticks: { color: '#4ade80' },
          grid: { display: false },
          beginAtZero: true,
        },
        yTemp: {
          type: 'linear', position: 'left',
          title: { display: true, text: 'Temperature Range (°C)', color: '#f97316', font: { size: 11 } },
          ticks: { color: '#f97316' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        yHum: {
          type: 'linear', position: 'left',
          display: false, min: 0, max: 110,
          grid: { display: false },
        },
      }
    }
  });
}

function _renderDailyChart(canvas, dailyPreds) {
  const labels = dailyPreds.map(p => `Day ${p.day}`);
  weatherChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Max Temp (°C)', data: dailyPreds.map(p => p.temp_max), borderColor: '#f97316', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'y' },
        { label: 'Min Temp (°C)', data: dailyPreds.map(p => p.temp_min), borderColor: '#60a5fa', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'y' },
        { label: 'Rainfall (mm)', data: dailyPreds.map(p => p.rainfall), borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.15)', borderWidth: 2, tension: 0.4, type: 'bar', yAxisID: 'y1' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: { legend: { labels: { color: '#a8c488', font: { family: 'Inter', size: 12 } } } },
      scales: {
        x: { ticks: { color: '#6b8f51' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { type: 'linear', position: 'left', title: { display: true, text: 'Temperature (°C)', color: '#f97316' }, ticks: { color: '#f97316' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y1: { type: 'linear', position: 'right', title: { display: true, text: 'Rainfall (mm)', color: '#4ade80' }, ticks: { color: '#4ade80' }, grid: { display: false } }
      }
    }
  });
}

// ===== Crop Cards =====
function renderCrops(crops, llmPowered) {
  const grid = document.getElementById('cropsGrid');
  const count = document.getElementById('crops-count');
  count.textContent = `${crops.length} crops analyzed and ranked`;

  grid.innerHTML = crops.map((crop, index) => {
    const emoji = CROP_EMOJIS[crop.crop_id] || '🌱';
    const score = crop.suitability_score;
    const scoreClass = score >= 70 ? '' : score >= 45 ? 'medium' : 'low';
    const scoreColor = score >= 70 ? '#4ade80' : score >= 45 ? '#fbbf24' : '#f87171';
    const isTop3 = index < 3;

    const durRange = crop.duration_range && crop.duration_range.length === 2
      ? `${crop.duration_range[0]}–${crop.duration_range[1]} days growth cycle`
      : `${crop.growth_duration_days} days growth cycle`;

    // Score source badge
    const srcBadge = crop.score_source === 'ml_blended'
      ? `<span class="score-source-badge score-source-ml">ML</span>`
      : `<span class="score-source-badge score-source-rule">Rule</span>`;

    // Risk badge
    let riskClass = 'risk-low', riskLabel = 'Low Risk';
    if (crop.risk_assessment) {
      const level = crop.risk_assessment.overall_risk_level;
      riskClass = `risk-${level.toLowerCase()}`;
      riskLabel = `${level} Risk`;
    } else if (crop.risk_note) {
      if (crop.risk_note.includes('High')) riskClass = 'risk-high';
      else if (crop.risk_note.includes('Moderate')) riskClass = 'risk-medium';
      riskLabel = crop.risk_note;
    }

    // Pest mini badges
    const pestBadges = (crop.pest_warnings && crop.pest_warnings.length > 0)
      ? `<div class="pest-mini-badges">${crop.pest_warnings.map(p => `<span class="pest-mini-badge">🐛 ${p.name}</span>`).join('')}</div>`
      : '';

    // Growing tip
    const tipHtml = crop.growing_tip
      ? `<div class="growing-tip">💡 <strong>Tip:</strong> ${crop.growing_tip}</div>`
      : '';

    // LLM Explanation (top 3 only when llm_powered)
    let llmHtml = '';
    if (isTop3 && crop.llm_explanation && Object.keys(crop.llm_explanation).length > 0) {
      const exp = crop.llm_explanation;
      const whyGoodPill = exp.why_good
        ? `<span class="llm-pill"><span class="llm-pill-icon">✅</span>${exp.why_good}</span>`
        : '';
      const watchOutPill = exp.watch_out
        ? `<span class="llm-pill"><span class="llm-pill-icon">⚠️</span>${exp.watch_out}</span>`
        : '';
      const localLang = exp.hindi || exp.marathi
        ? `<div class="llm-local">${exp.hindi || exp.marathi}</div>`
        : '';

      llmHtml = `
        <div class="llm-explanation">
          <div class="llm-label">
            <span class="llm-label-dot"></span>
            AI Explanation · Gemini
          </div>
          <div class="llm-english">${exp.english || ''}</div>
          <div class="llm-pills">${whyGoodPill}${watchOutPill}</div>
          ${localLang}
        </div>`;
    }

    return `
      <div class="crop-card ${isTop3 ? 'top-3' : ''}" style="animation-delay: ${index * 0.07}s">
        <div class="crop-card-rank">#${index + 1}</div>
        <div class="crop-card-header">
          <div class="crop-card-image">${emoji}</div>
          <div>
            <div class="crop-card-title">${crop.crop}</div>
            <div class="crop-card-duration">${durRange}</div>
          </div>
        </div>

        <div class="score-bar-container">
          <div class="score-bar-label">
            <span>Suitability ${srcBadge}</span>
            <span style="color:${scoreColor};font-weight:800">${score.toFixed(1)}%</span>
          </div>
          <div class="score-bar">
            <div class="score-bar-fill ${scoreClass}" style="width:${score}%"></div>
          </div>
        </div>

        <div class="crop-details">
          <div class="crop-detail-item">
            <span class="label">Water Need</span>
            <span class="value">${crop.water_required_mm} mm</span>
          </div>
          <div class="crop-detail-item">
            <span class="label">Irrigation Needed</span>
            <span class="value">${crop.irrigation_needed_mm} mm</span>
          </div>
          <div class="crop-detail-item">
            <span class="label">Drought Tol.</span>
            <span class="value">${crop.drought_tolerance}</span>
          </div>
          <div class="crop-detail-item">
            <span class="label">Regional Score</span>
            <span class="value">${(crop.regional_suitability * 100).toFixed(0)}%</span>
          </div>
        </div>

        <span class="crop-risk-badge ${riskClass}">${riskLabel}</span>
        ${pestBadges}
        ${tipHtml}
        ${llmHtml}
      </div>`;
  }).join('');
}

// ===== Risk Assessment =====
function renderRiskAssessment(crops) {
  const container = document.getElementById('riskContent');
  const topCrop = crops.find(c => c.risk_assessment);
  if (!topCrop || !topCrop.risk_assessment) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">✅</div><p>No detailed risk assessment available</p></div>`;
    return;
  }

  const risk = topCrop.risk_assessment;
  const riskItems = [
    {
      title: '💧 Drought Risk',
      data: risk.drought_risk,
      detail: `Water deficit: ${risk.drought_risk?.water_deficit_mm || 0} mm (${risk.drought_risk?.water_deficit_pct || 0}%)`
    },
    {
      title: '🌡️ Temperature Stress',
      data: risk.temperature_stress,
      detail: `Heat stress: ${risk.temperature_stress?.heat_stress_days || 0} days, Cold: ${risk.temperature_stress?.cold_stress_days || 0} days`
    },
    {
      title: '⛈️ Extreme Weather',
      data: risk.extreme_weather,
      detail: `Heavy rain: ${risk.extreme_weather?.heavy_rain_days || 0} days, Heatwave: ${risk.extreme_weather?.heatwave_days || 0} days`
    }
  ];

  container.innerHTML = `
    <p style="margin-bottom:0.75rem;font-size:0.82rem;color:var(--text-muted)">
      Risk analysis for top crop: <strong style="color:var(--green-300)">${topCrop.crop}</strong>
    </p>
    <div class="risk-grid">
      ${riskItems.map(item => {
        const level = (item.data?.level || 'Low').toLowerCase();
        return `
          <div class="risk-item risk-item-${level}">
            <div class="risk-item-header">
              <span class="risk-item-title">${item.title}</span>
              <span class="crop-risk-badge risk-${level}">${item.data?.score || 0}</span>
            </div>
            <p class="risk-item-detail">${item.detail}</p>
            ${item.data?.warning ? `<p class="risk-item-detail" style="color:var(--warning);margin-top:4px">${item.data.warning}</p>` : ''}
          </div>`;
      }).join('')}
    </div>
    <div class="risk-recommendation">
      <strong>📋 Recommendation:</strong> ${risk.recommendation || 'Monitor conditions regularly.'}
    </div>`;
}

// ===== Pest Warnings =====
function renderPestWarnings(crops) {
  const container = document.getElementById('pestContent');
  const allWarnings = [];
  const seen = new Set();

  crops.forEach(crop => {
    (crop.pest_warnings || []).forEach(warning => {
      if (!seen.has(warning.id)) {
        seen.add(warning.id);
        allWarnings.push({ ...warning, crop_name: crop.crop });
      }
    });
  });

  if (allWarnings.length === 0) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">✅</div><p>No pest or disease warnings for current conditions</p></div>`;
    return;
  }

  container.innerHTML = `
    <div class="pest-list">
      ${allWarnings.slice(0, 6).map(w => {
        const isHigh = w.severity === 'High' || w.severity === 'Critical';
        return `
          <div class="pest-item ${isHigh ? 'severity-high' : ''}">
            <div class="pest-item-header">
              <span class="pest-item-name">${w.type === 'pest' ? '🐛' : '🦠'} ${w.name}</span>
              <span class="crop-risk-badge risk-${(w.severity || 'low').toLowerCase()}">${w.severity}</span>
            </div>
            <p class="pest-item-type">${w.type} · Affects: ${w.crop_name}</p>
            <p class="pest-item-desc">${w.description}</p>
            <div class="pest-item-prevention"><strong>Prevention:</strong> ${w.prevention}</div>
          </div>`;
      }).join('')}
    </div>`;
}

// ===== Planting Calendar (Visual Timeline) =====
function renderCalendar(calendars) {
  const container = document.getElementById('calendarContent');
  if (!calendars || calendars.length === 0) {
    container.innerHTML = `<div class="empty-state"><div class="empty-state-icon">📅</div><p>No planting calendar data available</p></div>`;
    return;
  }

  container.innerHTML = `
    <div class="calendar-list">
      ${calendars.slice(0, 5).map(cal => {
        const phases = cal.phases || [];
        const emoji = CROP_EMOJIS[cal.crop_id] || '🌱';

        const phaseBar = phases.map((phase, i) => {
          const color = PHASE_COLORS[Math.min(i, PHASE_COLORS.length - 1)];
          return `<div class="calendar-phase"
              style="flex:${phase.progress_pct};background:${color}"
              title="${phase.name}: ${phase.duration_days} days">
              ${phase.progress_pct > 12 ? phase.name : ''}
            </div>`;
        }).join('');

        const phaseLegend = phases.map((phase, i) => {
          const color = PHASE_COLORS[Math.min(i, PHASE_COLORS.length - 1)];
          return `<span class="calendar-legend-item">
              <span class="calendar-legend-dot" style="background:${color}"></span>
              ${phase.name} (${phase.duration_days}d)
            </span>`;
        }).join('');

        return `
          <div class="calendar-item">
            <div class="calendar-item-header">
              <span class="calendar-item-crop">${emoji} ${cal.crop_name}</span>
              <span class="calendar-item-dates">${cal.sowing_date} → ${cal.harvest_date} (${cal.total_duration_days} days)</span>
            </div>
            <div class="calendar-phases">${phaseBar}</div>
            <div class="calendar-legend">${phaseLegend}</div>
          </div>`;
      }).join('')}
    </div>`;
}

// ================================================================
// CHAT WIDGET — Multi-turn, Markdown, Suggestion Chips
// ================================================================

// Chat state
let _chatHistory   = [];   // [{role:"user"|"model", parts:[text]}, ...]
let _chatCropCtx   = '';   // Top crops from last recommendation

// ── Inline Markdown Renderer (no external library) ──────────────
function _renderMarkdown(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text* (not followed by another *)
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

  // Convert lines to process lists and paragraphs
  const lines = html.split(/\r?\n/);
  const out = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (/^[-•*]\s+/.test(line)) {
      // Bullet list item
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push('<li>' + line.replace(/^[-•*]\s+/, '') + '</li>');
    } else if (/^\d+\.\s+/.test(line)) {
      // Numbered list item
      if (!inList) { out.push('<ol>'); inList = true; }
      out.push('<li>' + line.replace(/^\d+\.\s+/, '') + '</li>');
    } else {
      if (inList) { out.push(inList === true ? '</ul>' : '</ol>'); inList = false; }
      if (line === '') {
        out.push(''); // blank line → paragraph break handled below
      } else {
        out.push('<p>' + line + '</p>');
      }
    }
  }
  if (inList) out.push('</ul>');

  return out.join('\n');
}

// ── Update turn counter badge ────────────────────────────────────
function _updateTurnBadge() {
  const badge = document.getElementById('chatTurnBadge');
  if (!badge) return;
  const turns = Math.floor(_chatHistory.length / 2);
  badge.textContent = turns > 0 ? `${turns} turn${turns > 1 ? 's' : ''}` : '';
}

// ── Show suggestion chips after recommendation ───────────────────
function _showSuggestionChips(crops) {
  const chipsEl = document.getElementById('chatChips');
  if (!chipsEl || !crops || crops.length === 0) return;

  const top = crops[0]?.crop || 'your top crop';

  const suggestions = [
    `🌡\ufe0f What is today's temperature in ${top ? 'my district' : 'this region'}?`,
    `💧 How much water does ${top} need?`,
    `📅 When is the best time to sow ${top}?`,
    `🐛 What pests should I watch out for?`,
    `🏛\ufe0f Which government schemes can help me?`,
    `💰 What is the market demand for ${top}?`,
  ];

  chipsEl.innerHTML = suggestions.map(s =>
    `<button class="chat-chip" onclick="sendChip(this)">${s}</button>`
  ).join('');
  chipsEl.classList.remove('hidden');
}

// ── Called when a chip is clicked ───────────────────────────────
function sendChip(btn) {
  const q = btn.textContent;
  const input = document.getElementById('chatInput');
  if (input) { input.value = q; }
  // Hide chips after selection so they don't clutter
  document.getElementById('chatChips').classList.add('hidden');
  sendChat();
}

// ── Clear conversation ───────────────────────────────────────────
function clearChat() {
  _chatHistory = [];
  _updateTurnBadge();

  const messages = document.getElementById('chatMessages');
  if (messages) {
    messages.innerHTML = `
      <div class="chat-message ai">
        <div class="chat-avatar">🌾</div>
        <div class="chat-bubble">
          Conversation cleared. Feel free to ask a new question about your crops, soil, pests, or farming practices!
        </div>
      </div>`;
  }

  // Re-show chips if context is still loaded
  if (_chatEnabled && _currentRegionId) {
    const chipsEl = document.getElementById('chatChips');
    if (chipsEl) chipsEl.classList.remove('hidden');
  }
}

// ── Send a chat message (streaming SSE) ─────────────────────────
async function sendChat() {
  const input    = document.getElementById('chatInput');
  const btn      = document.getElementById('chatSendBtn');
  const messages = document.getElementById('chatMessages');
  if (!input || !btn || !messages) return;

  const question = input.value.trim();
  if (!question) return;

  input.value  = '';
  btn.disabled = true;

  // Add user bubble
  messages.appendChild(_makeBubble('user', '🧑‍🌾', question, false));

  // Typing indicator shown until first token arrives
  const typingId = 'typing-' + Date.now();
  const typingEl = document.createElement('div');
  typingEl.id        = typingId;
  typingEl.className = 'chat-message ai';
  typingEl.innerHTML = `
    <div class="chat-avatar">🌾</div>
    <div class="chat-bubble typing">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>`;
  messages.appendChild(typingEl);
  messages.scrollTop = messages.scrollHeight;

  // AI bubble that will grow as tokens stream in
  let aiBubbleEl  = null;  // created on first token
  let rawText     = '';    // accumulated plain text
  let firstToken  = true;

  function _ensureAiBubble() {
    if (aiBubbleEl) return;
    // Replace typing indicator with a real bubble
    const typingNode = document.getElementById(typingId);
    if (typingNode) typingNode.remove();

    const div = document.createElement('div');
    div.className = 'chat-message ai';
    div.innerHTML = `<div class="chat-avatar">🌾</div><div class="chat-bubble stream-bubble"></div>`;
    messages.appendChild(div);
    aiBubbleEl = div.querySelector('.chat-bubble');
  }

  try {
    const payload = {
      question,
      region_id:    _currentRegionId || '',
      season:       _currentSeason   || '',
      history:      _chatHistory,
      crop_context: _chatCropCtx     || '',
    };
    // If we have a live weather summary (fetched from /weather/now or /recommend),
    // include it in the payload so the backend always has accurate temperature data.
    // The backend will use it to warm the cache before answering the question.
    if (_liveWeatherSummary && !_liveWeatherSummary.startsWith('Avg temp')) {
      payload.weather_summary_hint = _liveWeatherSummary;
    }

    const res = await fetch('/chat/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });

    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();   // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const chunk = line.slice(6);   // strip 'data: '

        if (chunk.startsWith('[DONE]')) {
          // Parse updated history after [DONE]
          try {
            const histJson = chunk.slice(6);
            if (histJson && histJson !== '[]') {
              _chatHistory = JSON.parse(histJson);
            } else {
              _chatHistory.push({ role: 'user',  parts: [question] });
              _chatHistory.push({ role: 'model', parts: [rawText]  });
            }
          } catch (_) {
            _chatHistory.push({ role: 'user',  parts: [question] });
            _chatHistory.push({ role: 'model', parts: [rawText]  });
          }
          _updateTurnBadge();

          // Replace streamed plain text with full markdown rendering
          if (aiBubbleEl) aiBubbleEl.innerHTML = _renderMarkdown(rawText);
          break;
        }

        if (chunk.startsWith('[ERROR]')) {
          const errMsg = chunk.slice(7);
          _ensureAiBubble();
          if (aiBubbleEl) aiBubbleEl.innerHTML = '⚠️ ' + _escapeHtml(errMsg);
          break;
        }

        // Regular token — un-escape newlines and append
        const token = chunk.replace(/\\n/g, '\n');
        rawText += token;

        if (firstToken) {
          firstToken = false;
          _ensureAiBubble();
        }

        // Live update with escaped plain text while streaming
        if (aiBubbleEl) {
          aiBubbleEl.innerHTML = _escapeHtml(rawText);
          messages.scrollTop = messages.scrollHeight;
        }
      }
    }

  } catch (err) {
    const typingNode = document.getElementById(typingId);
    if (typingNode) typingNode.remove();
    if (!aiBubbleEl) {
      messages.appendChild(_makeBubble('ai', '🌾', '⚠️ Chat unavailable — ensure GEMINI_API_KEY is set in .env', false));
    }
  } finally {
    btn.disabled = false;
    input.focus();
    messages.scrollTop = messages.scrollHeight;
  }
}

// ── Build a chat bubble ──────────────────────────────────────────
function _makeBubble(role, avatar, text, useMarkdown) {
  const div = document.createElement('div');
  div.className = `chat-message ${role}`;
  const content = useMarkdown ? _renderMarkdown(text) : _escapeHtml(text);
  div.innerHTML = `
    <div class="chat-avatar">${avatar}</div>
    <div class="chat-bubble">${content}</div>`;
  return div;
}

function _escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}
