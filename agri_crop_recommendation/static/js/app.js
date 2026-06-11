/**
 * AI Crop Advisor v3.0 — Frontend JS
 * Global location selection, LLaMA agent calls,
 * animated dashboard, 6-month chart, floating chatbot
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const STATE = {
  countryCode: '', countryName: '',
  stateCode:   '', stateName:   '',
  district:    '',
  lat: null, lon: null,
  analysisData: null,
  chatHistory: [],
  chatOpen: false,
  chatRegionKey: '',
  allCrops: [],
};

// ── CROP EMOJIS ────────────────────────────────────────────────────────────
const CROP_EMOJIS = {
  rice: '🌾', wheat: '🌾', maize: '🌽', corn: '🌽', soybean: '🫘',
  soya: '🫘', cotton: '🌸', sugarcane: '🎋', potato: '🥔',
  tomato: '🍅', onion: '🧅', garlic: '🧄', chilli: '🌶️', pepper: '🌶️',
  mango: '🥭', banana: '🍌', coconut: '🥥', cashew: '🥜', groundnut: '🥜',
  peanut: '🥜', mustard: '🌻', sunflower: '🌻', barley: '🌾', sorghum: '🌾',
  bajra: '🌾', jowar: '🌾', ragi: '🌾', tur: '🫘', arhar: '🫘',
  moong: '🫘', urad: '🫘', chickpea: '🫘', lentil: '🫘', dal: '🫘',
  coffee: '☕', tea: '🍵', jute: '🌿', rubber: '🌿', tobacco: '🌿',
  turmeric: '🟡', ginger: '🫚', cumin: '🌿', coriander: '🌿',
  apple: '🍎', orange: '🍊', lemon: '🍋', grapes: '🍇', watermelon: '🍉',
};

function getCropEmoji(name) {
  const n = (name || '').toLowerCase();
  for (const [k, v] of Object.entries(CROP_EMOJIS)) {
    if (n.includes(k)) return v;
  }
  return '🌱';
}

// ── COUNTRY FLAGS ──────────────────────────────────────────────────────────
function getCountryFlag(code) {
  if (!code || code.length !== 2) return '🌍';
  const points = [...code.toUpperCase()].map(c => 127397 + c.charCodeAt(0));
  try { return String.fromCodePoint(...points); } catch { return '🌍'; }
}

// ── DOM HELPERS ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => { const e = $(id); if (e) e.classList.remove('hidden'); };
const hide = id => { const e = $(id); if (e) e.classList.add('hidden'); };

// ── INIT ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadCountries();
  $('country').addEventListener('change', onCountryChange);
  $('state').addEventListener('change', onStateChange);
  $('district').addEventListener('change', onDistrictChange);
  $('analyzeForm').addEventListener('submit', onAnalyzeSubmit);

  // Show chat unread hint after 3s
  setTimeout(() => { $('unreadBadge').style.display = 'flex'; }, 3000);
});

// ── LOCATION LOADERS ───────────────────────────────────────────────────────
async function loadCountries() {
  try {
    const res  = await fetch('/api/countries');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const sel  = $('country');
    sel.innerHTML = '<option value="">🌍 Select Country...</option>';
    (data.countries || []).forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.code;                          // just the code
      opt.dataset.name = c.name;                   // name in data attr
      opt.textContent = `${getCountryFlag(c.code)} ${c.name}`;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load countries:', e);
    $('country').innerHTML = '<option value="">Error loading countries</option>';
  }
}

async function onCountryChange() {
  const sel = $('country');
  const code = sel.value;
  if (!code) return;
  const selectedOpt = sel.options[sel.selectedIndex];
  STATE.countryCode = code;
  STATE.countryName = selectedOpt ? selectedOpt.dataset.name : '';

  // Reset downstream
  const stateSel = $('state'), distSel = $('district');
  stateSel.innerHTML = '<option value="">Loading states...</option>';
  stateSel.disabled = true;
  distSel.innerHTML  = '<option value="">Select state first...</option>';
  distSel.disabled   = true;
  STATE.stateCode = STATE.stateName = STATE.district = '';

  try {
    const res  = await fetch(`/api/states/${code}`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    stateSel.innerHTML = '<option value="">🗺️ Select State...</option>';
    (data.states || []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.code;
      opt.dataset.name = s.name;
      opt.textContent = s.name;
      stateSel.appendChild(opt);
    });
    stateSel.disabled = false;
  } catch (e) {
    stateSel.innerHTML = '<option value="">No states data</option>';
  }
}

async function onStateChange() {
  const sel  = $('state');
  const code = sel.value;
  if (!code) return;
  const selectedOpt = sel.options[sel.selectedIndex];
  STATE.stateCode = code;
  STATE.stateName = selectedOpt ? (selectedOpt.dataset.name || code) : code;

  const distSel = $('district');
  distSel.innerHTML = '<option value="">Loading districts...</option>';
  distSel.disabled  = true;
  STATE.district = '';
  STATE.lat = null;
  STATE.lon = null;

  const url = `/api/districts/${STATE.countryCode}/${code}`;
  console.log('[Districts] fetching:', url);

  try {
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error('HTTP ' + res.status + ' — ' + (err.detail || 'error'));
    }
    const data = await res.json();
    console.log('[Districts] got', data.districts?.length, 'districts');
    distSel.innerHTML = '<option value="">📍 Select District/Region...</option>';
    (data.districts || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value       = d.name;
      opt.dataset.lat = d.lat;
      opt.dataset.lon = d.lon;
      opt.textContent = d.name;
      distSel.appendChild(opt);
    });
    distSel.disabled = false;
  } catch (e) {
    console.error('[Districts] error:', e);
    distSel.innerHTML = `<option value="">⚠ Failed: ${e.message}</option>`;
    distSel.disabled = false;
  }
}


function onDistrictChange() {
  const sel = $('district');
  const name = sel.value;
  if (!name) return;
  const selectedOpt = sel.options[sel.selectedIndex];
  STATE.district = name;
  STATE.lat = selectedOpt ? parseFloat(selectedOpt.dataset.lat) || null : null;
  STATE.lon = selectedOpt ? parseFloat(selectedOpt.dataset.lon) || null : null;
  console.log('[District selected]', STATE.district, STATE.lat, STATE.lon);
}

// ── ANALYZE SUBMIT — SSE Streaming ────────────────────────────────────────────
async function onAnalyzeSubmit(e) {
  e.preventDefault();

  if (!STATE.countryCode || !STATE.stateCode || !STATE.district) {
    alert('Please select Country, State, and District.');
    return;
  }

  showProgress();
  const btn = $('analyzeBtn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.querySelector('.btn-text').textContent = 'Agent Working...';

  const payload = {
    country_code: STATE.countryCode,
    country_name: STATE.countryName,
    state_code:   STATE.stateCode,
    state_name:   STATE.stateName,
    district:     STATE.district,
    lat:          STATE.lat,
    lon:          STATE.lon,
    irrigation:   $('irrigation').value,
    planning_days:parseInt($('planning_days').value) || 90,
    soil_texture: $('soil_texture').value || null,
    soil_ph:      $('soil_ph').value ? parseFloat($('soil_ph').value) : null,
  };

  try {
    // ── Use streaming endpoint so progress is driven by real backend events ──
    const res = await fetch('/api/analyze/stream', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Analysis failed (HTTP ' + res.status + ')');
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';
    let   finalData = null;

    // Read SSE chunks
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by double newline
      const parts = buffer.split('\n\n');
      buffer = parts.pop(); // keep incomplete last part

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.slice(5).trim();
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        if (event.step === 'error') {
          throw new Error(event.msg || 'Agent error');
        }

        if (event.step === 'done') {
          finalData = event.data;
          // Advance all remaining steps to done
          updateProgress(5, 100, event.msg || 'Analysis complete!');
          break;
        }

        // Regular progress update — advance stepper to this step
        if (typeof event.step === 'number') {
          updateProgress(event.step, event.pct, event.msg);
        }
      }

      if (finalData) break;
    }

    if (!finalData) throw new Error('No analysis result received from server');

    STATE.analysisData = finalData;
    STATE.chatRegionKey = `${STATE.countryCode}_${STATE.stateCode}_${STATE.district}`.toUpperCase();

    finishProgress();
    setTimeout(() => renderDashboard(finalData), 600);

  } catch (err) {
    hideProgress();
    alert('Analysis failed: ' + err.message);
    console.error('[Analyze]', err);
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = 'Analyze with AI Agent';
  }
}

// ── PROGRESS STEPPER — Real SSE-driven ────────────────────────────────────────
// Steps are indexed 1-5 matching the backend SSE step numbers.
const STEPS = [
  { id: 'step1', label: 'Resolve Location' },
  { id: 'step2', label: 'Live Weather'     },
  { id: 'step3', label: 'Forecast'         },
  { id: 'step4', label: 'Soil & Prices'    },
  { id: 'step5', label: 'Crop Analysis'    },
];
let _activeStep = 0; // 0 = none active yet

function showProgress() {
  show('agentProgress');
  hide('dashboard');
  $('progressLocation').textContent = STATE.district + ', ' + STATE.stateName;
  // Reset all steps
  STEPS.forEach(s => {
    const el = $(s.id);
    if (el) { el.classList.remove('active', 'done'); }
  });
  $('progressBar').style.width = '0%';
  $('progressMessage').textContent = 'Starting analysis…';
  _activeStep = 0;
}

/**
 * updateProgress(stepNum, pct, msg)
 * Called by the SSE reader when the backend emits a progress event.
 * stepNum: 1–5 matching backend steps.
 * pct: 0–100 progress bar percentage.
 * msg: human-readable status message.
 */
function updateProgress(stepNum, pct, msg) {
  // Mark all steps before this one as done
  for (let i = 1; i < stepNum; i++) {
    const el = $('step' + i);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  }
  // Mark the current step as active
  const cur = $('step' + stepNum);
  if (cur) { cur.classList.remove('done'); cur.classList.add('active'); }
  // Mark steps after this as neither active nor done
  for (let i = stepNum + 1; i <= STEPS.length; i++) {
    const el = $('step' + i);
    if (el) { el.classList.remove('active', 'done'); }
  }
  // Update bar and message
  $('progressBar').style.width = pct + '%';
  $('progressMessage').textContent = msg || '';
  _activeStep = stepNum;
}

function finishProgress() {
  STEPS.forEach(s => {
    const el = $(s.id);
    if (el) { el.classList.remove('active'); el.classList.add('done'); }
  });
  $('progressBar').style.width = '100%';
  $('progressMessage').textContent = 'Analysis complete! Loading dashboard…';
}

function hideProgress() {
  hide('agentProgress');
}

// ── RENDER DASHBOARD ────────────────────────────────────────────────────────
function renderDashboard(data) {
  hide('agentProgress');

  const loc      = data.location || {};
  const gathered = data.gathered_data || {};
  const current  = gathered.current || {};
  const forecast = gathered.forecast_6month || [];
  const soil     = gathered.soil || {};
  const market   = gathered.market_prices || {};
  const crops    = data.recommended_crops || [];

  // Location banner
  const flag = getCountryFlag(loc.country_code);
  $('locationFlag').textContent = flag;
  $('locationName').textContent = `${loc.district || '-'}, ${loc.state_name || '-'}`;
  $('locationMeta').textContent = `${loc.country_name || '-'} · ${loc.lat?.toFixed(3) || ''}°N, ${loc.lon?.toFixed(3) || ''}°E`;
  $('locSeason').textContent  = gathered.season  || '-';
  $('locClimate').textContent = gathered.climate_zone || '-';

  // Metric cards — animated count-up
  animateCount('val-temp',     current.temperature_c, 1);
  animateCount('val-humidity', current.humidity_pct,  0);
  animateCount('val-rain',     current.rainfall_7d_mm,1);
  animateCount('val-soil',     current.soil_temp_c,   1);
  animateCount('val-wind',     current.wind_kmh,      0);
  animateCount('val-uv',       current.uv_index,      1);

  $('sub-temp').textContent  = `Feels ${current.feels_like_c || '?'}°C`;
  $('sub-humidity').textContent = getHumidityDesc(current.humidity_pct);
  $('sub-uv').textContent = getUVDesc(current.uv_index);
  $('sub-soil').textContent  = 'Surface layer';

  // Soil card
  $('soil-type').textContent     = soil.type     || 'Unknown';
  $('soil-ph').textContent       = soil.ph       ? `pH ${soil.ph}` : '-';
  $('soil-organic').textContent  = soil.organic_matter || '-';
  $('soil-drainage').textContent = soil.drainage || '-';
  $('districtSummary').textContent = gathered.district_summary || '';

  // Forecast table
  renderForecastTable(forecast);

  // Market prices
  renderMarketPrices(market);

  // Crops (categorized)
  renderCrops(crops);

  // Chat context
  updateChatContext(loc, gathered);

  // Show dashboard
  const dash = $('dashboard');
  dash.classList.remove('hidden');
  dash.classList.add('dashboard');
  dash.scrollIntoView({behavior:'smooth'});

  // Show chat pill
  const pill = $('chatLocationPill');
  pill.style.display = 'block';
  pill.textContent = `${flag} ${loc.district}, ${loc.state_name}`;
}

// ── COUNT-UP ANIMATION ─────────────────────────────────────────────────────
function animateCount(id, target, decimals=0) {
  const el  = $(id);
  if (!el) return;
  const val = parseFloat(target) || 0;
  const dur = 1200;
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3); // cubic ease-out
    el.textContent = (val * ease).toFixed(decimals);
    if (t < 1) requestAnimationFrame(frame);
    else el.textContent = val.toFixed(decimals);
  }
  requestAnimationFrame(frame);
}

function getHumidityDesc(h) {
  if (!h) return '-';
  if (h < 30) return 'Very dry';
  if (h < 50) return 'Dry';
  if (h < 70) return 'Comfortable';
  if (h < 85) return 'Humid';
  return 'Very humid';
}

function getUVDesc(uv) {
  if (!uv) return '-';
  if (uv <= 2)  return 'Low';
  if (uv <= 5)  return 'Moderate';
  if (uv <= 7)  return 'High';
  if (uv <= 10) return 'Very High';
  return 'Extreme';
}

// ── FORECAST TABLE ──────────────────────────────────────────────────────────
function renderForecastTable(forecast) {
  const tbody = $('forecastTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!forecast || !forecast.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text3)">No forecast data available</td></tr>';
    return;
  }

  forecast.forEach((f, i) => {
    const row = document.createElement('tr');
    const rain = parseFloat(f.rainfall_mm) || 0;
    const rainClass = rain > 150 ? 'rain-high' : rain > 60 ? 'rain-med' : 'rain-low';
    const tempAvg = parseFloat(f.temp_avg) || '--';
    const tempMax = parseFloat(f.temp_max) || '--';
    const tempMin = parseFloat(f.temp_min) || '--';
    const hum    = parseFloat(f.humidity_pct) || '--';
    row.innerHTML = `
      <td class="ft-month">${f.month || '—'}</td>
      <td class="ft-cold">${tempMin !== '--' ? tempMin.toFixed(1) : '--'}°</td>
      <td class="ft-hot">${tempMax !== '--' ? tempMax.toFixed(1) : '--'}°</td>
      <td class="ft-avg">${tempAvg !== '--' ? tempAvg.toFixed(1) : '--'}°</td>
      <td class="ft-hum">${hum !== '--' ? hum.toFixed(0)+'%' : '--'}</td>
      <td class="ft-rain ${rainClass}">${rain.toFixed(0)} mm</td>
    `;
    tbody.appendChild(row);
  });
}

// ── MARKET PRICES ──────────────────────────────────────────────────────────
function renderMarketPrices(market) {
  const grid = $('marketGrid');
  grid.innerHTML = '';
  const entries = Object.entries(market);
  if (!entries.length) {
    grid.innerHTML = '<p style="color:var(--text3);font-size:0.82rem;">Market data being gathered by agent...</p>';
    return;
  }
  entries.forEach(([crop, price]) => {
    const div = document.createElement('div');
    div.className = 'market-item';
    div.innerHTML = `<div class="market-crop">${getCropEmoji(crop)} ${crop}</div><div class="market-price">${price}</div>`;
    grid.appendChild(div);
  });
}

// ── CROP CATEGORIES ────────────────────────────────────────────────────────
// PRIORITY ORDER matters: the first matching category wins.
// Spices & Pulses are checked BEFORE Vegetables to prevent mis-classification
// (e.g. pepper/chilli/pea would all fall into vegetables otherwise).
const CROP_CATEGORIES = {

  // 1. Spices & Herbs — checked first so pepper/chilli/fenugreek go here
  spices: [
    'pepper','black pepper','white pepper','long pepper','peppercorn',
    'chilli','chili','cayenne','paprika','red pepper',
    'turmeric','haldi','ginger','adrak','cumin','jeera',
    'coriander','dhania','cardamom','elaichi','clove','laung',
    'cinnamon','dalchini','nutmeg','jaiphal','fenugreek','methi',
    'ajwain','carom','fennel','saunf','saffron','kesar',
    'mint','pudina','basil','tulsi','oregano','thyme','rosemary','sage',
    'bay leaf','star anise','mace','asafoetida','hing',
    'lemongrass','curry leaf','kari patta','vanilla','stevia',
  ],

  // 2. Pulses & Beans — checked before vegetables so "pea" goes here not veg
  pulses: [
    'soybean','soya bean','chickpea','chana','gram','lentil','masoor',
    'moong','mung bean','green gram','urad','black gram','tur','arhar',
    'pigeon pea','toor','rajma','kidney bean','cowpea','lobia',
    'moth bean','horse gram','kulthi','field pea','pea','matar',
    'broad bean','fava bean','french bean','cluster bean','guar','dal',
  ],

  // 3. Fruits
  fruits: [
    'mango','aam','banana','kela','apple','seb','orange','narangi',
    'lemon','nimbu','lime','grapes','angur','watermelon','tarbooz',
    'papaya','papita','guava','amrud','pomegranate','anar',
    'pineapple','ananas','coconut','nariyal','strawberry','fig','anjeer',
    'date','khajur','litchi','lychee','mulberry','jamun','avocado',
    'kiwi','plum','ber','peach','aadoo','pear','nashpati',
    'apricot','cherry','chikoo','sapota','jackfruit','kathal',
    'passion fruit','dragon fruit','gooseberry','amla','custard apple',
  ],

  // 4. Cash Crops
  cash: [
    'cotton','kapas','sugarcane','ganna','tobacco','tambaku',
    'rubber','jute','coffee','tea','cocoa','cacao',
    'sunflower','surajmukhi','mustard','sarson','rapeseed','canola',
    'groundnut','mungfali','peanut','sesame','til',
    'castor','linseed','flaxseed','alsi','safflower','hemp',
  ],

  // 5. Grains & Cereals
  grains: [
    'rice','chawal','paddy','wheat','gehun','maize','makka','corn',
    'barley','jau','sorghum','jowar','bajra','pearl millet','ragi',
    'finger millet','oats','millets','foxtail millet','kodo millet',
    'buckwheat','quinoa','amaranth','rajgira','teff','swa',
  ],

  // 6. Vegetables — checked last (most general category)
  vegetables: [
    'tomato','tamatar','onion','pyaz','potato','aloo',
    'brinjal','baingan','eggplant','okra','bhindi','lady finger',
    'cabbage','cauliflower','carrot','gajar','radish','mooli',
    'spinach','palak','cucumber','kheera','pumpkin','kaddu',
    'bottle gourd','lauki','bitter gourd','karela','ridge gourd','tori',
    'capsicum','bell pepper','sweet pepper','beans','green beans',
    'lettuce','celery','beetroot','chukander','turnip','shalgam',
    'asparagus','broccoli','kale','leek','garlic','lahsun','spring onion',
  ],
};

function categorizeCrop(name) {
  const n = (name || '').toLowerCase();
  for (const [cat, keywords] of Object.entries(CROP_CATEGORIES)) {
    if (keywords.some(k => n.includes(k))) return cat;
  }
  return 'other';
}

function filterCrops(cat, btn) {
  // Update active tab
  document.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  // Show/hide cards
  document.querySelectorAll('.crop-card').forEach(card => {
    const cardCat = card.dataset.cat || 'other';
    card.style.display = (cat === 'all' || cardCat === cat) ? '' : 'none';
  });

  // Show empty message if nothing matches
  const grid = $('cropsGrid');
  const visible = [...document.querySelectorAll('.crop-card')].filter(c => c.style.display !== 'none');
  let noMsg = grid.querySelector('.no-cat-msg');
  if (visible.length === 0) {
    if (!noMsg) {
      noMsg = document.createElement('p');
      noMsg.className = 'no-cat-msg';
      noMsg.style.cssText = 'color:var(--text3);font-size:0.9rem;padding:1rem;grid-column:1/-1;text-align:center';
      grid.appendChild(noMsg);
    }
    noMsg.textContent = 'No crops in this category for the current analysis.';
  } else if (noMsg) {
    noMsg.remove();
  }
}

// ── CROPS ──────────────────────────────────────────────────────────────────
function renderCrops(crops) {
  const grid = $('cropsGrid');
  grid.innerHTML = '';
  STATE.allCrops = crops;
  $('crops-subtitle').textContent = `${crops.length} crops analyzed · Ranked by AI suitability score`;

  // Reset filter to 'All'
  document.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
  const allTab = document.querySelector('.cat-tab[data-cat="all"]');
  if (allTab) allTab.classList.add('active');

  crops.forEach((crop, i) => {
    const score = Math.min(Math.max(crop.suitability_score || 0, 0), 100);
    const scoreClass = score >= 75 ? 'score-high' : score >= 55 ? 'score-mid' : 'score-low';
    const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : 'rank-other';
    const rankLabel = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i+1}`;
    const reasons  = (crop.reasons || []).slice(0,3).map(r => `<li>${r}</li>`).join('');
    const warnings = (crop.warnings||[]).length ? `<p style="color:var(--gold);font-size:0.74rem;margin-top:6px">⚠️ ${crop.warnings[0]}</p>` : '';

    const cropCat = categorizeCrop(crop.crop_name);
    const card = document.createElement('div');
    card.className = 'crop-card';
    card.dataset.cat = cropCat;
    card.innerHTML = `
      <div class="crop-rank-badge ${rankClass}">${rankLabel}</div>
      <div class="crop-header">
        <div class="crop-emoji">${getCropEmoji(crop.crop_name)}</div>
        <div class="crop-title">
          <div class="crop-name">${crop.crop_name}</div>
          <div class="crop-local">${crop.local_name !== crop.crop_name ? crop.local_name : ''}</div>
        </div>
      </div>
      <div class="score-bar-wrap">
        <div class="score-label">
          <span>Suitability Score</span><span>${score}%</span>
        </div>
        <div class="score-bar">
          <div class="score-fill ${scoreClass}" style="width:0%" data-target="${score}"></div>
        </div>
      </div>
      <div class="crop-meta">
        <span class="crop-tag risk-${crop.risk_level || 'Medium'}">Risk: ${crop.risk_level||'Medium'}</span>
        <span class="crop-tag">${crop.duration_days||'?'} days</span>
        <span class="crop-tag">💧 ${crop.water_need||'Medium'}</span>
        <span class="crop-tag">📈 ${crop.market_demand||'Medium'} demand</span>
      </div>
      <ul class="crop-reasons">${reasons}</ul>
      ${warnings}
      <div class="crop-tip">💡 ${crop.growing_tip||''}</div>
      <div class="crop-window">🗓️ Plant: <strong>${crop.planting_window||'Current season'}</strong> · Yield: ${crop.estimated_yield||'N/A'}</div>
    `;
    grid.appendChild(card);

    // Animate score bar
    setTimeout(() => {
      const fill = card.querySelector('.score-fill');
      if (fill) fill.style.width = score + '%';
    }, 200 + i * 80);
  });
}

// ── CHAT ───────────────────────────────────────────────────────────────────
function updateChatContext(loc, gathered) {
  const chips = ['🌡️ Current weather?', '🌱 Best crop to plant now?', '💰 Market prices near me?',
                 '🐛 Any pest alerts?', '💧 How much water needed?', '📅 When to harvest?'];
  const chipsDiv = $('chatChips');
  chipsDiv.innerHTML = '';
  chips.forEach(c => {
    const chip = document.createElement('span');
    chip.className = 'chip'; chip.textContent = c;
    chip.onclick = () => { $('chatInput').value = c; sendChat(); };
    chipsDiv.appendChild(chip);
  });
}

function toggleChat() {
  STATE.chatOpen = !STATE.chatOpen;
  const panel = $('chatPanel');
  panel.classList.toggle('open', STATE.chatOpen);
  if (STATE.chatOpen) {
    $('unreadBadge').style.display = 'none';
    $('chatInput').focus();
  }
}

function clearChat() {
  STATE.chatHistory = [];
  $('chatMessages').innerHTML = `
    <div class="chat-msg ai">
      <div class="chat-avatar">🌾</div>
      <div class="chat-bubble-msg">Chat cleared. How can I help you with your farm?</div>
    </div>`;
}

async function sendChat() {
  const input = $('chatInput');
  const question = input.value.trim();
  if (!question) return;

  // Show user message
  appendChatMsg('user', question);
  input.value = '';

  // Show typing
  const typingId = 'typing_' + Date.now();
  appendChatMsg('ai', '...', typingId, true);

  try {
    // Build context from analysis data
    const loc     = STATE.analysisData?.location || {};
    const cropCtx = (STATE.analysisData?.recommended_crops || [])
      .slice(0,3).map(c => c.crop_name).join(', ');

    const body = {
      question,
      region_id:    STATE.chatRegionKey,
      season:       STATE.analysisData?.gathered_data?.season || '',
      history:      STATE.chatHistory,
      crop_context: cropCtx,
    };

    // Use streaming endpoint
    const res = await fetch('/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });

    if (!res.body) throw new Error('No streaming body');

    // Remove typing indicator, add streaming message
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    const msgEl = appendChatMsg('ai', '');
    const bubbleEl = msgEl.querySelector('.chat-bubble-msg');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload.startsWith('[DONE]')) {
          try {
            STATE.chatHistory = JSON.parse(payload.slice(6));
          } catch {}
          break;
        }
        if (payload.startsWith('[ERROR]')) break;
        const token = payload.replace(/\\n/g, '\n');
        fullText += token;
        bubbleEl.textContent = fullText;
        scrollChatToBottom();
      }
    }

  } catch (err) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    appendChatMsg('ai', 'Sorry, I had trouble connecting. Please try again.');
    console.error('Chat error:', err);
  }
}

function appendChatMsg(role, text, id, isTyping=false) {
  const msgs = $('chatMessages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}${isTyping?' typing':''}`;
  if (id) div.id = id;
  div.innerHTML = `
    ${role === 'ai' ? '<div class="chat-avatar">🌾</div>' : ''}
    <div class="chat-bubble-msg">${text}</div>
    ${role === 'user' ? '<div class="chat-avatar">👤</div>' : ''}
  `;
  msgs.appendChild(div);
  scrollChatToBottom();
  return div;
}

function scrollChatToBottom() {
  const msgs = $('chatMessages');
  msgs.scrollTop = msgs.scrollHeight;
}
