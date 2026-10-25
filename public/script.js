/**
 * AgriPredict AI — Frontend Application Logic (State-Aware)
 * Integrates directly with Flask backend REST API (Variant C: State-Aware Architecture)
 */

// 1. API Base URL configuration
const API_BASE_URL = (typeof window !== 'undefined' && (window.location.protocol === 'file:' || window.location.port === '8000'))
  ? 'http://localhost:5000'
  : '';

// Global state
let appOptions = null;
let seasonalChartInstance = null;
let reconnectTimer = null;
let isFetchingInitialData = false;
let lastServerCheckFailed = false;

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSeasonalChart();
  initFormValidation();
  loadInitialData();
});

/* ==========================================================================
   2. DATA LOADING & INITIALIZATION
   ========================================================================== */

/**
 * Loads dynamic options, cascading combinations, weather stats, and model metrics.
 * Auto-reconnects every 4 seconds in the background if the server is unreachable.
 */
async function loadInitialData() {
  if (isFetchingInitialData) return;
  isFetchingInitialData = true;

  const serverErrorBanner = document.getElementById('serverErrorBanner');
  const stateSelect = document.getElementById('stateSelect');
  const cropSelect = document.getElementById('cropSelect');
  const seasonSelect = document.getElementById('seasonSelect');

  try {
    const optionsResp = await fetch(`${API_BASE_URL}/options`);
    if (!optionsResp.ok) {
      throw new Error(`Server returned HTTP ${optionsResp.status}`);
    }
    appOptions = await optionsResp.json();
    lastServerCheckFailed = false;

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    if (serverErrorBanner) {
      serverErrorBanner.style.display = 'none';
    }

    // Populate State dropdown
    if (stateSelect && Array.isArray(appOptions.states)) {
      stateSelect.innerHTML = '<option value="" disabled selected>Select state</option>';
      appOptions.states.forEach(st => {
        const opt = document.createElement('option');
        opt.value = st;
        opt.textContent = st;
        stateSelect.appendChild(opt);
      });
    }

    // Reset Crop dropdown until state is selected
    if (cropSelect) {
      cropSelect.innerHTML = '<option value="" disabled selected>Select state first</option>';
      cropSelect.disabled = true;
    }

    // Reset Season dropdown until crop is selected
    if (seasonSelect) {
      seasonSelect.innerHTML = '<option value="" disabled selected>Select crop first</option>';
      seasonSelect.disabled = true;
    }

    // Setup input hard limits
    if (appOptions.hard_limits) {
      setupInputRanges(appOptions.hard_limits);
    }

    // Initial trends chart with national wheat or first available state/crop
    if (appOptions.states && appOptions.states.length > 0) {
      const defaultState = appOptions.states.includes('Punjab') ? 'Punjab' : appOptions.states[0];
      const validCropsForDefault = appOptions.valid_combos && appOptions.valid_combos[defaultState]
        ? Object.keys(appOptions.valid_combos[defaultState])
        : [];
      const defaultCrop = validCropsForDefault.includes('Wheat') ? 'Wheat' : validCropsForDefault[0];

      if (defaultCrop) {
        await loadSeasonalTrends(defaultCrop, defaultState);
      }
    }

    // Load model evaluation metrics for footer
    loadModelMetrics();

  } catch (err) {
    console.error('Failed to load options from backend:', err);
    lastServerCheckFailed = true;

    if (serverErrorBanner) {
      serverErrorBanner.innerHTML = 'Cannot reach the prediction server. Start the backend and try again. <a href="#" id="retryNowLink" style="color: #991B1B; font-weight: 600; text-decoration: underline; margin-left: 8px; cursor: pointer;">Retry now</a>';
      serverErrorBanner.style.display = 'block';

      const retryLink = document.getElementById('retryNowLink');
      if (retryLink) {
        retryLink.onclick = (e) => {
          e.preventDefault();
          if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
          }
          loadInitialData();
        };
      }
    }

    if (stateSelect) {
      stateSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }
    if (cropSelect) {
      cropSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }
    if (seasonSelect) {
      seasonSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }
    reconnectTimer = setTimeout(() => {
      loadInitialData();
    }, 4000);

  } finally {
    isFetchingInitialData = false;
  }
}

/**
 * Configure hard limits and default helper hints for numerical inputs.
 */
function setupInputRanges(limits) {
  const rainInput = document.getElementById('rainfallInput');
  const rainHint = document.getElementById('hint-rainfall');
  if (rainInput) {
    rainInput.min = limits.rainfall_min !== undefined ? limits.rainfall_min : 0;
    rainInput.max = limits.rainfall_max !== undefined ? limits.rainfall_max : 5000;
    if (rainHint && !rainHint.dataset.custom) {
      rainHint.textContent = `Hard limit: ${rainInput.min} – ${rainInput.max} mm`;
    }
  }

  const tempInput = document.getElementById('tempInput');
  const tempHint = document.getElementById('hint-temperature');
  if (tempInput) {
    tempInput.min = limits.temperature_min !== undefined ? limits.temperature_min : -5;
    tempInput.max = limits.temperature_max !== undefined ? limits.temperature_max : 45;
    if (tempHint && !tempHint.dataset.custom) {
      tempHint.textContent = `Hard limit: ${tempInput.min} – ${tempInput.max} °C`;
    }
  }

  const areaInput = document.getElementById('areaInput');
  const areaHint = document.getElementById('hint-area');
  if (areaInput) {
    areaInput.min = limits.area_min !== undefined ? limits.area_min : 0.1;
    areaInput.max = limits.area_max !== undefined ? limits.area_max : 10000;
    if (areaHint) {
      areaHint.textContent = `Allowed range: ${areaInput.min} – ${areaInput.max} ha`;
    }
  }
}

/**
 * Fetches /metrics and builds dynamic summary line for footer.
 */
async function loadModelMetrics() {
  const footerMetrics = document.getElementById('footerMetrics');
  if (!footerMetrics) return;

  try {
    const resp = await fetch(`${API_BASE_URL}/metrics`);
    if (!resp.ok) return;
    const data = await resp.json();

    const valMeta = data.validation_metrics || {};
    const meanWindows = valMeta.mean_over_three_rolling_windows_2007_2015 || {};
    const modelStats = meanWindows.model_variant_C || {};

    const maeVal = typeof modelStats.mae === 'number' ? modelStats.mae.toFixed(2) : '0.80';
    const relMaePct = typeof modelStats.macro_crop_rel_mae_pct === 'number' ? Math.round(modelStats.macro_crop_rel_mae_pct) : 31;

    footerMetrics.textContent = `Model: HistGradientBoosting (state-aware) | MAE ${maeVal} t/ha | typical error about ${relMaePct}% of a crop's average yield (rolling validation 2007-2015)`;
  } catch (err) {
    console.warn('Metrics endpoint unavailable:', err);
  }
}

/* ==========================================================================
   3. SEASONAL CHART & TRENDS INTEGRATION
   ========================================================================== */

function initSeasonalChart() {
  const canvas = document.getElementById('seasonalChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, 'rgba(74, 139, 82, 0.22)');
  gradient.addColorStop(1, 'rgba(74, 139, 82, 0.01)');

  seasonalChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Historical Yield (t/ha)',
          data: [],
          borderColor: '#4A8B52',
          borderWidth: 3,
          backgroundColor: gradient,
          fill: true,
          tension: 0.35,
          pointRadius: 6,
          pointHoverRadius: 8,
          pointBackgroundColor: ['#4A8B52'],
          pointBorderColor: '#FFFFFF',
          pointBorderWidth: 2.5,
          pointHoverBorderWidth: 3,
          pointHoverBackgroundColor: ['#4A8B52']
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#173A2E',
          titleFont: { family: 'DM Sans', size: 13, weight: '600' },
          bodyFont: { family: 'DM Sans', size: 12 },
          padding: 10,
          cornerRadius: 8,
          displayColors: false,
          callbacks: {
            label: function (context) {
              return `Average yield: ${context.parsed.y.toFixed(2)} t/ha`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false, drawBorder: false },
          ticks: {
            color: '#6B7280',
            font: { family: 'DM Sans', size: 12.5, weight: '500' },
            padding: 8
          }
        },
        y: {
          beginAtZero: false,
          grid: { color: '#EFECE4', drawBorder: false, lineWidth: 1 },
          ticks: {
            color: '#6B7280',
            font: { family: 'DM Sans', size: 11.5 },
            padding: 8,
            callback: function (val) {
              return val + ' t/ha';
            }
          }
        }
      },
      animation: {
        duration: 650,
        easing: 'easeInOutQuart'
      }
    }
  });
}

/**
 * Fetches historical seasonal trends for a crop and state and updates Chart.js.
 */
async function loadSeasonalTrends(cropName, stateName) {
  if (!cropName) return;

  try {
    let url = `${API_BASE_URL}/trends?crop=${encodeURIComponent(cropName)}`;
    if (stateName) {
      url += `&state=${encodeURIComponent(stateName)}`;
    }

    const resp = await fetch(url);
    if (!resp.ok) return;
    const data = await resp.json();

    const trends = data.trends || [];
    const validSeasons = data.valid_seasons || [];

    const badgeText = document.getElementById('chartDataBadgeText');
    if (badgeText) {
      badgeText.textContent = stateName ? `Historical data (${stateName})` : 'National historical data';
    }

    let bestSeason = validSeasons[0] || 'Kharif';
    let maxYield = -Infinity;
    trends.forEach(item => {
      if (item.average_yield > maxYield) {
        maxYield = item.average_yield;
        bestSeason = item.season;
      }
    });

    const highlightTitle = document.getElementById('highlightTitle');
    const highlightDesc = document.getElementById('highlightDesc');
    if (highlightTitle) {
      highlightTitle.textContent = `Best outlook: ${bestSeason} season`;
    }
    if (highlightDesc) {
      const locationLabel = stateName ? ` in ${stateName}` : ' nationally';
      highlightDesc.textContent = `Highest average historical yield for ${cropName}${locationLabel}.`;
    }

    if (seasonalChartInstance) {
      const labels = trends.map(t => t.season);
      const yields = trends.map(t => t.average_yield);

      seasonalChartInstance.data.labels = labels;
      seasonalChartInstance.data.datasets[0].data = yields;

      const chartLegendText = document.getElementById('chartLegendText');

      if (trends.length <= 1) {
        seasonalChartInstance.data.datasets[0].showLine = false;
        seasonalChartInstance.data.datasets[0].pointRadius = 8;
        seasonalChartInstance.data.datasets[0].pointHoverRadius = 10;
        seasonalChartInstance.data.datasets[0].pointBackgroundColor = ['#D9A441'];
        seasonalChartInstance.data.datasets[0].pointHoverBackgroundColor = ['#D9A441'];

        if (chartLegendText) {
          const singleSeason = trends[0] ? trends[0].season : 'Single Season';
          chartLegendText.textContent = `Single season crop (${singleSeason})${stateName ? ' in ' + stateName : ''} · historical average yield`;
        }
      } else {
        seasonalChartInstance.data.datasets[0].showLine = true;
        seasonalChartInstance.data.datasets[0].borderWidth = 3;
        seasonalChartInstance.data.datasets[0].pointRadius = 6;
        seasonalChartInstance.data.datasets[0].pointHoverRadius = 8;

        const bestIdx = labels.indexOf(bestSeason);
        const pointColors = labels.map((_, i) => (i === bestIdx ? '#D9A441' : '#4A8B52'));
        seasonalChartInstance.data.datasets[0].pointBackgroundColor = pointColors;
        seasonalChartInstance.data.datasets[0].pointHoverBackgroundColor = pointColors;

        if (chartLegendText) {
          chartLegendText.textContent = stateName
            ? `Historical yield in ${stateName} · tonnes per hectare`
            : 'Historical yield · tonnes per hectare';
        }
      }

      seasonalChartInstance.update();
    }
  } catch (err) {
    console.error(`Failed to load trends for crop ${cropName}:`, err);
  }
}

/* ==========================================================================
   4. CASCADING DROPDOWNS & PRE-FILLING TYPICAL WEATHER
   ========================================================================== */

function handleStateChange() {
  const stateSelect = document.getElementById('stateSelect');
  const cropSelect = document.getElementById('cropSelect');
  const seasonSelect = document.getElementById('seasonSelect');
  const rainfallInput = document.getElementById('rainfallInput');
  const tempInput = document.getElementById('tempInput');
  const hintRain = document.getElementById('hint-rainfall');
  const hintTemp = document.getElementById('hint-temperature');

  if (!stateSelect || !cropSelect || !seasonSelect || !appOptions) return;

  const state = stateSelect.value;
  if (!state || !appOptions.valid_combos || !appOptions.valid_combos[state]) {
    cropSelect.innerHTML = '<option value="" disabled selected>Select state first</option>';
    cropSelect.disabled = true;
    seasonSelect.innerHTML = '<option value="" disabled selected>Select crop first</option>';
    seasonSelect.disabled = true;
    return;
  }

  // Populate valid crops for this state
  const validCrops = Object.keys(appOptions.valid_combos[state]).sort();
  cropSelect.innerHTML = '<option value="" disabled selected>Select crop</option>';
  validCrops.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    cropSelect.appendChild(opt);
  });
  cropSelect.disabled = false;

  // Reset Season
  seasonSelect.innerHTML = '<option value="" disabled selected>Select crop first</option>';
  seasonSelect.disabled = true;

  // Reset weather inputs & hints
  if (rainfallInput) rainfallInput.value = '';
  if (tempInput) tempInput.value = '';
  if (hintRain) {
    delete hintRain.dataset.custom;
    hintRain.textContent = appOptions.hard_limits ? `Hard limit: ${appOptions.hard_limits.rainfall_min} – ${appOptions.hard_limits.rainfall_max} mm` : '';
  }
  if (hintTemp) {
    delete hintTemp.dataset.custom;
    hintTemp.textContent = appOptions.hard_limits ? `Hard limit: ${appOptions.hard_limits.temperature_min} – ${appOptions.hard_limits.temperature_max} °C` : '';
  }
}

function handleCropChange() {
  const stateSelect = document.getElementById('stateSelect');
  const cropSelect = document.getElementById('cropSelect');
  const seasonSelect = document.getElementById('seasonSelect');

  if (!stateSelect || !cropSelect || !seasonSelect || !appOptions) return;

  const state = stateSelect.value;
  const crop = cropSelect.value;

  if (!state || !crop || !appOptions.valid_combos[state] || !appOptions.valid_combos[state][crop]) {
    seasonSelect.innerHTML = '<option value="" disabled selected>Select crop first</option>';
    seasonSelect.disabled = true;
    return;
  }

  const validSeasons = appOptions.valid_combos[state][crop];
  seasonSelect.innerHTML = '<option value="" disabled selected>Select season</option>';
  validSeasons.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    seasonSelect.appendChild(opt);
  });
  seasonSelect.disabled = false;

  // Update trends chart for state & crop
  loadSeasonalTrends(crop, state);
}

function handleSeasonChange() {
  const stateSelect = document.getElementById('stateSelect');
  const seasonSelect = document.getElementById('seasonSelect');
  const rainfallInput = document.getElementById('rainfallInput');
  const tempInput = document.getElementById('tempInput');
  const hintRain = document.getElementById('hint-rainfall');
  const hintTemp = document.getElementById('hint-temperature');

  if (!stateSelect || !seasonSelect || !appOptions) return;

  const state = stateSelect.value;
  const season = seasonSelect.value;

  if (!state || !season) return;

  // Look up typical weather for state and season
  const typical = appOptions.typical_weather && appOptions.typical_weather[state]
    ? appOptions.typical_weather[state][season]
    : null;

  if (typical) {
    if (rainfallInput) {
      rainfallInput.value = typical.rain_mean;
      const grp = document.getElementById('group-rainfall');
      if (grp) grp.classList.remove('has-error');
    }
    if (tempInput) {
      tempInput.value = typical.temp_mean;
      const grp = document.getElementById('group-temperature');
      if (grp) grp.classList.remove('has-error');
    }

    if (hintRain) {
      hintRain.dataset.custom = 'true';
      hintRain.textContent = `Historical average for ${state}, ${season} (observed: ${typical.rain_min} – ${typical.rain_max} mm). Edit if you know this year's value.`;
    }

    if (hintTemp) {
      hintTemp.dataset.custom = 'true';
      hintTemp.textContent = `Historical average for ${state}, ${season} (observed: ${typical.temp_min} – ${typical.temp_max} °C). Edit if you know this year's value.`;
    }
  }
}

/* ==========================================================================
   5. FORM HANDLING & REAL PREDICTION API INTEGRATION
   ========================================================================== */

function initFormValidation() {
  const form = document.getElementById('yieldForm');
  const predictBtn = document.getElementById('predictBtn');
  const resultPanel = document.getElementById('predictionResult');
  const resultWarningBox = document.getElementById('resultWarningBox');
  const resultWarningText = document.getElementById('resultWarningText');
  const heroYieldValue = document.getElementById('heroYieldValue');
  const heroYieldCaption = document.getElementById('heroYieldCaption');
  const formErrorBanner = document.getElementById('formErrorBanner');

  const stateSelect = document.getElementById('stateSelect');
  const cropSelect = document.getElementById('cropSelect');
  const seasonSelect = document.getElementById('seasonSelect');

  if (!form) return;

  const fields = [
    { id: 'stateSelect', groupId: 'group-state', errorId: 'error-state' },
    { id: 'cropSelect', groupId: 'group-crop', errorId: 'error-crop' },
    { id: 'seasonSelect', groupId: 'group-season', errorId: 'error-season' },
    { id: 'areaInput', groupId: 'group-area', errorId: 'error-area' },
    { id: 'rainfallInput', groupId: 'group-rainfall', errorId: 'error-rainfall' },
    { id: 'tempInput', groupId: 'group-temperature', errorId: 'error-temperature' }
  ];

  const clearFieldError = (f) => {
    const grp = document.getElementById(f.groupId);
    if (grp) grp.classList.remove('has-error');
    if (formErrorBanner) formErrorBanner.style.display = 'none';
  };

  fields.forEach(f => {
    const el = document.getElementById(f.id);
    if (!el) return;
    el.addEventListener('input', () => clearFieldError(f));
    el.addEventListener('change', () => clearFieldError(f));
  });

  // Cascading event bindings
  if (stateSelect) {
    stateSelect.addEventListener('change', () => {
      handleStateChange();
    });
  }

  if (cropSelect) {
    cropSelect.addEventListener('change', () => {
      handleCropChange();
    });
  }

  if (seasonSelect) {
    seasonSelect.addEventListener('change', () => {
      handleSeasonChange();
    });
  }

  // Server re-check helper
  const ensureServerConnected = async () => {
    if (!appOptions || lastServerCheckFailed) {
      predictBtn.classList.add('loading');
      predictBtn.disabled = true;
      const btnTextEl = predictBtn.querySelector('.btn-text');
      const originalText = btnTextEl ? btnTextEl.textContent : 'Predict Yield';
      if (btnTextEl) btnTextEl.textContent = 'Connecting...';

      try {
        await loadInitialData();
      } finally {
        predictBtn.classList.remove('loading');
        predictBtn.disabled = false;
        if (btnTextEl) btnTextEl.textContent = originalText;
      }

      if (!appOptions || lastServerCheckFailed) {
        if (formErrorBanner) {
          formErrorBanner.textContent = 'Cannot reach the prediction server. Start the backend and try again.';
          formErrorBanner.style.display = 'block';
        }
        return false;
      }
    }
    return true;
  };

  if (predictBtn) {
    predictBtn.addEventListener('click', async (e) => {
      if (!appOptions || lastServerCheckFailed) {
        e.preventDefault();
        await ensureServerConnected();
      }
    });
  }

  // Handle Form Submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (formErrorBanner) {
      formErrorBanner.style.display = 'none';
      formErrorBanner.textContent = '';
    }

    if (!appOptions || lastServerCheckFailed) {
      const isConnected = await ensureServerConnected();
      if (!isConnected) return;
    }

    let hasErrors = false;
    const formData = {};

    // 1. Presence validation
    fields.forEach(f => {
      const el = document.getElementById(f.id);
      const grp = document.getElementById(f.groupId);
      if (!el || !grp) return;

      const val = el.value.trim();
      if (!val) {
        grp.classList.add('has-error');
        hasErrors = true;
      } else {
        grp.classList.remove('has-error');
        formData[el.name] = val;
      }
    });

    if (hasErrors) {
      const firstError = form.querySelector('.has-error .form-control');
      if (firstError) firstError.focus();
      return;
    }

    // 2. Combination check against valid_combos
    const validCombos = appOptions.valid_combos || {};
    if (!validCombos[formData.state] || !validCombos[formData.state][formData.crop]) {
      if (formErrorBanner) {
        formErrorBanner.textContent = `Crop ${formData.crop} is not cultivated in ${formData.state}.`;
        formErrorBanner.style.display = 'block';
      }
      return;
    }

    const validSeasons = validCombos[formData.state][formData.crop] || [];
    if (!validSeasons.includes(formData.season)) {
      const grp = document.getElementById('group-season');
      const errEl = document.getElementById('error-season');
      if (grp) grp.classList.add('has-error');
      if (errEl) errEl.textContent = `${formData.crop} in ${formData.state} is not grown in ${formData.season} season.`;
      if (formErrorBanner) {
        formErrorBanner.textContent = `Invalid season: ${formData.crop} in ${formData.state} is grown only in ${validSeasons.join(', ')}.`;
        formErrorBanner.style.display = 'block';
      }
      return;
    }

    // 3. Hard limits validation
    const limits = appOptions.hard_limits || {
      rainfall_min: 0,
      rainfall_max: 5000,
      temperature_min: -5,
      temperature_max: 45,
      area_min: 0.1,
      area_max: 10000
    };

    const rainVal = parseFloat(formData.rainfall);
    if (isNaN(rainVal) || rainVal < limits.rainfall_min || rainVal > limits.rainfall_max) {
      const grp = document.getElementById('group-rainfall');
      const errEl = document.getElementById('error-rainfall');
      if (grp) grp.classList.add('has-error');
      if (errEl) errEl.textContent = `Rainfall must be between ${limits.rainfall_min} and ${limits.rainfall_max} mm.`;
      hasErrors = true;
    }

    const tempVal = parseFloat(formData.temperature);
    if (isNaN(tempVal) || tempVal < limits.temperature_min || tempVal > limits.temperature_max) {
      const grp = document.getElementById('group-temperature');
      const errEl = document.getElementById('error-temperature');
      if (grp) grp.classList.add('has-error');
      if (errEl) errEl.textContent = `Temperature must be between ${limits.temperature_min} and ${limits.temperature_max} °C.`;
      hasErrors = true;
    }

    const areaVal = parseFloat(formData.area);
    if (isNaN(areaVal) || areaVal < limits.area_min || areaVal > limits.area_max) {
      const grp = document.getElementById('group-area');
      const errEl = document.getElementById('error-area');
      if (grp) grp.classList.add('has-error');
      if (errEl) errEl.textContent = `Area must be between ${limits.area_min} and ${limits.area_max} ha.`;
      hasErrors = true;
    }

    if (hasErrors) {
      const firstError = form.querySelector('.has-error .form-control');
      if (firstError) firstError.focus();
      return;
    }

    // 4. Loading State
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    const originalBtnText = predictBtn.querySelector('.btn-text').textContent;
    predictBtn.querySelector('.btn-text').textContent = 'Predicting...';

    try {
      const payload = {
        state: formData.state,
        crop: formData.crop,
        season: formData.season,
        rainfall: rainVal,
        temperature: tempVal,
        area: areaVal
      };

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const responseData = await response.json().catch(() => ({}));

      if (!response.ok) {
        const serverErrorMessage = responseData.error || `Server prediction error (${response.status})`;
        if (formErrorBanner) {
          formErrorBanner.textContent = serverErrorMessage;
          formErrorBanner.style.display = 'block';
        }
        if (resultPanel) {
          resultPanel.classList.remove('visible');
        }
        return;
      }

      if (formErrorBanner) {
        formErrorBanner.style.display = 'none';
      }

      const predictedYield = responseData.predicted_yield;
      const totalProduction = responseData.total_production;
      const modelNote = responseData.note || 'Estimate produced by state-aware agronomic HistGradientBoosting pipeline.';

      document.getElementById('resultCropTag').textContent = `${payload.crop} • ${payload.season} (${payload.state})`;
      document.getElementById('resultYieldValue').innerHTML = `${predictedYield.toFixed(2)} <span class="metric-unit">t/ha</span>`;
      document.getElementById('resultProductionValue').innerHTML = `${totalProduction.toFixed(2)} <span class="metric-unit">tonnes</span>`;
      document.getElementById('resultAreaNote').textContent = `Across ${payload.area} hectare${payload.area > 1 ? 's' : ''}`;

      const analysisText = document.getElementById('resultAnalysisText');
      if (analysisText) {
        analysisText.textContent = modelNote;
      }

      // Display warnings as soft amber note if present
      if (resultWarningBox && resultWarningText) {
        if (Array.isArray(responseData.warnings) && responseData.warnings.length > 0) {
          resultWarningText.innerHTML = responseData.warnings.join('<br>');
          resultWarningBox.style.display = 'block';
        } else {
          resultWarningBox.style.display = 'none';
          resultWarningText.innerHTML = '';
        }
      }

      resultPanel.classList.add('visible');

      if (heroYieldValue) {
        heroYieldValue.textContent = `${predictedYield.toFixed(2)} t/ha`;
      }
      if (heroYieldCaption) {
        heroYieldCaption.textContent = `${payload.crop} (${payload.state})`;
      }

      loadSeasonalTrends(payload.crop, payload.state);

      if (window.innerWidth < 768) {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

    } catch (err) {
      console.error('Prediction network request failed:', err);
      lastServerCheckFailed = true;
      if (formErrorBanner) {
        formErrorBanner.textContent = 'Cannot reach the prediction server. Start the backend and try again.';
        formErrorBanner.style.display = 'block';
      }
      if (resultPanel) {
        resultPanel.classList.remove('visible');
      }
    } finally {
      predictBtn.classList.remove('loading');
      predictBtn.disabled = false;
      predictBtn.querySelector('.btn-text').textContent = originalBtnText;
    }
  });
}

/* ==========================================================================
   6. NAVIGATION & MOBILE DRAWER
   ========================================================================== */

function initNavigation() {
  const hamburgerBtn = document.getElementById('hamburgerBtn');
  const mobileMenu = document.getElementById('mobileMenu');
  const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');

  if (hamburgerBtn && mobileMenu) {
    hamburgerBtn.addEventListener('click', () => {
      const isOpen = mobileMenu.classList.toggle('open');
      hamburgerBtn.classList.toggle('open', isOpen);
      hamburgerBtn.setAttribute('aria-expanded', isOpen);
    });
  }

  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const targetId = link.getAttribute('href');
      if (targetId && targetId.startsWith('#')) {
        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: 'smooth' });
          if (mobileMenu && mobileMenu.classList.contains('open')) {
            mobileMenu.classList.remove('open');
            if (hamburgerBtn) {
              hamburgerBtn.classList.remove('open');
              hamburgerBtn.setAttribute('aria-expanded', 'false');
            }
          }
        }
      }
    });
  });
}
