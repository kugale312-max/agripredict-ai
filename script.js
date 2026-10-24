/**
 * AgriPredict AI — Frontend Application Logic
 * Integrates directly with Flask backend REST API
 */

// 1. API Base URL configuration
// Use '' (same origin) by default, except when opened from a file or port 8000 where it stays http://localhost:5000
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
 * Loads dynamic options, input ranges, and model metrics from the Flask backend.
 * Auto-reconnects every 4 seconds in the background if the server is unreachable.
 */
async function loadInitialData() {
  if (isFetchingInitialData) return;
  isFetchingInitialData = true;

  const serverErrorBanner = document.getElementById('serverErrorBanner');
  const cropSelect = document.getElementById('cropSelect');
  const soilSelect = document.getElementById('soilSelect');
  const seasonSelect = document.getElementById('seasonSelect');

  try {
    // 1. Fetch options (crops, seasons, soils, valid combinations, input ranges)
    const optionsResp = await fetch(`${API_BASE_URL}/options`);
    if (!optionsResp.ok) {
      throw new Error(`Server returned HTTP ${optionsResp.status}`);
    }
    appOptions = await optionsResp.json();
    lastServerCheckFailed = false;

    // Stop background reconnect timer after success
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    // Hide error banner automatically
    if (serverErrorBanner) {
      serverErrorBanner.style.display = 'none';
    }

    // Populate Crop dropdown
    if (cropSelect && Array.isArray(appOptions.crops)) {
      cropSelect.innerHTML = '<option value="" disabled selected>Select crop</option>';
      appOptions.crops.forEach(crop => {
        const opt = document.createElement('option');
        opt.value = crop;
        opt.textContent = crop;
        cropSelect.appendChild(opt);
      });
    }

    // Populate Soil dropdown
    if (soilSelect && Array.isArray(appOptions.soil_types)) {
      soilSelect.innerHTML = '<option value="" disabled selected>Select soil</option>';
      appOptions.soil_types.forEach(soil => {
        const opt = document.createElement('option');
        opt.value = soil;
        opt.textContent = soil;
        soilSelect.appendChild(opt);
      });
    }

    // Initialize Season dropdown (all disabled until a crop is selected)
    if (seasonSelect && Array.isArray(appOptions.seasons)) {
      seasonSelect.innerHTML = '<option value="" disabled selected>Select season</option>';
      appOptions.seasons.forEach(season => {
        const opt = document.createElement('option');
        opt.value = season;
        opt.textContent = season;
        opt.disabled = true;
        seasonSelect.appendChild(opt);
      });
    }

    // Setup input ranges, placeholders, and allowed range helper hints
    if (appOptions.input_ranges) {
      setupInputRanges(appOptions.input_ranges);
    }

    // Load seasonal trends chart for the first available crop
    if (appOptions.crops && appOptions.crops.length > 0) {
      const firstCrop = appOptions.crops[0];
      await loadSeasonalTrends(firstCrop);
    }

    // Load model evaluation metrics for footer
    loadModelMetrics();

  } catch (err) {
    console.error('Failed to load options from backend:', err);
    lastServerCheckFailed = true;

    // Keep the red banner and add small "Retry now" link
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

    // Replace "Server unreachable" in the dropdowns with "Connecting to server..."
    if (cropSelect) {
      cropSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }
    if (soilSelect) {
      soilSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }
    if (seasonSelect) {
      seasonSelect.innerHTML = '<option value="" disabled selected>Connecting to server...</option>';
    }

    // Auto-reconnect: retry every 4 seconds in the background
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
 * Configure min/max/placeholder and helper hints for numerical inputs.
 */
function setupInputRanges(ranges) {
  // Rainfall
  const rainInput = document.getElementById('rainfallInput');
  const rainHint = document.getElementById('hint-rainfall');
  if (rainInput && ranges.rainfall) {
    rainInput.min = ranges.rainfall.min;
    rainInput.max = ranges.rainfall.max;
    rainInput.placeholder = 'e.g. 700';
    if (rainHint) {
      rainHint.textContent = `Allowed range: ${ranges.rainfall.min} – ${ranges.rainfall.max} mm`;
    }
  }

  // Temperature
  const tempInput = document.getElementById('tempInput');
  const tempHint = document.getElementById('hint-temperature');
  if (tempInput && ranges.temperature) {
    tempInput.min = ranges.temperature.min;
    tempInput.max = ranges.temperature.max;
    tempInput.placeholder = 'e.g. 26';
    if (tempHint) {
      tempHint.textContent = `Allowed range: ${ranges.temperature.min} – ${ranges.temperature.max} °C`;
    }
  }

  // Area
  const areaInput = document.getElementById('areaInput');
  const areaHint = document.getElementById('hint-area');
  if (areaInput && ranges.area) {
    areaInput.min = ranges.area.min;
    areaInput.max = ranges.area.max;
    areaInput.placeholder = 'e.g. 2.5';
    if (areaHint) {
      areaHint.textContent = `Allowed range: ${ranges.area.min} – ${ranges.area.max} ha`;
    }
  }
}

/**
 * Fetches /metrics and appends summary line to footer.
 */
async function loadModelMetrics() {
  const footerMetrics = document.getElementById('footerMetrics');
  if (!footerMetrics) return;

  try {
    const resp = await fetch(`${API_BASE_URL}/metrics`);
    if (!resp.ok) return;
    const metrics = await resp.json();

    const modelName = metrics.model_name || 'HistGradientBoosting';
    const r2Val = typeof metrics.time_based_r2 === 'number' ? metrics.time_based_r2.toFixed(4) : metrics.time_based_r2;
    const maeVal = typeof metrics.time_based_mae === 'number' ? metrics.time_based_mae.toFixed(4) : metrics.time_based_mae;

    footerMetrics.textContent = `Model: ${modelName} | R² ${r2Val} | MAE ${maeVal} t/ha (tested on 2013-2015)`;
  } catch (err) {
    console.warn('Metrics endpoint unavailable:', err);
  }
}

/* ==========================================================================
   3. SEASONAL CHART & TRENDS INTEGRATION
   ========================================================================== */

/**
 * Initializes the Chart.js canvas instance.
 */
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
 * Fetches real historical trends for a crop and updates Chart.js, badge, and Best Outlook.
 */
async function loadSeasonalTrends(cropName) {
  if (!cropName) return;

  try {
    const resp = await fetch(`${API_BASE_URL}/trends?crop=${encodeURIComponent(cropName)}`);
    if (!resp.ok) return;
    const data = await resp.json();

    const trends = data.trends || [];
    const validSeasons = data.valid_seasons || [];

    // 1. Update data badge to 'Based on historical data'
    const badgeText = document.getElementById('chartDataBadgeText');
    if (badgeText) {
      badgeText.textContent = 'Based on historical data';
    }

    // 2. Identify season with highest average yield
    let bestSeason = validSeasons[0] || 'Kharif';
    let maxYield = -Infinity;
    trends.forEach(item => {
      if (item.average_yield > maxYield) {
        maxYield = item.average_yield;
        bestSeason = item.season;
      }
    });

    // 3. Update Best Outlook box
    const highlightTitle = document.getElementById('highlightTitle');
    const highlightDesc = document.getElementById('highlightDesc');
    if (highlightTitle) {
      highlightTitle.textContent = `Best outlook: ${bestSeason} season`;
    }
    if (highlightDesc) {
      highlightDesc.textContent = `Highest average historical yield for ${cropName}.`;
    }

    // 4. Update Chart.js dataset
    if (seasonalChartInstance) {
      const labels = trends.map(t => t.season);
      const yields = trends.map(t => t.average_yield);

      seasonalChartInstance.data.labels = labels;
      seasonalChartInstance.data.datasets[0].data = yields;

      const chartLegendText = document.getElementById('chartLegendText');

      if (trends.length <= 1) {
        // Single valid season: show single point and note instead of line
        seasonalChartInstance.data.datasets[0].showLine = false;
        seasonalChartInstance.data.datasets[0].pointRadius = 8;
        seasonalChartInstance.data.datasets[0].pointHoverRadius = 10;
        seasonalChartInstance.data.datasets[0].pointBackgroundColor = ['#D9A441'];
        seasonalChartInstance.data.datasets[0].pointHoverBackgroundColor = ['#D9A441'];

        if (chartLegendText) {
          const singleSeason = trends[0] ? trends[0].season : 'Single Season';
          chartLegendText.textContent = `Single season crop (${singleSeason}) · historical average yield`;
        }
      } else {
        // Multiple seasons: show connected line with gold highlight on best season
        seasonalChartInstance.data.datasets[0].showLine = true;
        seasonalChartInstance.data.datasets[0].borderWidth = 3;
        seasonalChartInstance.data.datasets[0].pointRadius = 6;
        seasonalChartInstance.data.datasets[0].pointHoverRadius = 8;

        const bestIdx = labels.indexOf(bestSeason);
        const pointColors = labels.map((_, i) => (i === bestIdx ? '#D9A441' : '#4A8B52'));
        seasonalChartInstance.data.datasets[0].pointBackgroundColor = pointColors;
        seasonalChartInstance.data.datasets[0].pointHoverBackgroundColor = pointColors;

        if (chartLegendText) {
          chartLegendText.textContent = 'Historical yield · tonnes per hectare';
        }
      }

      seasonalChartInstance.update();
    }
  } catch (err) {
    console.error(`Failed to load trends for crop ${cropName}:`, err);
  }
}

/* ==========================================================================
   4. FORM HANDLING & REAL PREDICTION API INTEGRATION
   ========================================================================== */

function initFormValidation() {
  const form = document.getElementById('yieldForm');
  const predictBtn = document.getElementById('predictBtn');
  const resultPanel = document.getElementById('predictionResult');
  const heroYieldValue = document.getElementById('heroYieldValue');
  const heroYieldCaption = document.getElementById('heroYieldCaption');
  const formErrorBanner = document.getElementById('formErrorBanner');
  const cropSelect = document.getElementById('cropSelect');
  const seasonSelect = document.getElementById('seasonSelect');

  if (!form) return;

  const fields = [
    { id: 'cropSelect', groupId: 'group-crop', errorId: 'error-crop' },
    { id: 'seasonSelect', groupId: 'group-season', errorId: 'error-season' },
    { id: 'rainfallInput', groupId: 'group-rainfall', errorId: 'error-rainfall' },
    { id: 'tempInput', groupId: 'group-temperature', errorId: 'error-temperature' },
    { id: 'soilSelect', groupId: 'group-soil', errorId: 'error-soil' },
    { id: 'areaInput', groupId: 'group-area', errorId: 'error-area' }
  ];

  // Helper to clear error state
  const clearFieldError = (f) => {
    const grp = document.getElementById(f.groupId);
    if (grp) grp.classList.remove('has-error');
    if (formErrorBanner) formErrorBanner.style.display = 'none';
  };

  // Attach clear error listeners
  fields.forEach(f => {
    const el = document.getElementById(f.id);
    if (!el) return;

    el.addEventListener('input', () => clearFieldError(f));
    el.addEventListener('change', () => clearFieldError(f));
  });

  // Handle crop change: filter seasons and update seasonal chart
  if (cropSelect) {
    cropSelect.addEventListener('change', () => {
      const selectedCrop = cropSelect.value;
      if (!selectedCrop) return;

      // Update seasonal chart
      loadSeasonalTrends(selectedCrop);

      // Filter seasons based on valid_crop_seasons from options
      if (seasonSelect && appOptions && appOptions.valid_crop_seasons) {
        const validSeasons = appOptions.valid_crop_seasons[selectedCrop] || [];

        Array.from(seasonSelect.options).forEach(opt => {
          if (!opt.value) return; // Keep "Select season" placeholder
          const isValid = validSeasons.includes(opt.value);
          opt.disabled = !isValid;
        });

        // Reset season if the currently selected season is not valid for this crop
        if (seasonSelect.value && !validSeasons.includes(seasonSelect.value)) {
          seasonSelect.value = '';
          const grp = document.getElementById('group-season');
          if (grp) grp.classList.remove('has-error');
        }
      }
    });
  }

  // Re-check server connection helper for Predict button
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

    // Re-check server if the last check failed, instead of failing silently
    if (!appOptions || lastServerCheckFailed) {
      const isConnected = await ensureServerConnected();
      if (!isConnected) {
        return;
      }
    }

    let hasErrors = false;
    const formData = {};

    // 1. Client-side field presence validation
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

    // 2. Client-side crop & season pairing validation
    if (appOptions && appOptions.valid_crop_seasons) {
      const validSeasons = appOptions.valid_crop_seasons[formData.crop] || [];
      if (!validSeasons.includes(formData.season)) {
        const grp = document.getElementById('group-season');
        const errEl = document.getElementById('error-season');
        if (grp) grp.classList.add('has-error');
        if (errEl) errEl.textContent = `${formData.crop} is not cultivated in ${formData.season} season.`;
        if (formErrorBanner) {
          formErrorBanner.textContent = `Invalid crop and season combination: ${formData.crop} is only grown in ${validSeasons.join(', ')}.`;
          formErrorBanner.style.display = 'block';
        }
        return;
      }
    }

    // 3. Client-side numerical range validation
    if (appOptions && appOptions.input_ranges) {
      const ranges = appOptions.input_ranges;

      // Rainfall check
      const rainVal = parseFloat(formData.rainfall);
      if (ranges.rainfall && (rainVal < ranges.rainfall.min || rainVal > ranges.rainfall.max)) {
        const grp = document.getElementById('group-rainfall');
        const errEl = document.getElementById('error-rainfall');
        if (grp) grp.classList.add('has-error');
        if (errEl) errEl.textContent = `Rainfall must be between ${ranges.rainfall.min} and ${ranges.rainfall.max} mm.`;
        hasErrors = true;
      }

      // Temperature check
      const tempVal = parseFloat(formData.temperature);
      if (ranges.temperature && (tempVal < ranges.temperature.min || tempVal > ranges.temperature.max)) {
        const grp = document.getElementById('group-temperature');
        const errEl = document.getElementById('error-temperature');
        if (grp) grp.classList.add('has-error');
        if (errEl) errEl.textContent = `Temperature must be between ${ranges.temperature.min} and ${ranges.temperature.max} °C.`;
        hasErrors = true;
      }

      // Area check
      const areaVal = parseFloat(formData.area);
      if (ranges.area && (areaVal < ranges.area.min || areaVal > ranges.area.max)) {
        const grp = document.getElementById('group-area');
        const errEl = document.getElementById('error-area');
        if (grp) grp.classList.add('has-error');
        if (errEl) errEl.textContent = `Area must be between ${ranges.area.min} and ${ranges.area.max} ha.`;
        hasErrors = true;
      }

      if (hasErrors) {
        const firstError = form.querySelector('.has-error .form-control');
        if (firstError) firstError.focus();
        return;
      }
    }

    // 4. Enter Loading State
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    const originalBtnText = predictBtn.querySelector('.btn-text').textContent;
    predictBtn.querySelector('.btn-text').textContent = 'Predicting...';

    try {
      const payload = {
        crop: formData.crop,
        season: formData.season,
        rainfall: parseFloat(formData.rainfall),
        temperature: parseFloat(formData.temperature),
        soil: formData.soil,
        area: parseFloat(formData.area)
      };

      // Real fetch() POST to Flask /predict endpoint
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const responseData = await response.json().catch(() => ({}));

      // Handle non-200 responses (e.g. 400 Bad Request)
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

      // 5. Successful Prediction: update result panel with gentle fade-in
      if (formErrorBanner) {
        formErrorBanner.style.display = 'none';
      }

      const predictedYield = responseData.predicted_yield;
      const totalProduction = responseData.total_production;
      const modelNote = responseData.note || 'This is an estimate based on historical district-level data.';

      document.getElementById('resultCropTag').textContent = `${payload.crop} • ${payload.season}`;
      document.getElementById('resultYieldValue').innerHTML = `${predictedYield.toFixed(2)} <span class="metric-unit">t/ha</span>`;
      document.getElementById('resultProductionValue').innerHTML = `${totalProduction.toFixed(2)} <span class="metric-unit">tonnes</span>`;
      document.getElementById('resultAreaNote').textContent = `Across ${payload.area} hectare${payload.area > 1 ? 's' : ''}`;

      const analysisText = document.getElementById('resultAnalysisText');
      if (analysisText) {
        analysisText.textContent = modelNote;
      }

      resultPanel.classList.add('visible');

      // 6. Update Hero Card with latest prediction
      if (heroYieldValue) {
        heroYieldValue.textContent = `${predictedYield.toFixed(2)} t/ha`;
      }
      if (heroYieldCaption) {
        heroYieldCaption.textContent = 'Latest prediction';
      }

      // Also refresh the seasonal trends chart to match the predicted crop
      loadSeasonalTrends(payload.crop);

      // Smooth scroll if result is below visible area on smaller screens
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
   5. NAVIGATION & MOBILE DRAWER
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
