const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const USER_DATA = path.join(process.env.TEMP || 'C:\\Temp', 'chrome-profile-e2e-' + Date.now());

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch (_) {}
    await sleep(400);
  }
  return false;
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();
    this.events = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const cb = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) cb.reject(new Error(msg.error.message));
          else cb.resolve(msg.result);
        }
        if (msg.method && this.events.has(msg.method)) {
          const listeners = this.events.get(msg.method);
          listeners.forEach(fn => fn(msg.params));
        }
      };
    });
  }

  on(event, handler) {
    if (!this.events.has(event)) this.events.set(event, []);
    this.events.get(event).push(handler);
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = this.id++;
      this.callbacks.set(msgId, { resolve, reject });
      this.ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    return res.result.value;
  }

  async navigate(url) {
    const loadPromise = new Promise(resolve => {
      const timer = setTimeout(resolve, 8000);
      this.on('Page.loadEventFired', () => {
        clearTimeout(timer);
        resolve();
      });
    });
    await this.send('Page.navigate', { url });
    await loadPromise;
    await sleep(2000);
  }

  async reload() {
    const loadPromise = new Promise(resolve => {
      const timer = setTimeout(resolve, 8000);
      this.on('Page.loadEventFired', () => {
        clearTimeout(timer);
        resolve();
      });
    });
    await this.send('Page.reload');
    await loadPromise;
    await sleep(2000);
  }
}

async function main() {
  console.log('================================================================');
  console.log('AGRIPREDICT AI — E2E TEST: FLASK SERVING & AUTO-RECONNECT');
  console.log('================================================================\n');

  // STEP 1: Launch Backend Server
  console.log('[STEP 1] Starting Flask backend (python backend/app.py)...');
  let flaskProcess = spawn('python', ['backend/app.py'], {
    cwd: path.resolve(__dirname, '..'),
    stdio: 'pipe'
  });

  const serverReady = await waitForServer('http://127.0.0.1:5000/health');
  if (!serverReady) {
    console.error('Flask server failed to start within timeout.');
    flaskProcess.kill();
    process.exit(1);
  }
  console.log('[OK] Flask backend is running on http://127.0.0.1:5000\n');

  // STEP 2: Launch Headless Chrome
  console.log('[STEP 2] Launching Chrome with remote debugging on port 9222...');
  const chromeProcess = spawn(CHROME_PATH, [
    '--headless=new',
    '--remote-debugging-port=9222',
    `--user-data-dir=${USER_DATA}`,
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ]);

  await sleep(1500);
  const jsonResp = await fetch('http://127.0.0.1:9222/json');
  const tabs = await jsonResp.json();
  const cdp = new CDPClient(tabs[0].webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  console.log('[OK] Connected to Chrome via Chrome DevTools Protocol\n');

  try {
    // -------------------------------------------------------------
    // PHASE 1: Verify Initial Load with Backend Running
    // -------------------------------------------------------------
    console.log('--- PHASE 1: Initial Page Load (Backend Running) ---');
    console.log('Navigating to http://127.0.0.1:5000...');
    await cdp.navigate('http://127.0.0.1:5000');

    const phase1 = await cdp.eval(`
      ({
        title: document.title,
        bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
        cropOptionsCount: document.getElementById('cropSelect')?.options.length,
        firstCrop: document.getElementById('cropSelect')?.options[0]?.text,
        sampleCrops: Array.from(document.getElementById('cropSelect')?.options || []).slice(1, 5).map(o => o.text),
        soilOptionsCount: document.getElementById('soilSelect')?.options.length,
        soils: Array.from(document.getElementById('soilSelect')?.options || []).slice(1).map(o => o.text),
        footerMetrics: document.getElementById('footerMetrics')?.textContent,
        predictBtnText: document.getElementById('predictBtn')?.querySelector('.btn-text')?.textContent
      })
    `);

    console.log('Phase 1 Evaluation:');
    console.log('  Page Title          :', phase1.title);
    console.log('  Banner Display      :', phase1.bannerDisplay, phase1.bannerDisplay === 'none' ? '(HIDDEN - OK)' : '(VISIBLE)');
    console.log('  Crop Options Count  :', phase1.cropOptionsCount, `(${phase1.firstCrop}, samples: ${phase1.sampleCrops.join(', ')})`);
    console.log('  Soil Options Count  :', phase1.soilOptionsCount, `(soils: ${phase1.soils.join(', ')})`);
    console.log('  Footer Metrics      :', phase1.footerMetrics);
    console.log('  Predict Button Text :', phase1.predictBtnText);

    const shot1 = await cdp.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.resolve(__dirname, 'phase1_running.png'), Buffer.from(shot1.data, 'base64'));

    if (phase1.cropOptionsCount > 20 && phase1.soilOptionsCount >= 5 && phase1.bannerDisplay === 'none') {
      console.log('>>> [PASS] Phase 1: Full page loaded with populated dropdowns from Flask.\n');
    } else {
      throw new Error('Phase 1 validation failed');
    }

    // -------------------------------------------------------------
    // PHASE 2: Stop Server & Refresh
    // -------------------------------------------------------------
    console.log('--- PHASE 2: Stop Server & Refresh Page ---');
    console.log('Stopping Flask backend process...');
    flaskProcess.kill('SIGKILL');
    await sleep(1500);

    console.log('Refreshing page in browser while server is stopped...');
    await cdp.reload();

    const phase2 = await cdp.eval(`
      ({
        bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
        bannerText: document.getElementById('serverErrorBanner')?.innerText?.trim(),
        hasRetryNowLink: !!document.getElementById('retryNowLink'),
        cropSelectedText: document.getElementById('cropSelect')?.options[0]?.text,
        soilSelectedText: document.getElementById('soilSelect')?.options[0]?.text,
        predictBtnText: document.getElementById('predictBtn')?.querySelector('.btn-text')?.textContent
      })
    `);

    console.log('Phase 2 Evaluation:');
    console.log('  Banner Display      :', phase2.bannerDisplay, phase2.bannerDisplay === 'block' ? '(VISIBLE - OK)' : '(HIDDEN)');
    console.log('  Banner Text         :', phase2.bannerText);
    console.log('  Has "Retry now" Link:', phase2.hasRetryNowLink);
    console.log('  Crop Select Text    :', phase2.cropSelectedText);
    console.log('  Soil Select Text    :', phase2.soilSelectedText);

    const shot2 = await cdp.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.resolve(__dirname, 'phase2_disconnected.png'), Buffer.from(shot2.data, 'base64'));

    if (
      phase2.bannerDisplay === 'block' &&
      phase2.hasRetryNowLink &&
      phase2.cropSelectedText === 'Connecting to server...' &&
      phase2.soilSelectedText === 'Connecting to server...'
    ) {
      console.log('>>> [PASS] Phase 2: Page displays "Connecting to server..." and red banner with "Retry now" link.\n');
    } else {
      throw new Error('Phase 2 validation failed');
    }

    // -------------------------------------------------------------
    // PHASE 3: Restart Server & Auto-Reconnect (No Page Refresh!)
    // -------------------------------------------------------------
    console.log('--- PHASE 3: Restart Server & Test Auto-Reconnect (NO Refresh) ---');
    console.log('Restarting Flask backend process...');
    flaskProcess = spawn('python', ['backend/app.py'], {
      cwd: path.resolve(__dirname, '..'),
      stdio: 'pipe'
    });

    const restarted = await waitForServer('http://127.0.0.1:5000/health');
    console.log('[OK] Flask backend is back online.');
    console.log('Waiting 5 seconds for background reconnect timer (4s interval) WITHOUT refreshing page...');
    await sleep(5500);

    const phase3 = await cdp.eval(`
      ({
        bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
        cropOptionsCount: document.getElementById('cropSelect')?.options.length,
        firstCrop: document.getElementById('cropSelect')?.options[0]?.text,
        sampleCrops: Array.from(document.getElementById('cropSelect')?.options || []).slice(1, 5).map(o => o.text),
        soilOptionsCount: document.getElementById('soilSelect')?.options.length,
        soils: Array.from(document.getElementById('soilSelect')?.options || []).slice(1).map(o => o.text),
        footerMetrics: document.getElementById('footerMetrics')?.textContent
      })
    `);

    console.log('Phase 3 Evaluation:');
    console.log('  Banner Display      :', phase3.bannerDisplay, phase3.bannerDisplay === 'none' ? '(AUTOMATICALLY HIDDEN - OK)' : '(STILL VISIBLE)');
    console.log('  Crop Options Count  :', phase3.cropOptionsCount, `(${phase3.firstCrop}, samples: ${phase3.sampleCrops.join(', ')})`);
    console.log('  Soil Options Count  :', phase3.soilOptionsCount, `(soils: ${phase3.soils.join(', ')})`);
    console.log('  Footer Metrics      :', phase3.footerMetrics);

    const shot3 = await cdp.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(path.resolve(__dirname, 'phase3_reconnected.png'), Buffer.from(shot3.data, 'base64'));

    if (
      phase3.bannerDisplay === 'none' &&
      phase3.cropOptionsCount > 20 &&
      phase3.soilOptionsCount >= 5 &&
      phase3.footerMetrics
    ) {
      console.log('>>> [PASS] Phase 3: Page reconnected automatically without refresh!');
      console.log('>>> Dropdowns filled, chart & metrics loaded, and error banner hidden.\n');
    } else {
      throw new Error('Phase 3 validation failed');
    }

    console.log('================================================================');
    console.log('ALL LIFECYCLE TESTS COMPLETED SUCCESSFULLY!');
    console.log('================================================================');

  } finally {
    try { flaskProcess.kill('SIGKILL'); } catch (_) {}
    try { chromeProcess.kill('SIGKILL'); } catch (_) {}
  }
}

main().catch(err => {
  console.error('\n[FATAL ERROR]', err.message);
  process.exit(1);
});
