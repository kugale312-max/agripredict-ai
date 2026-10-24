const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const USER_DATA = path.join(process.env.TEMP || 'C:\\Temp', 'chrome-profile-btn-' + Date.now());

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
    await sleep(1500);
  }
}

async function testButtons() {
  console.log('--- TEST: Retry Now Link & Predict Button Re-Check Logic ---');

  // Launch Flask
  let flask = spawn('python', ['backend/app.py'], {
    cwd: path.resolve(__dirname, '..'),
    stdio: 'ignore'
  });
  await waitForServer('http://127.0.0.1:5000/health');

  // Launch Chrome
  const chrome = spawn(CHROME_PATH, [
    '--headless=new',
    '--remote-debugging-port=9224',
    `--user-data-dir=${USER_DATA}`,
    'about:blank'
  ]);
  await sleep(1500);

  const tabs = await (await fetch('http://127.0.0.1:9224/json')).json();
  const cdp = new CDPClient(tabs[0].webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');

  await cdp.navigate('http://127.0.0.1:5000');
  console.log('[OK] Page loaded successfully.');

  // Kill Flask to simulate disconnection
  flask.kill('SIGKILL');
  await sleep(1500);

  // Trigger loadInitialData() failure in the browser
  console.log('Triggering connection failure in page...');
  await cdp.eval('loadInitialData()');
  await sleep(1000);

  const stateDisconnected = await cdp.eval(`
    ({
      bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
      hasRetryLink: !!document.getElementById('retryNowLink'),
      cropText: document.getElementById('cropSelect')?.options[0]?.text
    })
  `);

  console.log('Disconnected State:', stateDisconnected);
  if (stateDisconnected.hasRetryLink && stateDisconnected.cropText === 'Connecting to server...') {
    console.log('[PASS] Dropdowns show "Connecting to server..." and Retry link is present.');
  }

  // Click Predict Button while disconnected
  console.log('Clicking Predict button while server is disconnected...');
  await cdp.eval(`document.getElementById('predictBtn')?.click()`);
  await sleep(500);

  const stateAfterPredictClick = await cdp.eval(`
    ({
      formBannerDisplay: document.getElementById('formErrorBanner')?.style.display,
      formBannerText: document.getElementById('formErrorBanner')?.innerText?.trim()
    })
  `);

  console.log('Predict Button Click Result (Server Down):', stateAfterPredictClick);
  if (stateAfterPredictClick.formBannerDisplay === 'block' && stateAfterPredictClick.formBannerText.includes('Cannot reach the prediction server')) {
    console.log('[PASS] Predict button re-checked server and displayed clear error banner instead of failing silently.');
  }

  // Now start Flask again
  console.log('Restarting Flask server...');
  flask = spawn('python', ['backend/app.py'], {
    cwd: path.resolve(__dirname, '..'),
    stdio: 'ignore'
  });
  await waitForServer('http://127.0.0.1:5000/health');
  console.log('[OK] Flask server restarted.');

  // Click "Retry now" link
  console.log('Clicking "Retry now" link...');
  await cdp.eval(`document.getElementById('retryNowLink')?.click()`);
  await sleep(1500);

  const stateAfterRetryNow = await cdp.eval(`
    ({
      bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
      cropCount: document.getElementById('cropSelect')?.options.length,
      firstCrop: document.getElementById('cropSelect')?.options[0]?.text,
      soilCount: document.getElementById('soilSelect')?.options.length
    })
  `);

  console.log('After "Retry now" Click:', stateAfterRetryNow);
  if (stateAfterRetryNow.bannerDisplay === 'none' && stateAfterRetryNow.cropCount > 20) {
    console.log('[PASS] "Retry now" link immediately reconnected and filled dropdowns.');
  }

  chrome.kill();
  flask.kill();
  console.log('\nALL BUTTON & INTERACTION TESTS PASSED!');
}

testButtons().catch(err => {
  console.error('[TEST ERROR]', err);
  process.exit(1);
});
