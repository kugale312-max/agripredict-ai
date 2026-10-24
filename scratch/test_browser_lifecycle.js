const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const USER_DATA = path.join(process.env.TEMP || 'C:\\Temp', 'chrome-test-profile-' + Date.now());

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();
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
      };
    });
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
}

async function runTest() {
  console.log('[STEP 1] Launching headless Chrome with remote debugging on port 9222...');
  const chrome = spawn(CHROME_PATH, [
    '--headless=new',
    '--remote-debugging-port=9222',
    `--user-data-dir=${USER_DATA}`,
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ], { detached: false });

  await sleep(1500);

  try {
    const resp = await fetch('http://127.0.0.1:9222/json');
    const tabs = await resp.json();
    const tab = tabs[0];
    console.log('[OK] Connected to Chrome page:', tab.id);

    const client = new CDPClient(tab.webSocketDebuggerUrl);
    await client.connect();
    await client.send('Page.enable');
    await client.send('Runtime.enable');

    // 1. Navigate to http://127.0.0.1:5000
    console.log('\n--- TEST A: Initial Load with Backend Running ---');
    console.log('Navigating to http://127.0.0.1:5000...');
    await client.send('Page.navigate', { url: 'http://127.0.0.1:5000' });
    await sleep(2500);

    const initialInfo = await client.eval(`
      ({
        title: document.title,
        cropCount: document.getElementById('cropSelect')?.options.length,
        firstCrop: document.getElementById('cropSelect')?.options[0]?.text,
        secondCrop: document.getElementById('cropSelect')?.options[1]?.text,
        soilCount: document.getElementById('soilSelect')?.options.length,
        firstSoil: document.getElementById('soilSelect')?.options[0]?.text,
        secondSoil: document.getElementById('soilSelect')?.options[1]?.text,
        bannerDisplay: document.getElementById('serverErrorBanner')?.style.display,
        footerText: document.getElementById('footerMetrics')?.textContent
      })
    `);

    console.log('Initial page evaluation:', initialInfo);
    if (initialInfo.cropCount > 10 && initialInfo.bannerDisplay === 'none') {
      console.log('>>> [PASS TEST A] Dropdowns populated with crops and soils. Banner is hidden.');
    } else {
      console.log('>>> [FAIL TEST A] Unexpected state on initial load.');
    }

    // Save screenshot
    const shotA = await client.send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('scratch/screenshot_a_connected.png', Buffer.from(shotA.data, 'base64'));
    console.log('[Artifact] Saved scratch/screenshot_a_connected.png');

    // Return client and chrome for subsequent lifecycle testing
    return { client, chrome };
  } catch (err) {
    console.error('Chrome test error:', err);
    chrome.kill();
    throw err;
  }
}

runTest().then(({ client, chrome }) => {
  // Keep process alive for further lifecycle steps or exit cleanly
  setTimeout(() => {
    chrome.kill();
    process.exit(0);
  }, 2000);
}).catch(err => {
  console.error(err);
  process.exit(1);
});
