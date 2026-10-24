const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const ARTIFACT_DIR = 'C:\\Users\\91942\\.gemini\\antigravity-ide\\brain\\4418a986-dbdc-40c0-828d-436526bffa03';
const CHROME_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
const USER_DATA = path.join(process.env.TEMP || 'C:\\Temp', 'chrome-profile-shot-' + Date.now());

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
}

async function capture() {
  console.log('Starting Flask backend...');
  const flask = spawn('python', ['backend/app.py'], {
    cwd: path.resolve(__dirname, '..'),
    stdio: 'ignore'
  });

  // Wait for server
  for (let i = 0; i < 20; i++) {
    try {
      const res = await fetch('http://127.0.0.1:5000/health');
      if (res.ok) break;
    } catch (_) {}
    await sleep(300);
  }

  console.log('Starting Chrome...');
  const chrome = spawn(CHROME_PATH, [
    '--headless=new',
    '--remote-debugging-port=9223',
    `--user-data-dir=${USER_DATA}`,
    'about:blank'
  ]);
  await sleep(1500);

  const tabs = await (await fetch('http://127.0.0.1:9223/json')).json();
  const cdp = new CDPClient(tabs[0].webSocketDebuggerUrl);
  await cdp.connect();
  await cdp.send('Page.enable');
  await cdp.send('Page.navigate', { url: 'http://127.0.0.1:5000' });
  await sleep(2500);

  console.log('Capturing screenshot...');
  const shot = await cdp.send('Page.captureScreenshot', { format: 'png' });
  console.log('Shot result keys:', Object.keys(shot));
  console.log('Shot data length:', shot.data ? shot.data.length : 'none');

  if (shot.data) {
    const targetFile = path.join(ARTIFACT_DIR, 'agripredict_live.png');
    fs.writeFileSync(targetFile, Buffer.from(shot.data, 'base64'));
    console.log('Successfully saved to:', targetFile);
  }

  chrome.kill();
  flask.kill();
}

capture().catch(err => {
  console.error(err);
  process.exit(1);
});
