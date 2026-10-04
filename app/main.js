// Electron main process: starts llama-server and the Python backend (see
// services.js), owns the WebSocket connection to the Python bridge and
// forwards validated messages to the overlay and control panel windows.
// Answered questions are kept in the conversation log (conversation.js) and
// the panel's Settings tab edits gvision.config.json (settings.js).
//
//   npm start                    Qwen + live perception + voice agent
//   npm start -- --demo          synthetic demo, no Qwen
//   npm start -- --no-services   connect only; start the processes by hand
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { app, BrowserWindow, globalShortcut, ipcMain, nativeTheme, screen, shell } = require('electron');
const { parseMessage, validateMessage, makeMessage } = require('./protocol');
const { CONFIG_FILE, REPO_ROOT, createServices, loadConfig, serviceCommands, visionService } = require('./services');
const models = require('./models');
const { ConversationLog } = require('./conversation');
const { SPEC, readFile, resolveSettings, saveSettings } = require('./settings');

const argv = process.argv.slice(1);
const MODE = argv.includes('--demo') ? 'demo' : null;
const MANAGE_SERVICES = !argv.includes('--no-services') && !process.env.GVISION_NO_SERVICES;
const LOG_DIR = path.join(REPO_ROOT, 'logs');
const conversation = new ConversationLog(path.join(LOG_DIR, 'conversation'));

let config = null;
let configError = null;
try {
  config = loadConfig(MODE);
} catch (err) {
  configError = `${CONFIG_FILE}: ${err.message}`;
}
const BRIDGE_PORT = config ? config.python.port : 8765;
const BRIDGE_URL = process.env.GVISION_BRIDGE_URL || `ws://127.0.0.1:${BRIDGE_PORT}`;
const RECONNECT_MS = 1000;
const DISMISS_HOTKEY = 'CommandOrControl+Shift+X';

let overlay = null;
let panel = null;
let socket = null;
let connected = false;
let services = [];
let quitting = false;

// Settings as saved now; a bad file reads as the defaults.
function currentSettings() {
  try {
    return resolveSettings(readFile(CONFIG_FILE).settings);
  } catch {
    return resolveSettings();
  }
}

function toPanel(channel, payload) {
  if (panel && !panel.isDestroyed()) panel.webContents.send(channel, payload);
}

function broadcast(channel, payload) {
  for (const win of [overlay, panel]) {
    if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
  }
}

function sendToBridge(msg) {
  const result = validateMessage(msg);
  if (!result.ok) {
    console.warn('[bridge] refusing to send invalid message:', result.errors);
    return false;
  }
  if (!connected) return false;
  socket.send(JSON.stringify(msg));
  return true;
}

function connectBridge() {
  socket = new WebSocket(BRIDGE_URL);
  socket.addEventListener('open', () => {
    connected = true;
    broadcast('gvision:connection', { connected, url: BRIDGE_URL });
    // The backend starts with its default dim strength; send the saved one.
    sendToBridge(makeMessage('config_changed', { changes: { 'visual_effects.dim_strength': currentSettings().dimStrength } }));
  });
  socket.addEventListener('message', (event) => {
    const result = parseMessage(String(event.data));
    if (result.ok && result.msg.type === 'exchange') {
      // Screenshots are big and the overlay has no use for them.
      try {
        toPanel('gvision:exchange', conversation.add(result.msg));
      } catch (err) {
        console.error('[conversation]', err);
      }
    } else if (result.ok) {
      broadcast('gvision:message', result.msg);
    } else {
      console.warn('[bridge] invalid message:', result.errors);
    }
  });
  socket.addEventListener('close', () => {
    if (connected) broadcast('gvision:connection', { connected: false, url: BRIDGE_URL });
    connected = false;
    setTimeout(connectBridge, RECONNECT_MS);
  });
  socket.addEventListener('error', () => {
    // 'close' follows and schedules the reconnect.
  });
}

function createOverlay() {
  const { bounds } = screen.getPrimaryDisplay();
  overlay = new BrowserWindow({
    ...bounds,
    transparent: true,
    frame: false,
    resizable: false,
    movable: false,
    focusable: false,
    skipTaskbar: true,
    hasShadow: false,
    alwaysOnTop: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  overlay.setAlwaysOnTop(true, 'screen-saver');
  overlay.setIgnoreMouseEvents(true); // click-through
  overlay.setContentProtection(true); // excluded from screen capture (plan 5.5)
  overlay.loadFile(path.join(__dirname, 'src', 'overlay.html'));
}

function createPanel() {
  nativeTheme.themeSource = 'dark';
  panel = new BrowserWindow({
    width: 1180,
    height: 800,
    minWidth: 820,
    minHeight: 560,
    title: 'G-VISION',
    backgroundColor: '#0b0d12',
    autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  panel.loadFile(path.join(__dirname, 'src', 'panel.html'));
  panel.on('closed', () => app.quit());
}

function dismiss(reason) {
  const msg = makeMessage('clear', { reason });
  broadcast('gvision:message', msg);
  sendToBridge(msg);
}

function servicesState() {
  if (configError) return { managed: true, error: configError, services: [] };
  return { managed: MANAGE_SERVICES, error: null, services: services.map((s) => s.info()) };
}

function watchService(s) {
  s.on('change', () => broadcast('gvision:services', servicesState()));
}

function startServices() {
  if (!MANAGE_SERVICES || !config) return;
  services = createServices(config, { logDir: LOG_DIR });
  for (const s of services) {
    watchService(s);
    s.start().catch((err) => console.error(`[${s.name}]`, err));
  }
}

// Vision model picker (Settings > Vision): download the files if missing,
// save the choice, then start, swap or stop the second llama-server and
// restart the backend so look and the situation notes use it.
let visionDownload = null;

function llamaDir() {
  return (config || loadConfig(MODE)).llama.dir;
}

function visionModels() {
  let dir = null;
  try {
    dir = llamaDir();
  } catch {
    // No usable config: nothing reads as downloaded.
  }
  return models.VISION_MODELS.map((m) => ({
    id: m.id,
    downloaded: !m.files.length || (dir !== null && models.missingFiles(m.id, dir).length === 0),
  }));
}

async function syncVisionService() {
  if (!MANAGE_SERVICES || !config) return;
  const cfg = loadConfig(MODE);
  const i = services.findIndex((x) => x.name === 'vision');
  if (i >= 0) {
    const old = services[i];
    services.splice(i, 1);
    await old.stop();
  }
  if (cfg.vision) {
    const s = visionService(cfg, LOG_DIR);
    watchService(s);
    services.splice(Math.max(0, services.findIndex((x) => x.name === 'backend')), 0, s);
    s.start().catch((err) => s._set('failed', err.message));
  }
  broadcast('gvision:services', servicesState());
  serviceAction('backend', 'restart');
}

async function useVisionModel(id) {
  if (!models.VISION_MODELS.some((m) => m.id === id)) return { ok: false, error: `unknown vision model "${id}"` };
  if (visionDownload) visionDownload.abort();
  const controller = new AbortController();
  visionDownload = controller;
  try {
    if (id !== 'same') {
      await models.download(id, llamaDir(), {
        signal: controller.signal,
        onProgress: (p) => toPanel('gvision:vision-download', { id, ...p }),
      });
    }
    if (controller.signal.aborted) return { ok: false, error: 'cancelled' };
    const values = saveSettings(CONFIG_FILE, { visionModel: id });
    await syncVisionService();
    return { ok: true, values, models: visionModels() };
  } catch (err) {
    return { ok: false, error: controller.signal.aborted ? 'cancelled' : err.message, models: visionModels() };
  } finally {
    if (visionDownload === controller) visionDownload = null;
  }
}

// Start, restart or stop one service from the panel. A (re)start re-reads
// gvision.config.json, so a fixed path applies without relaunching the app.
function serviceAction(name, action) {
  const s = services.find((x) => x.name === name);
  if (!s) return;
  if (action === 'restart') {
    try {
      const cmd = serviceCommands(loadConfig(MODE))[name];
      if (cmd) s.update(cmd);
    } catch (err) {
      s._set('failed', `${CONFIG_FILE}: ${err.message}`);
      return;
    }
  }
  s[action]().catch((err) => s._set('failed', err.message));
}

// Update button: stop the services, then pull and reinstall what changed and
// start again. On Windows the launcher does that in a visible console window.
async function updateAndRestart() {
  quitting = true;
  await Promise.allSettled(services.map((s) => s.stop()));
  if (process.platform === 'win32') {
    const launcher = path.join(REPO_ROOT, 'G-VISION.bat');
    const extra = MODE === 'demo' ? ' --demo' : '';
    // Verbatim so cmd sees the quotes around a path with spaces as written.
    spawn('cmd.exe', ['/c', `start "G-VISION update" "${launcher}"${extra}`], {
      detached: true,
      stdio: 'ignore',
      windowsVerbatimArguments: true,
    }).unref();
  } else {
    const r = require('node:child_process').spawnSync(process.execPath, [path.join(__dirname, 'update.js')], {
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
      stdio: 'inherit',
    });
    if (r.error) console.error('[update]', r.error);
    app.relaunch();
  }
  app.exit(0);
}

app.whenReady().then(() => {
  ipcMain.handle('gvision:send', (_event, msg) => {
    // A clear from the panel must reach the overlay even with no bridge.
    if (msg && msg.type === 'clear') broadcast('gvision:message', msg);
    return sendToBridge(msg);
  });
  ipcMain.handle('gvision:connection', () => ({ connected, url: BRIDGE_URL }));
  ipcMain.handle('gvision:services', () => servicesState());
  ipcMain.handle('gvision:restart-service', (_event, name) => serviceAction(name, 'restart'));
  ipcMain.handle('gvision:stop-service', (_event, name) => serviceAction(name, 'stop'));
  ipcMain.handle('gvision:update', () => updateAndRestart());
  ipcMain.handle('gvision:get-log', () => conversation.list());
  ipcMain.handle('gvision:clear-log', () => conversation.clear());
  ipcMain.handle('gvision:get-settings', () => {
    let error = null;
    try {
      readFile(CONFIG_FILE);
    } catch (err) {
      error = `${CONFIG_FILE}: ${err.message}`;
    }
    return { spec: SPEC, values: currentSettings(), file: CONFIG_FILE, error };
  });
  // Saves to gvision.config.json; live settings reach the backend now, the
  // rest on its next restart.
  ipcMain.handle('gvision:save-settings', (_event, changes) => {
    try {
      const values = saveSettings(CONFIG_FILE, changes);
      if ('dimStrength' in changes) {
        sendToBridge(makeMessage('config_changed', { changes: { 'visual_effects.dim_strength': values.dimStrength } }));
      }
      return { ok: true, values };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });
  ipcMain.handle('gvision:vision-models', () => visionModels());
  ipcMain.handle('gvision:use-vision-model', (_event, id) => useVisionModel(id));
  ipcMain.handle('gvision:cancel-vision-download', () => visionDownload && visionDownload.abort());
  ipcMain.handle('gvision:open-logs', () => {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    return shell.openPath(LOG_DIR);
  });
  createOverlay();
  createPanel();
  globalShortcut.register(DISMISS_HOTKEY, () => dismiss('dismiss hotkey'));
  startServices();
  connectBridge();
});

// Stop the processes we started before exiting, so nothing keeps the GPU,
// the microphone or the ports busy after the app is closed.
app.on('before-quit', (event) => {
  if (quitting || !services.some((s) => s.child)) return;
  event.preventDefault();
  quitting = true;
  Promise.allSettled(services.map((s) => s.stop())).then(() => app.quit());
});

app.on('will-quit', () => globalShortcut.unregisterAll());
