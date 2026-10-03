// Electron main process: starts llama-server and the Python backend (see
// services.js), owns the WebSocket connection to the Python bridge and
// forwards validated messages to the overlay and control panel windows.
//
//   npm start                    Qwen + live perception + voice agent
//   npm start -- --demo          synthetic demo, no Qwen
//   npm start -- --no-services   connect only; start the processes by hand
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { app, BrowserWindow, globalShortcut, ipcMain, screen, shell } = require('electron');
const { parseMessage, validateMessage, makeMessage } = require('./protocol');
const { CONFIG_FILE, REPO_ROOT, createServices, loadConfig } = require('./services');

const argv = process.argv.slice(1);
const MODE = argv.includes('--demo') ? 'demo' : null;
const MANAGE_SERVICES = !argv.includes('--no-services') && !process.env.GVISION_NO_SERVICES;
const LOG_DIR = path.join(REPO_ROOT, 'logs');

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
  });
  socket.addEventListener('message', (event) => {
    const result = parseMessage(String(event.data));
    if (result.ok) {
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
  panel = new BrowserWindow({
    width: 960,
    height: 680,
    title: 'G-VISION',
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

function startServices() {
  if (!MANAGE_SERVICES || !config) return;
  services = createServices(config, { logDir: LOG_DIR });
  for (const s of services) {
    s.on('change', () => broadcast('gvision:services', servicesState()));
    s.start().catch((err) => console.error(`[${s.name}]`, err));
  }
}

app.whenReady().then(() => {
  ipcMain.handle('gvision:send', (_event, msg) => {
    // A clear from the panel must reach the overlay even with no bridge.
    if (msg && msg.type === 'clear') broadcast('gvision:message', msg);
    return sendToBridge(msg);
  });
  ipcMain.handle('gvision:connection', () => ({ connected, url: BRIDGE_URL }));
  ipcMain.handle('gvision:services', () => servicesState());
  ipcMain.handle('gvision:restart-service', (_event, name) => {
    const s = services.find((x) => x.name === name);
    if (s) s.restart().catch((err) => console.error(`[${name}]`, err));
  });
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
