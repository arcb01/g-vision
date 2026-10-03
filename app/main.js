// Electron main process: owns the WebSocket connection to the Python bridge
// and forwards validated messages to the overlay and control panel windows.
'use strict';

const path = require('node:path');
const { app, BrowserWindow, globalShortcut, ipcMain, screen } = require('electron');
const { parseMessage, validateMessage, makeMessage } = require('./protocol');

const BRIDGE_URL = process.env.GVISION_BRIDGE_URL || 'ws://127.0.0.1:8765';
const RECONNECT_MS = 1000;
const DISMISS_HOTKEY = 'CommandOrControl+Shift+X';

let overlay = null;
let panel = null;
let socket = null;
let connected = false;

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

app.whenReady().then(() => {
  ipcMain.handle('gvision:send', (_event, msg) => {
    // A clear from the panel must reach the overlay even with no bridge.
    if (msg && msg.type === 'clear') broadcast('gvision:message', msg);
    return sendToBridge(msg);
  });
  ipcMain.handle('gvision:connection', () => ({ connected, url: BRIDGE_URL }));
  createOverlay();
  createPanel();
  globalShortcut.register(DISMISS_HOTKEY, () => dismiss('dismiss hotkey'));
  connectBridge();
});

app.on('will-quit', () => globalShortcut.unregisterAll());
