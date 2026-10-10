// Electron main process: starts llama-server and the Python backend (see
// services.js), owns the WebSocket connection to the Python bridge and
// forwards validated messages to the overlay and control panel windows.
// Answered questions are kept in the conversation log (conversation.js), and
// in the active game session too (sessions.js), whose game's wiki the backend
// answers from. The panel's Settings tab edits gvision.config.json (settings.js).
//
// Two-PC mode (Settings > Network > This PC is): a gaming PC runs only the
// overlay, the panel and the edge (screen and voice link), and connects to
// an AI server PC's bridge; the AI server runs the models with no overlay and
// a control server the gaming PC's panel uses for its services, settings,
// downloads and wikis (remote.js).
//
//   npm start                    Qwen + live perception + voice agent
//   npm start -- --demo          synthetic demo, no Qwen
//   npm start -- --server        AI server for a gaming PC, whatever Settings says
//   npm start -- --no-services   connect only; start the processes by hand
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { relaunchCommand } = require('./update');
const { app, BrowserWindow, globalShortcut, ipcMain, nativeTheme, screen, shell } = require('electron');
const { parseMessage, validateMessage, makeMessage } = require('./protocol');
const {
  CONFIG_FILE, DEFAULTS, REPO_ROOT, WIKI_DIR, bridgeUrl, createServices, loadConfig, portOpen, serviceCommands, visionService,
} = require('./services');
const models = require('./models');
const { ConversationLog } = require('./conversation');
const { SessionStore, gameList, wikiIndexes } = require('./sessions');
const { ClockSync, RemoteClient, createControlServer, gitBuild, localAddresses } = require('./remote');
const GAMES = require('./games.json');
const { SPEC, readFile, resolveSettings, saveSettings } = require('./settings');

const argv = process.argv.slice(1);
const MODE = argv.includes('--demo') ? 'demo' : null;
const ROLE_ARG = argv.includes('--server') && !MODE ? 'server' : null;
const MANAGE_SERVICES = !argv.includes('--no-services') && !process.env.GVISION_NO_SERVICES;
const LOG_DIR = path.join(REPO_ROOT, 'logs');
const conversation = new ConversationLog(path.join(LOG_DIR, 'conversation'));
const sessions = new SessionStore(path.join(LOG_DIR, 'sessions'));
let wikiStatus = null; // the backend's last word on the session game's wiki index

let config = null;
let configError = null;
try {
  config = loadConfig(MODE, ROLE_ARG);
} catch (err) {
  configError = `${CONFIG_FILE}: ${err.message}`;
}
const ROLE = config ? config.role : ROLE_ARG || 'standalone';
const GAMING = ROLE === 'gaming';
const SERVER = ROLE === 'server';
const SERVER_HOST = config ? config.serverHost : '';
const CONTROL_PORT = config ? config.controlPort : DEFAULTS.network.controlPort;
const BRIDGE_PORT = config ? config.python.port : 8765;
const BRIDGE_URL = process.env.GVISION_BRIDGE_URL
  || (GAMING ? SERVER_HOST && bridgeUrl(SERVER_HOST, BRIDGE_PORT) : `ws://127.0.0.1:${BRIDGE_PORT}`);
const BUILD = gitBuild(REPO_ROOT);
const RECONNECT_MS = 1000;
const DISMISS_HOTKEY = 'CommandOrControl+Shift+X';

let overlay = null;
let panel = null;
let socket = null;
let connected = false;
let services = [];
let quitting = false;

// Gaming PC: the AI server's control server, and what it last said.
const remote = GAMING && SERVER_HOST ? new RemoteClient(`http://${bridgeUrl(SERVER_HOST, CONTROL_PORT).slice(5)}`) : null;
const clock = new ClockSync(); // gaming PC time minus AI server time, for the overlay
let serverInfo = null;
let serverError = GAMING && !SERVER_HOST ? 'Set the AI server\'s address in Settings > Network.' : null;
let serverIndexes = []; // the AI server's wiki indexes, for the session picker
// AI server: the gaming PC that last called, and when.
let peer = null;
let controlError = null;
const SERVER_SIDE = new Set(SPEC.filter((s) => s.side === 'server').map((s) => s.key));
const PEER_TIMEOUT_MS = 6000;
const POLL_MS = 2000;

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
  if (!BRIDGE_URL) return; // a gaming PC without the AI server's address
  socket = new WebSocket(BRIDGE_URL);
  socket.addEventListener('open', () => {
    connected = true;
    broadcast('gvision:connection', { connected, url: BRIDGE_URL });
    // The gaming PC's panel owns the session and the overlay; the AI
    // server's own panel only watches.
    if (SERVER) return;
    // The backend starts with its default dim strength; send the saved one.
    sendToBridge(makeMessage('config_changed', { changes: { 'visual_effects.dim_strength': currentSettings().dimStrength } }));
    sendSession();
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
      try {
        const added = sessions.add(result.msg);
        if (added) toPanel('gvision:session-exchange', added);
      } catch (err) {
        console.error('[sessions]', err);
      }
    } else if (result.ok && result.msg.type === 'wiki_status') {
      wikiStatus = result.msg;
      toPanel('gvision:wiki', wikiStatus);
    } else if (result.ok) {
      // Frame times are the AI server's clock; the overlay extrapolates with this PC's.
      if (GAMING && result.msg.type === 'objects') result.msg.frame_ts += clock.offset;
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

// The backend answers with the active session's game wiki; tell it which.
function sendSession() {
  const active = sessions.active();
  sendToBridge(makeMessage('session', { session_id: active ? active.id : null, game: active ? active.game : null }));
}

async function sessionsState() {
  const active = sessions.active();
  if (remote) {
    try {
      serverIndexes = await remote.get('/games', 1500);
    } catch {
      // keep what the AI server said last
    }
  }
  return {
    sessions: sessions.list(),
    activeId: active ? active.id : null,
    games: gameList(GAMES, GAMING ? serverIndexes : WIKI_DIR),
    wiki: wikiStatus && active && wikiStatus.game_id === active.game.id ? wikiStatus : null,
  };
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

// On a gaming PC, the AI server's services are listed too, as "server:<name>".
function servicesState() {
  if (configError) return { managed: true, error: configError, services: [] };
  const list = services.map((s) => s.info());
  if (GAMING) {
    if (serverInfo) {
      for (const s of serverInfo.services || []) list.push({ ...s, name: `server:${s.name}`, label: `${s.label} · AI server` });
    } else {
      list.push({ name: 'server', label: 'AI server', state: SERVER_HOST ? 'failed' : 'stopped', detail: serverError, remote: true });
    }
  }
  return { managed: MANAGE_SERVICES || GAMING, error: null, services: list };
}

// --- Two-PC mode -------------------------------------------------------------------

function networkState() {
  const state = {
    role: ROLE, forced: Boolean(ROLE_ARG), build: BUILD, addresses: localAddresses(), bridgePort: BRIDGE_PORT,
    controlPort: CONTROL_PORT,
  };
  if (GAMING) {
    state.server = {
      host: SERVER_HOST, ok: Boolean(serverInfo), error: serverError, build: serverInfo ? serverInfo.build : null,
      sameVersion: !serverInfo || !serverInfo.build || !BUILD || serverInfo.build === BUILD,
      backend: serverInfo ? serverInfo.backend : false,
    };
  }
  if (SERVER) {
    state.peer = peer && Date.now() - peer.seen < PEER_TIMEOUT_MS ? { address: peer.address } : null;
    state.error = controlError;
  }
  return state;
}

function broadcastNetwork() {
  toPanel('gvision:network', networkState());
  broadcast('gvision:services', servicesState());
}

async function pollServer() {
  if (!remote) return;
  const sent = Date.now();
  try {
    const info = await remote.get('/info');
    clock.add(sent, Date.now(), info.time);
    serverInfo = info;
    serverError = null;
  } catch (err) {
    serverInfo = null;
    serverError = err.message;
  }
  broadcastNetwork();
}

// Can this PC reach an AI server at `host`? For Settings > Network's test button.
async function testConnection(host) {
  host = String(host || '').trim();
  if (!host) return { ok: false, error: 'Type the AI server\'s address first.' };
  const client = new RemoteClient(`http://${bridgeUrl(host, CONTROL_PORT).slice(5)}`);
  const sent = Date.now();
  try {
    const info = await client.get('/info', 4000);
    const ms = Date.now() - sent;
    if (info.app !== 'g-vision') return { ok: false, error: `${host} answered, but it isn't a G-VISION AI server.` };
    const backend = await portOpen(BRIDGE_PORT, host, 1500);
    return { ok: true, ms, build: info.build, sameVersion: !info.build || !BUILD || info.build === BUILD, backend };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// What a gaming PC may do on this AI server. Network settings are each PC's
// own, so only the server-side ones can be changed from there.
function controlRoutes() {
  return {
    'GET /info': () => ({
      app: 'g-vision', role: ROLE, build: BUILD, time: Date.now() / 1000, addresses: localAddresses(),
      services: services.map((s) => s.info()), backend: connected,
    }),
    'GET /settings': () => ({ ok: true, values: currentSettings() }),
    'POST /settings': ({ body }) => {
      const changes = body && typeof body.changes === 'object' && body.changes ? body.changes : {};
      const refused = Object.keys(changes).filter((k) => !SERVER_SIDE.has(k));
      if (refused.length) throw new Error(`can't change ${refused.join(', ')} from another PC`);
      const values = saveSettings(CONFIG_FILE, changes);
      applySettings(changes, values);
      toPanel('gvision:settings-changed', values);
      return { ok: true, values };
    },
    'GET /vision-models': () => visionModels(),
    'POST /vision-model': ({ body }) => {
      visionJob = { busy: true, id: body.id, progress: null, result: null };
      const job = visionJob;
      useVisionModel(String(body.id)).then((result) => {
        job.busy = false;
        job.result = result;
        if (result.ok) toPanel('gvision:settings-changed', result.values);
      });
      return { ok: true };
    },
    'GET /vision-download': () => visionJob || { busy: false, progress: null, result: null },
    'POST /vision-download/cancel': () => {
      if (visionDownload) visionDownload.abort();
      return { ok: true };
    },
    'POST /services/:name/:action': ({ params }) => {
      if (!['restart', 'stop'].includes(params.action)) throw new Error(`unknown action "${params.action}"`);
      serviceAction(params.name, params.action);
      return { ok: true };
    },
    'GET /games': () => wikiIndexes(WIKI_DIR),
    'POST /update': () => {
      setTimeout(() => updateAndRestart(), 300); // after the answer is out
      return { ok: true };
    },
  };
}

function startControlServer() {
  const server = createControlServer(controlRoutes(), {
    onRequest: (address) => {
      const fresh = !peer || Date.now() - peer.seen >= PEER_TIMEOUT_MS || peer.address !== address;
      peer = { address, seen: Date.now() };
      if (fresh) broadcastNetwork();
    },
  });
  server.on('error', (err) => {
    controlError = err.code === 'EADDRINUSE'
      ? `Port ${CONTROL_PORT} is taken, so the gaming PC can't reach this one. Close whatever uses it, or set network.controlPort in gvision.config.json on both PCs.`
      : `Control server: ${err.message}`;
    console.error('[control]', controlError);
    broadcastNetwork();
  });
  server.listen(CONTROL_PORT, '0.0.0.0', () => console.log(`[control] AI server listening on port ${CONTROL_PORT}`));
}

// Settings > Network: a new role or address takes a restart of the whole app.
async function relaunchApp() {
  quitting = true;
  await Promise.allSettled(services.map((s) => s.stop()));
  app.relaunch({ args: process.argv.slice(1).filter((a) => a !== '--server') });
  app.exit(0);
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
let visionJob = null; // AI server: the download a gaming PC asked for, polled by it

function llamaDir() {
  return (config || loadConfig(MODE, ROLE_ARG)).llama.dir;
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
  if (!MANAGE_SERVICES || !config || GAMING) return;
  const cfg = loadConfig(MODE, ROLE_ARG);
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
        onProgress: (p) => {
          if (visionJob && visionJob.busy) visionJob.progress = { id, ...p };
          toPanel('gvision:vision-download', { id, ...p });
        },
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
  // On a gaming PC the backend (Settings' restart button) and the
  // "server:" services are the AI server's.
  if (GAMING && (name === 'backend' || name.startsWith('server:'))) {
    if (!remote) return;
    remote.post(`/services/${encodeURIComponent(name.replace(/^server:/, ''))}/${action}`)
      .then(() => setTimeout(pollServer, 300))
      .catch((err) => toPanel('gvision:error', err.message));
    return;
  }
  const s = services.find((x) => x.name === name);
  if (!s) return;
  if (action === 'restart') {
    try {
      const cmd = serviceCommands(loadConfig(MODE, ROLE_ARG))[name];
      if (cmd) s.update(cmd);
    } catch (err) {
      s._set('failed', `${CONFIG_FILE}: ${err.message}`);
      return;
    }
  }
  s[action]().catch((err) => s._set('failed', err.message));
}

// Settings > *: on a gaming PC the server-side settings are the AI server's,
// read and saved over the network; the rest are this PC's.
async function getSettings() {
  let error = null;
  try {
    readFile(CONFIG_FILE);
  } catch (err) {
    error = `${CONFIG_FILE}: ${err.message}`;
  }
  const values = currentSettings();
  let serverSettings = null;
  if (GAMING) {
    serverSettings = { ok: false, error: serverError };
    if (remote) {
      try {
        const r = await remote.get('/settings');
        for (const k of SERVER_SIDE) values[k] = r.values[k];
        serverSettings = { ok: true, error: null };
      } catch (err) {
        serverSettings = { ok: false, error: err.message };
      }
    }
  }
  // First launch: nobody has said yet what this PC is (the panel asks).
  let roleChosen = Boolean(ROLE_ARG) || MODE === 'demo';
  try {
    roleChosen = roleChosen || 'role' in (readFile(CONFIG_FILE).settings || {});
  } catch {
    roleChosen = true; // a broken file has bigger problems; don't stack a dialog on them
  }
  return { spec: SPEC, values, file: CONFIG_FILE, error, role: ROLE, forced: Boolean(ROLE_ARG), serverSettings, roleChosen };
}

// Live settings reach the backend now, the rest on its next restart.
function applySettings(changes, values) {
  if ('dimStrength' in changes) {
    sendToBridge(makeMessage('config_changed', { changes: { 'visual_effects.dim_strength': values.dimStrength } }));
  }
  // The vision server's own flags change with it: restart it (and the backend) now.
  if ('visionReasoning' in changes) syncVisionService();
}

async function saveSettingsAnywhere(changes) {
  try {
    const here = {};
    const there = {};
    for (const [k, v] of Object.entries(changes)) (GAMING && SERVER_SIDE.has(k) ? there : here)[k] = v;
    let values = currentSettings();
    if (Object.keys(here).length) {
      values = saveSettings(CONFIG_FILE, here);
      applySettings(here, values);
      // The push-to-talk key is held on the gaming PC: its link restarts with it.
      if (GAMING && 'pttKey' in here) serviceAction('edge', 'restart');
    }
    if (Object.keys(there).length) {
      if (!remote) throw new Error(serverError || 'no AI server');
      const r = await remote.post('/settings', { changes: there });
      for (const k of SERVER_SIDE) values[k] = r.values[k];
    } else if (GAMING && remote) {
      const r = await remote.get('/settings').catch(() => null);
      if (r) for (const k of SERVER_SIDE) values[k] = r.values[k];
    }
    return { ok: true, values };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// Gaming PC: the AI server downloads and switches the vision model; this
// follows its progress for the panel.
async function useRemoteVisionModel(id) {
  try {
    await remote.post('/vision-model', { id });
    for (;;) {
      await new Promise((r) => setTimeout(r, 500));
      const job = await remote.get('/vision-download');
      if (job.progress) toPanel('gvision:vision-download', job.progress);
      if (!job.busy && job.result) {
        const result = { ...job.result };
        if (result.values) result.values = { ...currentSettings(), ...pick(result.values, SERVER_SIDE) };
        return result;
      }
    }
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

function pick(values, keys) {
  return Object.fromEntries(Object.entries(values).filter(([k]) => keys.has(k)));
}

// Update button: stop the services, then pull and reinstall what changed and
// start again. On Windows the launcher does that in a visible console window.
async function updateAndRestart() {
  quitting = true;
  // Both PCs have to run the same version: update the AI server too.
  if (remote) await remote.post('/update', {}, 2000).catch((err) => console.warn('[update] AI server:', err.message));
  await Promise.allSettled(services.map((s) => s.stop()));
  if (process.platform === 'win32') {
    const launcher = path.join(REPO_ROOT, 'G-VISION.bat');
    const extra = `${MODE === 'demo' ? ' --demo' : ''}${ROLE_ARG ? ' --server' : ''}`;
    // Verbatim so cmd sees the quotes around a path with spaces as written.
    spawn('cmd.exe', ['/c', relaunchCommand(launcher, extra)], {
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
  ipcMain.handle('gvision:sessions', () => sessionsState());
  ipcMain.handle('gvision:session', (_event, id) => sessions.get(id));
  ipcMain.handle('gvision:start-session', async (_event, game) => {
    try {
      const session = sessions.start(game);
      wikiStatus = null;
      sendSession();
      return { ok: true, session, state: await sessionsState() };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });
  ipcMain.handle('gvision:end-session', () => {
    sessions.end();
    wikiStatus = null;
    sendSession();
    return sessionsState();
  });
  ipcMain.handle('gvision:delete-session', (_event, id) => {
    const active = sessions.active();
    sessions.remove(id);
    if (active && active.id === id) sendSession();
    return sessionsState();
  });
  ipcMain.handle('gvision:update-wiki', (_event, full) => sendToBridge(makeMessage('wiki_update', { full: Boolean(full) })));
  ipcMain.handle('gvision:get-settings', () => getSettings());
  ipcMain.handle('gvision:save-settings', (_event, changes) => saveSettingsAnywhere(changes));
  ipcMain.handle('gvision:vision-models', () => (remote ? remote.get('/vision-models').catch(() => []) : visionModels()));
  ipcMain.handle('gvision:use-vision-model', (_event, id) => (remote ? useRemoteVisionModel(id) : useVisionModel(id)));
  ipcMain.handle('gvision:cancel-vision-download', () => (remote
    ? remote.post('/vision-download/cancel').catch(() => null)
    : visionDownload && visionDownload.abort()));
  ipcMain.handle('gvision:network', () => networkState());
  ipcMain.handle('gvision:test-connection', (_event, host) => testConnection(host));
  ipcMain.handle('gvision:relaunch', () => relaunchApp());
  ipcMain.handle('gvision:open-logs', () => {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    return shell.openPath(LOG_DIR);
  });
  if (!SERVER) createOverlay(); // the game, and so the overlay, is on the gaming PC
  createPanel();
  if (!SERVER) globalShortcut.register(DISMISS_HOTKEY, () => dismiss('dismiss hotkey'));
  startServices();
  connectBridge();
  if (SERVER) {
    startControlServer();
    setInterval(broadcastNetwork, POLL_MS); // the gaming PC shows up and goes away
  }
  if (remote) {
    pollServer();
    setInterval(pollServer, POLL_MS);
  }
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
