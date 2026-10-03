// Starts and supervises the processes the app needs, so one launch runs
// everything: Qwen in llama-server and the Python backend (bridge, perception,
// agent, audio). Paths and arguments come from gvision.config.json at the repo
// root, falling back to the defaults below.
'use strict';

const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn, spawnSync } = require('node:child_process');
const { EventEmitter } = require('node:events');

const REPO_ROOT = path.resolve(__dirname, '..');
const CONFIG_FILE = path.join(REPO_ROOT, 'gvision.config.json');
const IS_WIN = process.platform === 'win32';

const DEFAULTS = {
  llama: {
    enabled: true,
    // llama.cpp folder with llama-server and the Qwen weights.
    dir: '~/Desktop/coding projects/G-vision-lab',
    server: IS_WIN ? 'llama/llama-server.exe' : 'llama/llama-server',
    model: 'models/Qwen3.5-2B-Q4_K_M.gguf',
    mmproj: 'models/mmproj-F16.gguf',
    port: 8080,
    args: ['-ngl', '99', '-c', '8192', '--jinja', '--reasoning', 'off'],
    readyTimeoutS: 180,
  },
  python: {
    enabled: true,
    // Relative to python/ in this repo. null: the first virtual environment
    // found among $VIRTUAL_ENV, python/.venv and .venv at the repo root.
    exe: null,
    args: ['--live', '--agent'],
    port: 8765,
    readyTimeoutS: 120,
  },
};

const MODES = {
  agent: { llama: { enabled: true }, python: { args: ['--live', '--agent'] } },
  demo: { llama: { enabled: false }, python: { args: ['--demo'] } },
};

function isObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function merge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) {
    out[k] = isObject(v) && isObject(base[k]) ? merge(base[k], v) : v;
  }
  return out;
}

function expandPath(p, baseDir) {
  if (p === '~' || p.startsWith('~/') || p.startsWith('~\\')) p = path.join(os.homedir(), p.slice(1));
  return path.resolve(baseDir, p);
}

const VENV_PYTHON = IS_WIN ? 'Scripts/python.exe' : 'bin/python';

function pythonCandidates(pythonDir, env = process.env) {
  const dirs = [];
  if (env.VIRTUAL_ENV) dirs.push(env.VIRTUAL_ENV);
  dirs.push(path.join(pythonDir, '.venv'), path.join(path.dirname(pythonDir), '.venv'));
  return dirs.map((d) => path.join(d, VENV_PYTHON));
}

// Defaults <- config file <- mode (from the command line). Relative paths in
// the llama section resolve against llama.dir, the Python exe against python/.
function resolveConfig({ fileConfig = {}, mode = null, configDir = REPO_ROOT, env = process.env } = {}) {
  let cfg = merge(DEFAULTS, fileConfig);
  if (mode) {
    if (!MODES[mode]) throw new Error(`unknown mode "${mode}" (expected ${Object.keys(MODES).join(', ')})`);
    cfg = merge(cfg, MODES[mode]);
  }
  const llamaDir = expandPath(cfg.llama.dir, configDir);
  const pythonDir = path.join(REPO_ROOT, 'python');
  return {
    llama: {
      ...cfg.llama,
      dir: llamaDir,
      server: expandPath(cfg.llama.server, llamaDir),
      model: expandPath(cfg.llama.model, llamaDir),
      mmproj: cfg.llama.mmproj ? expandPath(cfg.llama.mmproj, llamaDir) : null,
    },
    python: { ...cfg.python, dir: pythonDir, ...resolvePython(cfg.python.exe, pythonDir, env) },
  };
}

function resolvePython(exe, pythonDir, env) {
  if (exe) return { exe: expandPath(exe, pythonDir), notFound: null };
  const candidates = pythonCandidates(pythonDir, env);
  const found = candidates.find((c) => fs.existsSync(c));
  if (found) return { exe: found, notFound: null };
  return {
    exe: candidates[0],
    notFound: `no Python environment found; looked for ${candidates.join(', ')}. Set python.exe in gvision.config.json`,
  };
}

function loadConfig(mode = null) {
  let fileConfig = {};
  if (fs.existsSync(CONFIG_FILE)) fileConfig = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
  return resolveConfig({ fileConfig, mode });
}

function llamaCommand(llama) {
  const args = ['-m', llama.model];
  if (llama.mmproj) args.push('--mmproj', llama.mmproj);
  args.push(...llama.args, '--host', '127.0.0.1', '--port', String(llama.port));
  return { name: 'qwen', label: 'Qwen (llama-server)', exe: llama.server, args, cwd: llama.dir };
}

function pythonCommand(py, llamaPort) {
  const args = ['-m', 'gvision', ...py.args, '--port', String(py.port)];
  if (py.args.includes('--agent') && !py.args.includes('--qwen-url')) {
    args.push('--qwen-url', `http://127.0.0.1:${llamaPort}`);
  }
  return { name: 'backend', label: 'Python backend', exe: py.exe, args, cwd: py.dir, notFound: py.notFound };
}

function portOpen(port, host = '127.0.0.1', timeoutMs = 500) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host });
    const done = (ok) => {
      sock.destroy();
      resolve(ok);
    };
    sock.setTimeout(timeoutMs, () => done(false));
    sock.once('connect', () => done(true));
    sock.once('error', () => done(false));
  });
}

async function httpOk(url, timeoutMs = 1000) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    return r.ok;
  } catch {
    return false;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// PIDs listening on a local TCP port, from `netstat -ano` on Windows. The
// state column is translated on non-English Windows ("ESCUCHANDO"), so a
// listening socket is recognized by its foreign address ending in ":0".
function parseNetstat(text, port) {
  const pids = new Set();
  for (const line of text.split(/\r?\n/)) {
    const cols = line.trim().split(/\s+/);
    if (cols[0] !== 'TCP' || cols.length < 5 || !cols[2].endsWith(':0')) continue;
    if (cols[1].endsWith(`:${port}`)) pids.add(Number(cols[cols.length - 1]));
  }
  pids.delete(0);
  return [...pids];
}

// Which process holds a port: used to stop a server this app did not start
// (an orphan from an earlier run, or one started by hand).
function listeningPids(port) {
  if (IS_WIN) {
    const r = spawnSync('netstat', ['-ano', '-p', 'TCP'], { encoding: 'utf8', windowsHide: true });
    return r.status === 0 ? parseNetstat(r.stdout, port) : [];
  }
  const r = spawnSync('lsof', ['-t', `-iTCP:${port}`, '-sTCP:LISTEN'], { encoding: 'utf8' });
  if (r.error || !r.stdout) return [];
  return r.stdout.split(/\s+/).filter(Boolean).map(Number);
}

function killTree(pid) {
  if (IS_WIN) {
    // Kill the whole tree; plain kill() would leave grandchildren behind (a
    // venv's python.exe is a launcher that runs the real interpreter).
    spawnSync('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true });
  } else {
    try {
      process.kill(pid, 'SIGTERM');
    } catch {
      // already gone
    }
  }
}

// One child process: spawn, wait until `isReady()` passes, keep the last log
// lines for the panel, stop the whole process tree on quit.
class Service extends EventEmitter {
  constructor({ name, label, exe, args, cwd, env, isReady, readyTimeoutS, logDir, notFound = null, port = null }) {
    super();
    Object.assign(this, { name, label, exe, args, cwd, env, isReady, readyTimeoutS, logDir, notFound, port });
    this.state = 'stopped';
    this.detail = '';
    this.child = null;
    this.external = false;
    this.tail = [];
  }

  info() {
    return { name: this.name, label: this.label, state: this.state, detail: this.detail };
  }

  // New command line (after gvision.config.json changed); used on the next start.
  update({ exe, args, cwd, notFound = null }) {
    Object.assign(this, { exe, args, cwd, notFound });
  }

  _set(state, detail = '') {
    this.state = state;
    this.detail = detail;
    this.emit('change', this.info());
  }

  async start() {
    if (this.child) return;
    // Something already answers (a llama-server started by hand, or an orphan
    // from a previous run): use it rather than fighting over the port.
    if (await this.isReady()) {
      this.external = true;
      this._set('ready', 'already running');
      return;
    }
    this.external = false;
    if (!fs.existsSync(this.exe)) {
      this._set('failed', this.notFound || `not found: ${this.exe}`);
      return;
    }
    if (this.cwd && !fs.existsSync(this.cwd)) {
      this._set('failed', `folder not found: ${this.cwd}`);
      return;
    }
    this._set('starting');
    this.tail = [];
    let log = null;
    if (this.logDir) {
      fs.mkdirSync(this.logDir, { recursive: true });
      log = fs.createWriteStream(path.join(this.logDir, `${this.name}.log`), { flags: 'w' });
    }
    const child = spawn(this.exe, this.args, {
      cwd: this.cwd,
      env: { ...process.env, ...this.env },
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    this.child = child;
    const onData = (buf) => {
      if (log) log.write(buf);
      for (const line of String(buf).split(/\r?\n/)) if (line.trim()) this.tail.push(line);
      if (this.tail.length > 40) this.tail.splice(0, this.tail.length - 40);
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('error', (err) => {
      this.child = null;
      this._set('failed', err.message);
    });
    // 'close' rather than 'exit': it fires after the last output is read.
    child.on('close', (code, signal) => {
      if (log) log.end();
      if (this.child === child) this.child = null;
      if (child.stopRequested) return;
      const last = this.tail.length ? this.tail[this.tail.length - 1] : '';
      this._set('failed', `exited (${signal || code})${last ? `: ${last}` : ''}`);
    });
    const deadline = Date.now() + this.readyTimeoutS * 1000;
    while (this.child === child && this.state === 'starting') {
      if (await this.isReady()) {
        if (this.child === child) this._set('ready');
        return;
      }
      if (Date.now() > deadline) {
        this._set('failed', `not ready after ${this.readyTimeoutS} s`);
        return;
      }
      await sleep(500);
    }
  }

  async stop() {
    const child = this.child;
    if (!child) {
      if (this.external) await this._stopExternal();
      else this._set('stopped');
      return;
    }
    child.stopRequested = true;
    this._set('stopping');
    const exited = child.exitCode !== null || child.signalCode !== null
      ? Promise.resolve()
      : new Promise((r) => child.once('exit', r));
    killTree(child.pid);
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000);
    await Promise.race([exited, sleep(6000)]);
    clearTimeout(timer);
    if (this.child === child) this.child = null;
    this._set('stopped');
  }

  // A server we found already running: stop whatever holds its port.
  async _stopExternal() {
    const pids = this.port ? listeningPids(this.port).filter((p) => p !== process.pid) : [];
    if (!pids.length) {
      this._set(this.state, `can't stop it from here: no process found on port ${this.port}`);
      return;
    }
    this._set('stopping');
    for (const pid of pids) killTree(pid);
    const deadline = Date.now() + 6000;
    while (Date.now() < deadline && (await this.isReady())) await sleep(250);
    if (await this.isReady()) {
      this._set('ready', `couldn't stop the process on port ${this.port} (pid ${pids.join(', ')})`);
      return;
    }
    this.external = false;
    this._set('stopped');
  }

  async restart() {
    await this.stop();
    if (this.state === 'stopped') await this.start();
  }
}

// The command line for each service name, from a (re)loaded config.
function serviceCommands(cfg) {
  return {
    qwen: llamaCommand(cfg.llama),
    backend: pythonCommand(cfg.python, cfg.llama.port),
  };
}

function createServices(cfg, { logDir = path.join(REPO_ROOT, 'logs') } = {}) {
  const services = [];
  if (cfg.llama.enabled) {
    const port = cfg.llama.port;
    services.push(
      new Service({
        ...llamaCommand(cfg.llama),
        isReady: () => httpOk(`http://127.0.0.1:${port}/health`),
        port,
        readyTimeoutS: cfg.llama.readyTimeoutS,
        logDir,
      }),
    );
  }
  if (cfg.python.enabled) {
    const port = cfg.python.port;
    services.push(
      new Service({
        ...pythonCommand(cfg.python, cfg.llama.port),
        env: { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
        isReady: () => portOpen(port),
        port,
        readyTimeoutS: cfg.python.readyTimeoutS,
        logDir,
      }),
    );
  }
  return services;
}

module.exports = {
  CONFIG_FILE,
  DEFAULTS,
  MODES,
  REPO_ROOT,
  Service,
  createServices,
  llamaCommand,
  loadConfig,
  merge,
  parseNetstat,
  portOpen,
  pythonCandidates,
  pythonCommand,
  resolveConfig,
  serviceCommands,
};
