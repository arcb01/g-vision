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
    // Relative to python/ in this repo.
    exe: IS_WIN ? '.venv/Scripts/python.exe' : '.venv/bin/python',
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

// Defaults <- config file <- mode (from the command line). Relative paths in
// the llama section resolve against llama.dir, the Python exe against python/.
function resolveConfig({ fileConfig = {}, mode = null, configDir = REPO_ROOT } = {}) {
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
    python: { ...cfg.python, dir: pythonDir, exe: expandPath(cfg.python.exe, pythonDir) },
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
  return { name: 'backend', label: 'Python backend', exe: py.exe, args, cwd: py.dir };
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

// One child process: spawn, wait until `isReady()` passes, keep the last log
// lines for the panel, stop the whole process tree on quit.
class Service extends EventEmitter {
  constructor({ name, label, exe, args, cwd, env, isReady, readyTimeoutS, logDir }) {
    super();
    Object.assign(this, { name, label, exe, args, cwd, env, isReady, readyTimeoutS, logDir });
    this.state = 'stopped';
    this.detail = '';
    this.child = null;
    this.external = false;
    this.tail = [];
  }

  info() {
    return { name: this.name, label: this.label, state: this.state, detail: this.detail };
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
      this._set('failed', `not found: ${this.exe}`);
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
      if (!this.external) this._set('stopped');
      return;
    }
    child.stopRequested = true;
    const exited = new Promise((r) => child.once('exit', r));
    if (IS_WIN) {
      // Kill the whole tree; plain kill() would leave grandchildren behind.
      spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
    } else {
      child.kill('SIGTERM');
    }
    const timer = setTimeout(() => child.kill('SIGKILL'), 3000);
    await exited;
    clearTimeout(timer);
    if (this.child === child) this.child = null;
    this._set('stopped');
  }

  async restart() {
    await this.stop();
    await this.start();
  }
}

function createServices(cfg, { logDir = path.join(REPO_ROOT, 'logs') } = {}) {
  const services = [];
  if (cfg.llama.enabled) {
    const port = cfg.llama.port;
    services.push(
      new Service({
        ...llamaCommand(cfg.llama),
        isReady: () => httpOk(`http://127.0.0.1:${port}/health`),
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
  portOpen,
  pythonCommand,
  resolveConfig,
};
