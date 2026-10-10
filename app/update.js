// Brings the checkout up to date before the app starts: git pull, then
// reinstall the app's npm packages or the Python package only when their
// dependency files changed. Run by G-VISION.bat on every launch and by the
// panel's Update button. Never blocks the launch: if anything fails (offline,
// local changes, no git) it says why and the app starts with what is there.
//
//   node app/update.js
'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { REPO_ROOT, loadConfig } = require('./services');

const IS_WIN = process.platform === 'win32';
const DEFAULT_EXTRAS = 'dev,perception,voice';

// Pythons to build python/.venv from on a fresh install, best first.
// G-VISION.bat (app/prereqs.cmd) makes sure one of them is 3.11 or newer.
const BASE_PYTHONS = IS_WIN
  ? [['py', ['-3.12']], ['py', ['-3']], ['python', []]]
  : [['python3', []], ['python', []]];
const PYTHON_OK = 'import sys; sys.exit(sys.version_info < (3, 11))';
const TORCH_INDEX = 'https://download.pytorch.org/whl/cu128';

// onnxruntime and onnxruntime-gpu install the same module, and whichever
// pip wrote last wins. Kokoro needs the GPU build, so after an install put
// it back on top if the CPU build replaced it.
const HAS_CUDA = "import onnxruntime as o, sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)";

function ensureGpuOnnxruntime(python, run, say) {
  if (run(python, ['-c', HAS_CUDA], null).ok) return;
  say('Putting the GPU build of onnxruntime back on top (for the voice)...');
  const r = run(python, ['-m', 'pip', 'install', '--force-reinstall', '--no-deps', 'onnxruntime-gpu'], null, { live: true });
  if (!r.ok) say('Could not reinstall onnxruntime-gpu; the voice will run on the CPU.');
}

// Creates python/.venv when there is no Python environment yet (a fresh
// install), with the CUDA build of PyTorch on Windows: installed after the
// package, pip would pull the CPU one. Returns the venv's python, or null
// after removing a half-made venv so the next launch tries again.
function createVenv({ repoRoot = REPO_ROOT, run = defaultRun, log = console.log, gpu = IS_WIN } = {}) {
  const base = BASE_PYTHONS.find(([cmd, args]) => run(cmd, [...args, '-c', PYTHON_OK], null).ok);
  if (!base) {
    log('Could not create the Python environment: no Python 3.11 or newer found.');
    return null;
  }
  const venv = path.join(repoRoot, 'python', '.venv');
  const python = path.join(venv, IS_WIN ? 'Scripts/python.exe' : 'bin/python');
  const fail = (why) => {
    log(`Could not create the Python environment (will retry next launch): ${why}`);
    fs.rmSync(venv, { recursive: true, force: true });
    return null;
  };
  log(`Creating the Python environment in ${venv}, first run only...`);
  const [cmd, args] = base;
  if (!run(cmd, [...args, '-m', 'venv', venv], null, { live: true }).ok) return fail('python -m venv failed');
  run(python, ['-m', 'pip', 'install', '--upgrade', 'pip'], null, { live: true });
  if (gpu) {
    log('Installing the CUDA build of PyTorch (a few GB)...');
    const r = run(python, ['-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', TORCH_INDEX], null, { live: true });
    if (!r.ok) return fail('could not install PyTorch, see the output above');
  }
  return python;
}

function fileHash(file) {
  try {
    return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  } catch {
    return null;
  }
}

function readState(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return {};
  }
}

// `live` shows the command's own progress in the launcher window (installs).
function defaultRun(cmd, args, cwd, { live = false } = {}) {
  const r = spawnSync(cmd, args, {
    cwd,
    encoding: 'utf8',
    shell: IS_WIN && cmd === 'npm',
    windowsHide: true,
    stdio: live ? 'inherit' : 'pipe',
  });
  return { ok: !r.error && r.status === 0, out: `${r.stdout || ''}${r.stderr || ''}`.trim(), error: r.error };
}

// Returns { pulled, updated, installed: [...], messages: [...] }.
// `freshPython` installs the Python package whatever the saved state says
// (a venv createVenv just made is empty).
function runUpdate({
  repoRoot = REPO_ROOT, run = defaultRun, log = console.log, python = null, extras = DEFAULT_EXTRAS, gpu = IS_WIN,
  freshPython = false,
} = {}) {
  const result = { pulled: false, updated: false, installed: [], messages: [] };
  const say = (m) => {
    result.messages.push(m);
    log(m);
  };

  const head = () => {
    const r = run('git', ['rev-parse', 'HEAD'], repoRoot);
    return r.ok ? r.out : null;
  };
  const before = head();
  if (!before) {
    say('Update skipped: git is not available or this is not a git checkout.');
  } else {
    say('Checking for updates...');
    const pull = run('git', ['pull', '--ff-only'], repoRoot);
    if (!pull.ok) {
      say(`Update skipped, starting the current version. git said: ${pull.out.split('\n').slice(-3).join(' ')}`);
    } else {
      result.pulled = true;
      const after = head();
      result.updated = after !== before;
      say(result.updated ? `Updated ${before.slice(0, 7)} -> ${after.slice(0, 7)}.` : 'Already up to date.');
    }
  }

  // Reinstall only what changed since the last successful install.
  const stateFile = path.join(repoRoot, 'logs', 'install-state.json');
  const state = readState(stateFile);
  if (freshPython) delete state.python;
  const deps = [
    {
      key: 'npm',
      file: path.join(repoRoot, 'app', 'package-lock.json'),
      label: "the app's packages",
      install: () => run('npm', ['install', '--no-audit', '--no-fund'], path.join(repoRoot, 'app'), { live: true }),
    },
    {
      key: 'python',
      file: path.join(repoRoot, 'python', 'pyproject.toml'),
      label: 'the Python package',
      install: () => (python
        ? run(python, ['-m', 'pip', 'install', '-e', `.[${extras}]`], path.join(repoRoot, 'python'), { live: true })
        : { ok: false, out: 'no Python environment found' }),
    },
  ];
  for (const dep of deps) {
    const hash = fileHash(dep.file);
    if (!hash || state[dep.key] === hash) continue;
    if (state[dep.key] === undefined && dep.key === 'npm' && !result.updated) {
      // First run of the updater: node_modules came from the launcher's own
      // npm install. The Python package is installed once anyway, since
      // extras added since the last manual install may be missing.
      state[dep.key] = hash;
      continue;
    }
    say(`Installing ${dep.label} (dependencies changed)...`);
    const r = dep.install();
    if (r.ok) {
      state[dep.key] = hash;
      result.installed.push(dep.key);
    } else {
      const why = r.out || (r.error ? r.error.message : 'see the output above');
      say(`Could not install ${dep.label} (will retry next launch): ${why.split('\n').slice(-3).join(' ')}`);
    }
  }
  if (gpu && python && result.installed.includes('python')) ensureGpuOnnxruntime(python, run, say);
  fs.mkdirSync(path.dirname(stateFile), { recursive: true });
  fs.writeFileSync(stateFile, JSON.stringify(state, null, 2));
  return result;
}

// What the Update button runs with `cmd.exe /c` on Windows. `start` on a .bat
// opens it with `cmd /k`, which leaves the window open after the launcher
// exits, one more per update; `cmd /c` closes it. The doubled outer quotes
// are what cmd /c strips, keeping the quotes around a path with spaces.
function relaunchCommand(launcher, extra = '') {
  return `start "G-VISION update" cmd /c ""${launcher}"${extra}"`;
}

if (require.main === module) {
  let python = null;
  let freshPython = false;
  let noPython = false;
  try {
    const cfg = loadConfig();
    if (!cfg.python.notFound) python = cfg.python.exe;
    noPython = Boolean(cfg.python.notFound);
  } catch (err) {
    console.log(`Ignoring gvision.config.json for the update: ${err.message}`);
  }
  try {
    if (noPython) {
      python = createVenv();
      freshPython = python !== null;
    }
    const r = runUpdate({ python, freshPython });
    // Exit code 2 tells the launcher to leave the window open a moment so a
    // failed install can be read.
    if (r.messages.some((m) => m.startsWith('Could not install'))) process.exitCode = 2;
  } catch (err) {
    console.log(`Update failed, starting anyway: ${err.message}`);
  }
}

module.exports = { createVenv, fileHash, relaunchCommand, runUpdate };
