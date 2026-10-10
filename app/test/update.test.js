'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createVenv, runUpdate } = require('../update');

const git = (cwd, ...args) => {
  const r = spawnSync('git', args, { cwd, encoding: 'utf8' });
  assert.strictEqual(r.status, 0, r.stderr);
  return r.stdout.trim();
};

function write(root, file, text) {
  fs.mkdirSync(path.dirname(path.join(root, file)), { recursive: true });
  fs.writeFileSync(path.join(root, file), text);
}

// An "origin" checkout and a clone of it, like Arnau's PC and GitHub.
function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-update-'));
  const origin = path.join(dir, 'origin');
  fs.mkdirSync(origin);
  git(origin, 'init', '-q', '-b', 'main');
  git(origin, 'config', 'user.email', 't@t');
  git(origin, 'config', 'user.name', 't');
  write(origin, 'app/package-lock.json', '{"v":1}');
  write(origin, 'python/pyproject.toml', 'v = 1');
  write(origin, 'README.md', 'one');
  git(origin, 'add', '-A');
  git(origin, 'commit', '-qm', 'one');
  const local = path.join(dir, 'local');
  git(dir, 'clone', '-q', origin, local);
  return { origin, local };
}

function commit(origin, file, text) {
  write(origin, file, text);
  git(origin, 'commit', '-qam', `change ${file}`);
}

// Real git, fake npm and pip.
function recorder() {
  const calls = [];
  const run = (cmd, args, cwd) => {
    if (cmd === 'git') {
      const r = spawnSync(cmd, args, { cwd, encoding: 'utf8' });
      return { ok: r.status === 0, out: `${r.stdout}${r.stderr}`.trim() };
    }
    calls.push(cmd === 'npm' ? 'npm' : 'pip');
    return { ok: true, out: '' };
  };
  return { calls, run };
}

const quiet = () => {};

test('first run installs the Python package once and leaves npm alone', () => {
  const { local } = setup();
  const { calls, run } = recorder();
  const r = runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.strictEqual(r.pulled, true);
  assert.strictEqual(r.updated, false);
  assert.deepStrictEqual(calls, ['pip']);
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.deepStrictEqual(calls, ['pip']);
});

test('pulls new commits and reinstalls only what changed', () => {
  const { origin, local } = setup();
  const { calls, run } = recorder();
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  calls.length = 0;

  commit(origin, 'README.md', 'two');
  let r = runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.strictEqual(r.updated, true);
  assert.strictEqual(fs.readFileSync(path.join(local, 'README.md'), 'utf8'), 'two');
  assert.deepStrictEqual(calls, []);

  commit(origin, 'python/pyproject.toml', 'v = 2');
  r = runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.deepStrictEqual(r.installed, ['python']);
  assert.deepStrictEqual(calls, ['pip']);

  commit(origin, 'app/package-lock.json', '{"v":2}');
  r = runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.deepStrictEqual(r.installed, ['npm']);
  assert.deepStrictEqual(calls, ['pip', 'npm']);
});

test('a failed install is retried on the next launch', () => {
  const { origin, local } = setup();
  runUpdate({ repoRoot: local, run: recorder().run, log: quiet, python: 'python' });
  commit(origin, 'python/pyproject.toml', 'v = 2');
  const failing = (cmd, args, cwd) => (cmd === 'git' ? recorder().run(cmd, args, cwd) : { ok: false, out: 'no network' });
  const r = runUpdate({ repoRoot: local, run: failing, log: quiet, python: 'python' });
  assert.deepStrictEqual(r.installed, []);
  assert.ok(r.messages.some((m) => m.includes('no network')));
  const { calls, run } = recorder();
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  assert.deepStrictEqual(calls, ['pip']);
});

test('local edits that conflict with the update skip the pull without failing', () => {
  const { origin, local } = setup();
  commit(origin, 'README.md', 'upstream');
  write(local, 'README.md', 'my local edit');
  const r = runUpdate({ repoRoot: local, run: recorder().run, log: quiet, python: 'python' });
  assert.strictEqual(r.pulled, false);
  assert.ok(r.messages.some((m) => m.startsWith('Update skipped')));
  assert.strictEqual(fs.readFileSync(path.join(local, 'README.md'), 'utf8'), 'my local edit');
});

test('outside a git checkout it only reports that it skipped', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-nogit-'));
  const r = runUpdate({ repoRoot: dir, run: recorder().run, log: quiet });
  assert.strictEqual(r.pulled, false);
  assert.match(r.messages[0], /not a git checkout/);
});

test('after a Python install the GPU onnxruntime is put back if the CPU one replaced it', () => {
  const { local } = setup();
  const seen = [];
  let cuda = false;
  const run = (cmd, args, cwd) => {
    if (cmd === 'git') {
      const r = spawnSync(cmd, args, { cwd, encoding: 'utf8' });
      return { ok: r.status === 0, out: `${r.stdout}${r.stderr}`.trim() };
    }
    if (args[0] === '-c') return { ok: cuda, out: '' };
    seen.push(args.slice(2).join(' '));
    if (args.includes('onnxruntime-gpu')) cuda = true;
    return { ok: true, out: '' };
  };
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python', gpu: true });
  assert.deepStrictEqual(seen, ['install -e .[dev,perception,voice]', 'install --force-reinstall --no-deps onnxruntime-gpu']);
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python', gpu: true });
  assert.strictEqual(seen.length, 2); // nothing changed: no install, no check
});

test('a fresh venv gets the Python package even if an older install was recorded', () => {
  const { local } = setup();
  const { calls, run } = recorder();
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python' });
  runUpdate({ repoRoot: local, run, log: quiet, python: 'python', freshPython: true });
  assert.deepStrictEqual(calls, ['pip', 'pip']);
});

// Fake Pythons: `versions` maps a command to whether it is 3.11+.
function venvRun(versions, failOn = null) {
  const seen = [];
  const run = (cmd, args) => {
    if (args.includes('-c')) return { ok: Boolean(versions[cmd]), out: '' };
    seen.push(`${path.basename(cmd)} ${args.join(' ')}`);
    if (failOn && args.includes(failOn)) return { ok: false, out: '' };
    if (args.includes('venv')) fs.mkdirSync(args[args.length - 1], { recursive: true });
    return { ok: true, out: '' };
  };
  return { seen, run };
}

test('a fresh install gets python/.venv from the first usable Python, with CUDA torch on Windows', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-venv-'));
  const { seen, run } = venvRun({ python3: false, python: true, py: false });
  const python = createVenv({ repoRoot: dir, run, log: quiet, gpu: true });
  const venv = path.join(dir, 'python', '.venv');
  assert.ok(python.startsWith(venv));
  assert.strictEqual(seen[0], `python -m venv ${venv}`);
  assert.ok(seen.some((s) => s.includes('install torch torchvision --index-url https://download.pytorch.org/whl/cu128')));
});

test('no usable Python, or a failed torch download, leaves no venv behind', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-venv-'));
  assert.strictEqual(createVenv({ repoRoot: dir, run: venvRun({}).run, log: quiet }), null);
  const { run } = venvRun({ python3: true, python: true, py: true }, 'torch');
  assert.strictEqual(createVenv({ repoRoot: dir, run, log: quiet, gpu: true }), null);
  assert.ok(!fs.existsSync(path.join(dir, 'python', '.venv')));
});

test('the Update button relaunches through cmd /c so its window closes', () => {
  const { relaunchCommand } = require('../update');
  const launcher = 'C:\\Users\\arnau\\Desktop\\coding projects\\G-vision\\G-VISION.bat';
  assert.strictEqual(
    relaunchCommand(launcher, ' --demo'),
    'start "G-VISION update" cmd /c ""C:\\Users\\arnau\\Desktop\\coding projects\\G-vision\\G-VISION.bat" --demo"',
  );
  assert.ok(!relaunchCommand(launcher).includes('/k'));
});
