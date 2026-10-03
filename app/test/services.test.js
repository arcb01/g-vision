'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Service, llamaCommand, portOpen, pythonCommand, resolveConfig } = require('../services');

test('config file overrides defaults and relative paths resolve against llama.dir', () => {
  const cfg = resolveConfig({ fileConfig: { llama: { dir: '/opt/lab', port: 9000 } }, configDir: '/repo' });
  assert.strictEqual(cfg.llama.port, 9000);
  assert.strictEqual(cfg.llama.model, path.resolve('/opt/lab', 'models/Qwen3.5-2B-Q4_K_M.gguf'));
  assert.deepStrictEqual(cfg.python.args, ['--live', '--agent']);
});

test('~ expands to the home folder', () => {
  const cfg = resolveConfig({ fileConfig: { llama: { dir: '~/lab' } } });
  assert.strictEqual(cfg.llama.dir, path.join(os.homedir(), 'lab'));
});

test('demo mode turns off Qwen and runs the synthetic demo', () => {
  const cfg = resolveConfig({ mode: 'demo' });
  assert.strictEqual(cfg.llama.enabled, false);
  assert.deepStrictEqual(pythonCommand(cfg.python, cfg.llama.port).args, ['-m', 'gvision', '--demo', '--port', '8765']);
  assert.throws(() => resolveConfig({ mode: 'nope' }));
});

test('commands point the backend at the llama-server port', () => {
  const cfg = resolveConfig({ fileConfig: { llama: { port: 9000 } } });
  const py = pythonCommand(cfg.python, cfg.llama.port);
  assert.deepStrictEqual(py.args.slice(-2), ['--qwen-url', 'http://127.0.0.1:9000']);
  const llama = llamaCommand(cfg.llama);
  assert.ok(llama.args.includes('--mmproj'));
  assert.deepStrictEqual(llama.args.slice(-2), ['--port', '9000']);
});

test('a service starts, becomes ready, and stops', async () => {
  const port = 39000 + Math.floor(Math.random() * 1000);
  const script = `require('node:net').createServer().listen(${port}, '127.0.0.1'); console.log('up');`;
  const logDir = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-'));
  const svc = new Service({
    name: 'fake', label: 'Fake', exe: process.execPath, args: ['-e', script], cwd: logDir,
    isReady: () => portOpen(port), readyTimeoutS: 10, logDir,
  });
  await svc.start();
  assert.strictEqual(svc.state, 'ready');
  await svc.stop();
  assert.strictEqual(svc.state, 'stopped');
  assert.strictEqual(await portOpen(port), false);
  assert.match(fs.readFileSync(path.join(logDir, 'fake.log'), 'utf8'), /up/);
});

test('a missing executable or a crash is reported as failed', async () => {
  const missing = new Service({ name: 'm', exe: '/no/such/exe', args: [], isReady: async () => false, readyTimeoutS: 1 });
  await missing.start();
  assert.strictEqual(missing.state, 'failed');
  assert.match(missing.detail, /not found/);

  const crash = new Service({
    name: 'c', exe: process.execPath, args: ['-e', "console.error('boom'); process.exit(3)"],
    isReady: async () => false, readyTimeoutS: 10,
  });
  await crash.start();
  assert.strictEqual(crash.state, 'failed');
  assert.match(crash.detail, /exited \(3\): boom/);
});

test('the Python environment is found in python/.venv, the repo root or $VIRTUAL_ENV', () => {
  const venvPython = process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python';
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-root-'));
  const pythonDir = path.join(root, 'python');
  const make = (dir) => {
    const exe = path.join(dir, venvPython);
    fs.mkdirSync(path.dirname(exe), { recursive: true });
    fs.writeFileSync(exe, '');
    return exe;
  };
  const { pythonCandidates } = require('../services');
  const candidates = pythonCandidates(pythonDir, { VIRTUAL_ENV: path.join(root, 'active') });
  assert.deepStrictEqual(candidates, [
    path.join(root, 'active', venvPython),
    path.join(pythonDir, '.venv', venvPython),
    path.join(root, '.venv', venvPython),
  ]);
  const rootVenv = make(path.join(root, '.venv'));
  assert.ok(candidates.filter((c) => fs.existsSync(c)).includes(rootVenv));
});

test('a missing Python environment says where it looked', async () => {
  const cfg = resolveConfig({ env: { VIRTUAL_ENV: '/no/such/venv' } });
  // The repo itself may have a venv; only check the message when none exists.
  if (cfg.python.notFound) {
    assert.match(cfg.python.notFound, /\/no\/such\/venv/);
    const svc = new Service({ ...pythonCommand(cfg.python, 8080), isReady: async () => false, readyTimeoutS: 1 });
    await svc.start();
    assert.match(svc.detail, /no Python environment found/);
  } else {
    assert.ok(fs.existsSync(cfg.python.exe));
  }
});

test('netstat output gives the PIDs listening on a port', () => {
  const { parseNetstat } = require('../services');
  const out = [
    '  Proto  Local Address          Foreign Address        State           PID',
    '  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1100',
    '  TCP    127.0.0.1:8080         0.0.0.0:0              LISTENING       4242',
    '  TCP    127.0.0.1:8080         127.0.0.1:50123        ESTABLISHED     4242',
    '  TCP    127.0.0.1:50123        127.0.0.1:8080         ESTABLISHED     7777',
    '  TCP    127.0.0.1:18080        0.0.0.0:0              LISTENING       5555',
    '  TCP    [::1]:8765             [::]:0                 ESCUCHANDO      6060',
  ].join('\r\n');
  assert.deepStrictEqual(parseNetstat(out, 8080), [4242]);
  assert.deepStrictEqual(parseNetstat(out, 8765), [6060]);
  assert.deepStrictEqual(parseNetstat(out, 9999), []);
});

function fakeServer(port) {
  return `require('node:net').createServer().listen(${port}, '127.0.0.1'); console.log('up');`;
}

function fakeService(port, extra = {}) {
  return new Service({
    name: 'fake', label: 'Fake', exe: process.execPath, args: ['-e', fakeServer(port)], cwd: os.tmpdir(),
    isReady: () => portOpen(port), readyTimeoutS: 10, port, ...extra,
  });
}

test('restart replaces the process and stop can interrupt a start', async () => {
  const port = 41000 + Math.floor(Math.random() * 1000);
  const svc = fakeService(port);
  await svc.start();
  const first = svc.child.pid;
  await svc.restart();
  assert.strictEqual(svc.state, 'ready');
  assert.notStrictEqual(svc.child.pid, first);
  await svc.stop();
  assert.strictEqual(svc.state, 'stopped');

  const slow = fakeService(port, { args: ['-e', `setTimeout(() => {}, 60000)`] });
  const starting = slow.start();
  await new Promise((r) => setTimeout(r, 300));
  assert.strictEqual(slow.state, 'starting');
  await slow.stop();
  await starting;
  assert.strictEqual(slow.state, 'stopped');
});

test('stop and restart also work on a server found already running',
  { skip: process.platform !== 'win32' && !require('node:child_process').spawnSync('lsof', ['-v']).pid },
  async () => {
    const port = 42000 + Math.floor(Math.random() * 1000);
    const { spawn } = require('node:child_process');
    const orphan = spawn(process.execPath, ['-e', fakeServer(port)], { stdio: 'ignore' });
    const orphanExit = new Promise((r) => orphan.once('exit', r));
    while (!(await portOpen(port))) await new Promise((r) => setTimeout(r, 50));
    const svc = fakeService(port);
    await svc.start();
    assert.strictEqual(svc.detail, 'already running');
    await svc.restart();
    await orphanExit;
    assert.strictEqual(svc.state, 'ready');
    assert.ok(svc.child, 'restart should start its own process');
    await svc.stop();
    assert.strictEqual(await portOpen(port), false);
  });

test('update changes the command used by the next start', async () => {
  const svc = new Service({ name: 'u', exe: '/no/such/exe', args: [], isReady: async () => false, readyTimeoutS: 1 });
  await svc.start();
  assert.strictEqual(svc.state, 'failed');
  svc.update({ exe: '/still/missing', args: [], cwd: undefined, notFound: 'custom hint' });
  await svc.restart();
  assert.strictEqual(svc.detail, 'custom hint');
});

test('PID lists from PowerShell or lsof are parsed', () => {
  const { parsePidList } = require('../services');
  assert.deepStrictEqual(parsePidList('1234\r\n5678\r\n1234\r\n'), [1234, 5678]);
  assert.deepStrictEqual(parsePidList('\r\n'), []);
  assert.deepStrictEqual(parsePidList('Get-NetTCPConnection : error\r\n0\r\n'), []);
});

test('a process can be found by its command line',
  { skip: process.platform !== 'win32' && !require('node:child_process').spawnSync('pgrep', ['-V']).pid },
  async () => {
    const { pidsByCommandLine } = require('../services');
    const { spawn } = require('node:child_process');
    const token = `gvision-test-${process.pid}-${Date.now()}`;
    const child = spawn(process.execPath, ['-e', `setTimeout(() => {}, 30000) // ${token}`], { stdio: 'ignore' });
    await new Promise((r) => setTimeout(r, 200));
    assert.deepStrictEqual(pidsByCommandLine(token), [child.pid]);
    child.kill();
  });
