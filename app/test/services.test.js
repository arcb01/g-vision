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
