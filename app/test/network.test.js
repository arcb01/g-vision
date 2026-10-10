'use strict';

// Two-PC mode: what each role runs, the AI server's control server and the
// gaming PC's client, and the settings that come with them.

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createServices, resolveConfig, serviceCommands } = require('../services');
const { ClockSync, RemoteClient, createControlServer, friendlyError, gitBuild, localAddresses } = require('../remote');
const { SPEC, check, settingsArgs, resolveSettings } = require('../settings');
const { gameList } = require('../sessions');

const host = { settings: { serverHost: '192.168.1.20' } };

test('standalone is the default and runs everything here', () => {
  const cfg = resolveConfig();
  assert.strictEqual(cfg.role, 'standalone');
  assert.deepStrictEqual(Object.keys(serviceCommands(cfg)).sort(), ['backend', 'qwen']);
  assert.ok(!serviceCommands(cfg).backend.args.includes('--source'));
});

test('an AI server takes its screen and voice from the network', () => {
  const cfg = resolveConfig({ fileConfig: { settings: { role: 'server' } } });
  const { backend, qwen } = serviceCommands(cfg);
  const args = backend.args.join(' ');
  assert.match(args, /--source edge/);
  assert.match(args, /--host 0\.0\.0\.0/);
  // llama-server stays private to this PC: only the backend talks to it.
  assert.match(qwen.args.join(' '), /--host 127\.0\.0\.1/);
});

test('--server wins over Settings, and the demo is always standalone', () => {
  assert.strictEqual(resolveConfig({ role: 'server' }).role, 'server');
  assert.strictEqual(resolveConfig({ mode: 'demo', fileConfig: { settings: { role: 'server' } } }).role, 'standalone');
});

test('a gaming PC runs only the screen and voice link, pointed at the AI server', () => {
  const cfg = resolveConfig({ fileConfig: { settings: { role: 'gaming', serverHost: '192.168.1.20', pttKey: 'f8' } } });
  const cmds = serviceCommands(cfg);
  assert.deepStrictEqual(Object.keys(cmds), ['edge']);
  assert.deepStrictEqual(cmds.edge.args, ['-m', 'gvision', '--edge', 'ws://192.168.1.20:8765/edge', '--ptt-key', 'f8']);
  assert.deepStrictEqual(createServices(cfg).map((s) => s.name), ['edge']);
  // Without an address there is nothing to start yet.
  const none = resolveConfig({ fileConfig: { settings: { role: 'gaming' } } });
  assert.deepStrictEqual(serviceCommands(none), {});
  assert.deepStrictEqual(createServices(none), []);
});

test('the edge is ready when its log says it is streaming, and starting while it reconnects', () => {
  const cfg = resolveConfig({ fileConfig: { settings: { role: 'gaming', ...host.settings } } });
  const [edge] = createServices(cfg, { logDir: null });
  edge.state = 'starting';
  edge.onLine('edge: connected to ws://192.168.1.20:8765/edge, streaming the screen at 10 fps', edge);
  assert.strictEqual(edge.state, 'ready');
  edge.onLine("edge: can't reach ws://192.168.1.20:8765/edge (refused), retrying", edge);
  assert.strictEqual(edge.state, 'starting');
  assert.match(edge.detail, /can't reach the AI server at 192\.168\.1\.20/);
});

test('network settings: roles, addresses and which PC owns each setting', () => {
  assert.strictEqual(check('role', 'gaming'), 'gaming');
  assert.ok(check('role', 'cloud') instanceof Error);
  assert.strictEqual(check('serverHost', ' 192.168.1.20 '), '192.168.1.20');
  assert.strictEqual(check('serverHost', 'gaming-rig'), 'gaming-rig');
  assert.strictEqual(check('serverHost', ''), '');
  assert.ok(check('serverHost', 'http://192.168.1.20') instanceof Error);
  assert.ok(check('serverHost', '192.168.1.20:8765') instanceof Error);
  const side = Object.fromEntries(SPEC.map((s) => [s.key, s.side]));
  assert.strictEqual(side.visionModel, 'server');
  assert.strictEqual(side.whisperModel, 'server');
  assert.strictEqual(side.pttKey, 'gaming');
  assert.strictEqual(side.role, 'local');
  assert.ok(SPEC.every((s) => s.side), 'every setting says which PC it belongs to');
  // Network settings never become backend flags.
  const args = settingsArgs(resolveSettings({ role: 'server', serverHost: '10.0.0.2' }));
  assert.ok(!args.some((a) => a.includes('10.0.0.2') || a === 'server'));
});

test('the session picker can use the AI server\'s wiki indexes', () => {
  const known = [{ id: 'minecraft', name: 'Minecraft', wiki: 'https://minecraft.wiki' }];
  const games = gameList(known, [
    { game: 'minecraft', pages: 9000, updated_ts: 5, source: 'Minecraft Wiki' },
    { game: 'hades', name: 'Hades', wiki: 'https://hades.fandom.com', pages: 300 },
  ]);
  assert.strictEqual(games[0].pages, 9000);
  assert.deepStrictEqual(games.map((g) => g.id), ['minecraft', 'hades']);
  assert.strictEqual(games[1].custom, true);
});

test('control server and client: routes, params, errors and the caller\'s address', async () => {
  const seen = [];
  const server = createControlServer({
    'GET /info': () => ({ app: 'g-vision', time: 1000 }),
    'POST /settings': ({ body }) => ({ ok: true, got: body.changes }),
    'POST /services/:name/:action': ({ params }) => ({ ok: true, params }),
    'GET /boom': () => {
      throw new Error('no such setting');
    },
  }, { onRequest: (peer) => seen.push(peer) });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const client = new RemoteClient(`http://127.0.0.1:${server.address().port}`);
  try {
    assert.deepStrictEqual(await client.get('/info'), { app: 'g-vision', time: 1000 });
    assert.deepStrictEqual((await client.post('/settings', { changes: { voice: 'bf_emma' } })).got, { voice: 'bf_emma' });
    assert.deepStrictEqual((await client.post('/services/backend/restart')).params, { name: 'backend', action: 'restart' });
    await assert.rejects(client.get('/boom'), /no such setting/);
    await assert.rejects(client.get('/nope'), /not found/);
    assert.strictEqual(seen[0], '127.0.0.1');
  } finally {
    server.close();
  }
});

test('a PC with nothing listening reads as "not running as an AI server"', async () => {
  const server = createControlServer({});
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  await new Promise((r) => server.close(r));
  const client = new RemoteClient(`http://127.0.0.1:${port}`);
  await assert.rejects(client.get('/info'), /isn't listening/);
});

test('friendly errors name the likely cause', () => {
  const err = (code) => Object.assign(new Error('fetch failed'), { cause: { code } });
  assert.match(friendlyError(err('ECONNREFUSED'), 'http://10.0.0.2:8770'), /AI server/);
  assert.match(friendlyError(err('ENOTFOUND'), 'http://rig:8770'), /Can't find a PC called rig/);
  const timeout = Object.assign(new Error('timed out'), { name: 'TimeoutError' });
  assert.match(friendlyError(timeout, 'http://10.0.0.2:8770'), /Windows Firewall/);
});

test('clock sync keeps the offset from the quickest round trip', () => {
  const clock = new ClockSync();
  assert.strictEqual(clock.offset, 0);
  // The AI server's clock is 2 s behind this PC's.
  clock.add(10000, 10040, 8.03); // slow trip, skewed estimate
  clock.add(20000, 20004, 18.002);
  assert.ok(Math.abs(clock.offset - 2) < 1e-9);
});

test('local addresses skip loopback and link-local ones', () => {
  const list = localAddresses({
    lo: [{ family: 'IPv4', internal: true, address: '127.0.0.1' }],
    eth0: [{ family: 'IPv4', internal: false, address: '192.168.1.20' }, { family: 'IPv6', internal: false, address: 'fe80::1' }],
    wifi: [{ family: 'IPv4', internal: false, address: '169.254.3.4' }],
  });
  assert.deepStrictEqual(list, [{ name: 'eth0', address: '192.168.1.20' }]);
});

test('the build is the checked-out commit, loose or packed', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gv-git-'));
  const git = path.join(dir, '.git');
  fs.mkdirSync(path.join(git, 'refs', 'heads'), { recursive: true });
  fs.writeFileSync(path.join(git, 'HEAD'), 'ref: refs/heads/main\n');
  fs.writeFileSync(path.join(git, 'packed-refs'), '# pack-refs\n0123456789abcdef0123 refs/heads/main\n');
  assert.strictEqual(gitBuild(dir), '0123456789ab');
  fs.writeFileSync(path.join(git, 'refs', 'heads', 'main'), 'fedcba9876543210\n');
  assert.strictEqual(gitBuild(dir), 'fedcba987654');
  assert.strictEqual(gitBuild(path.join(dir, 'nope')), null);
});
