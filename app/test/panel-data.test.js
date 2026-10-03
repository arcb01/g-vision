'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { ConversationLog } = require('../conversation');
const { check, resolveSettings, saveSettings, settingsArgs } = require('../settings');
const { pythonCommand, resolveConfig } = require('../services');

const tmp = () => fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-'));
const JPEG = `data:image/jpeg;base64,${Buffer.from('fake jpeg').toString('base64')}`;

function exchange(n, fields = {}) {
  return {
    v: 1, ts: 1000 + n + 1, type: 'exchange', exchange_id: `a${n}`, asked_ts: 1000 + n, question: `q${n}`,
    answer: `a${n}`, via: 'voice', tools: ['set_watch'], latency_ms: { llm_tool_call: 300 }, screenshot: JPEG, ...fields,
  };
}

test('the conversation log keeps exchanges and screenshots across restarts', () => {
  const dir = tmp();
  const log = new ConversationLog(dir);
  const entry = log.add(exchange(1));
  assert.strictEqual(entry.question, 'q1');
  assert.ok(entry.imageUrl.startsWith('file://'));
  assert.strictEqual(fs.readFileSync(path.join(dir, entry.image), 'utf8'), 'fake jpeg');
  log.add(exchange(2, { screenshot: null, via: 'typed' }));

  const reopened = new ConversationLog(dir).list();
  assert.deepStrictEqual(reopened.map((e) => [e.question, e.via, e.image != null]), [['q1', 'voice', true], ['q2', 'typed', false]]);
});

test('the conversation log drops the oldest exchanges and their screenshots past the cap', () => {
  const dir = tmp();
  const log = new ConversationLog(dir, { max: 3 });
  const first = log.add(exchange(1));
  for (let n = 2; n <= 5; n++) log.add(exchange(n));
  assert.deepStrictEqual(new ConversationLog(dir).list().map((e) => e.question), ['q3', 'q4', 'q5']);
  assert.ok(!fs.existsSync(path.join(dir, first.image)));
  assert.strictEqual(fs.readdirSync(path.join(dir, 'shots')).length, 3);
  log.clear();
  assert.deepStrictEqual(new ConversationLog(dir).list(), []);
});

test('a cut-off line in the log is skipped', () => {
  const dir = tmp();
  new ConversationLog(dir).add(exchange(1));
  fs.appendFileSync(path.join(dir, 'exchanges.jsonl'), '{"id": "broken');
  assert.strictEqual(new ConversationLog(dir).list().length, 1);
});

test('settings are checked before they are saved', () => {
  assert.strictEqual(check('pttKey', 'Alt + 3'), 'alt+3');
  assert.ok(check('pttKey', 'alt') instanceof Error);
  assert.ok(check('pttKey', 'win+3') instanceof Error);
  assert.ok(check('asr', 'siri') instanceof Error);
  assert.strictEqual(check('dimStrength', '0.3'), 0.3);
  assert.ok(check('dimStrength', 0.95) instanceof Error);
  assert.ok(check('speak', 'yes') instanceof Error);
  assert.ok(check('nope', 1) instanceof Error);
  // A bad value in the file falls back to the default.
  assert.strictEqual(resolveSettings({ asr: 'siri', voice: 'bf_emma' }).asr, 'whisper');
  assert.strictEqual(resolveSettings({ asr: 'siri', voice: 'bf_emma' }).voice, 'bf_emma');
});

test('saving settings keeps the rest of gvision.config.json', () => {
  const file = path.join(tmp(), 'gvision.config.json');
  fs.writeFileSync(file, JSON.stringify({ llama: { port: 9000 }, settings: { voice: 'bf_emma' } }));
  const values = saveSettings(file, { speak: false, pttKey: 'F8' });
  assert.strictEqual(values.pttKey, 'f8');
  const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
  assert.deepStrictEqual(saved, { llama: { port: 9000 }, settings: { voice: 'bf_emma', speak: false, pttKey: 'f8' } });
  assert.throws(() => saveSettings(file, { asr: 'siri' }), /Speech-to-text/);
  assert.deepStrictEqual(JSON.parse(fs.readFileSync(file, 'utf8')), saved);
});

test('settings become backend flags, but hand-written args win', () => {
  const settings = resolveSettings({ speak: false, readText: false, asr: 'nemotron', narrateEvery: 0 });
  const args = settingsArgs(settings, ['--live', '--agent', '--voice', 'am_adam']);
  assert.ok(!args.includes('--voice'));
  assert.deepStrictEqual(args.slice(args.indexOf('--asr'), args.indexOf('--asr') + 2), ['--asr', 'nemotron']);
  assert.ok(args.includes('--no-tts') && args.includes('--no-text') && !args.includes('--no-memory'));
  assert.deepStrictEqual(args.slice(args.indexOf('--narrate-every'), args.indexOf('--narrate-every') + 2), ['--narrate-every', '0']);
});

test('the backend command carries the settings in agent mode only', () => {
  const cfg = resolveConfig({ fileConfig: { settings: { pttKey: 'f8' } } });
  const agent = pythonCommand(cfg.python, cfg.llama.port, cfg.settings).args;
  assert.deepStrictEqual(agent.slice(agent.indexOf('--ptt-key'), agent.indexOf('--ptt-key') + 2), ['--ptt-key', 'f8']);
  assert.deepStrictEqual(agent.slice(-4, -2), ['--port', '8765']);
  const demo = resolveConfig({ mode: 'demo' });
  assert.deepStrictEqual(pythonCommand(demo.python, demo.llama.port, demo.settings).args, ['-m', 'gvision', '--demo', '--port', '8765']);
});
