'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { VISION_MODELS, download, missingFiles, modelPaths } = require('../models');
const { SPEC } = require('../settings');
const { resolveConfig, serviceCommands } = require('../services');

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'gvision-models-'));
}

function fakeFetch(seen) {
  return async (url) => {
    seen.push(url);
    const chunks = [Buffer.from('gguf'), Buffer.from('-data')];
    return { ok: true, status: 200, body: (async function* gen() { yield* chunks; })() };
  };
}

test('every vision model is a choice in Settings with a summary and both files', () => {
  const spec = SPEC.find((s) => s.key === 'visionModel');
  assert.deepStrictEqual(spec.choices.map(([id]) => id), VISION_MODELS.map((m) => m.id));
  for (const m of VISION_MODELS) {
    assert.ok(spec.details[m.id]);
    if (m.id !== 'same') assert.strictEqual(m.files.length, 2);
  }
  assert.strictEqual(spec.default, 'same');
});

test('a picked model downloads its missing files from Hugging Face', async () => {
  const dir = tmp();
  const seen = [];
  const progress = [];
  assert.strictEqual(missingFiles('qwen3.5-4b-ud-q8kxl', dir).length, 2);
  await download('qwen3.5-4b-ud-q8kxl', dir, { fetchFn: fakeFetch(seen), onProgress: (p) => progress.push(p) });
  assert.deepStrictEqual(seen, [
    'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q8_K_XL.gguf',
    'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf',
  ]);
  const paths = modelPaths('qwen3.5-4b-ud-q8kxl', dir);
  assert.strictEqual(fs.readFileSync(paths.model, 'utf8'), 'gguf-data');
  assert.ok(fs.existsSync(paths.mmproj));
  assert.strictEqual(progress.at(-1).file, 'mmproj-F16.gguf');
  assert.deepStrictEqual(missingFiles('qwen3.5-4b-ud-q8kxl', dir), []);
  // Nothing left to fetch the second time.
  await download('qwen3.5-4b-ud-q8kxl', dir, { fetchFn: fakeFetch(seen) });
  assert.strictEqual(seen.length, 2);
});

test('a failed download leaves no file behind', async () => {
  const dir = tmp();
  const failing = async () => ({ ok: true, status: 200, body: (async function* gen() { yield Buffer.from('x'); throw new Error('reset'); })() });
  await assert.rejects(download('qwen3.5-9b-q3km', dir, { fetchFn: failing }), /reset/);
  assert.strictEqual(missingFiles('qwen3.5-9b-q3km', dir).length, 2);
  assert.deepStrictEqual(fs.readdirSync(path.dirname(modelPaths('qwen3.5-9b-q3km', dir).model)), []);
  await assert.rejects(download('qwen3.5-9b-q3km', dir, { fetchFn: async () => ({ ok: false, status: 404 }) }), /HTTP 404/);
});

test('a downloaded vision model runs as a second llama-server the backend looks through', async () => {
  const dir = tmp();
  const fileConfig = { llama: { dir, port: 9000 }, settings: { visionModel: 'qwen3.5-4b-ud-q8kxl' } };
  // Not downloaded yet: the main Qwen keeps looking.
  assert.strictEqual(resolveConfig({ fileConfig }).vision, null);
  await download('qwen3.5-4b-ud-q8kxl', dir, { fetchFn: fakeFetch([]) });
  const cfg = resolveConfig({ fileConfig });
  assert.strictEqual(cfg.vision.port, 9001);
  const cmds = serviceCommands(cfg);
  assert.strictEqual(cmds.vision.name, 'vision');
  assert.deepStrictEqual(cmds.vision.args.slice(0, 4), ['-m', modelPaths('qwen3.5-4b-ud-q8kxl', dir).model, '--mmproj', modelPaths('qwen3.5-4b-ud-q8kxl', dir).mmproj]);
  assert.ok(cmds.vision.args.join(' ').includes('--port 9001'));
  const backend = cmds.backend.args.join(' ');
  assert.ok(backend.includes('--qwen-url http://127.0.0.1:9000'));
  assert.ok(backend.includes('--vision-url http://127.0.0.1:9001'));
  assert.ok(backend.includes('--vision-model Qwen3.5-4B-UD-Q8_K_XL'));
  assert.ok(!backend.includes('visionModel'));
  // "Same as the main Qwen" runs nothing extra.
  const same = resolveConfig({ fileConfig: { ...fileConfig, settings: { visionModel: 'same' } } });
  assert.strictEqual(same.vision, null);
  assert.ok(!('vision' in serviceCommands(same)));
  assert.ok(!serviceCommands(same).backend.args.includes('--vision-url'));
  const sameArgs = serviceCommands(same).backend.args;
  assert.strictEqual(sameArgs[sameArgs.indexOf('--vision-model') + 1], path.basename(same.llama.model, '.gguf'));
});

test('the vision-only toggle becomes --vision-only only when on', () => {
  const { resolveSettings, settingsArgs } = require('../settings');
  assert.ok(!settingsArgs(resolveSettings()).includes('--vision-only'));
  assert.ok(settingsArgs(resolveSettings({ visionOnly: true })).includes('--vision-only'));
});

test('vision reasoning lets only the vision server think and tells the backend', async () => {
  const dir = tmp();
  await download('qwen3.5-4b-ud-q8kxl', dir, { fetchFn: fakeFetch([]) });
  const settings = { visionModel: 'qwen3.5-4b-ud-q8kxl', visionReasoning: false };
  const off = serviceCommands(resolveConfig({ fileConfig: { llama: { dir }, settings } }));
  assert.ok(off.vision.args.join(' ').includes('--reasoning off'));
  assert.ok(!off.backend.args.includes('--vision-reasoning'));
  const on = serviceCommands(resolveConfig({ fileConfig: { llama: { dir }, settings: { visionModel: settings.visionModel } } }));  // on by default
  assert.ok(!on.vision.args.join(' ').includes('--reasoning off'));
  assert.ok(on.qwen.args.join(' ').includes('--reasoning off'));
  assert.ok(on.backend.args.includes('--vision-reasoning'));
});

test('image resolution becomes --look-size', () => {
  const { resolveSettings, settingsArgs } = require('../settings');
  const args = settingsArgs(resolveSettings());
  assert.strictEqual(args[args.indexOf('--look-size') + 1], '1600');
  const full = settingsArgs(resolveSettings({ lookSize: 'full' }));
  assert.strictEqual(full[full.indexOf('--look-size') + 1], 'full');
});
