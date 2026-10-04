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
  assert.strictEqual(missingFiles('qwen3.5-4b-q4', dir).length, 2);
  await download('qwen3.5-4b-q4', dir, { fetchFn: fakeFetch(seen), onProgress: (p) => progress.push(p) });
  assert.deepStrictEqual(seen, [
    'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf',
    'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf',
  ]);
  const paths = modelPaths('qwen3.5-4b-q4', dir);
  assert.strictEqual(fs.readFileSync(paths.model, 'utf8'), 'gguf-data');
  assert.ok(fs.existsSync(paths.mmproj));
  assert.strictEqual(progress.at(-1).file, 'mmproj-F16.gguf');
  assert.deepStrictEqual(missingFiles('qwen3.5-4b-q4', dir), []);
  // Nothing left to fetch the second time.
  await download('qwen3.5-4b-q4', dir, { fetchFn: fakeFetch(seen) });
  assert.strictEqual(seen.length, 2);
});

test('a failed download leaves no file behind', async () => {
  const dir = tmp();
  const failing = async () => ({ ok: true, status: 200, body: (async function* gen() { yield Buffer.from('x'); throw new Error('reset'); })() });
  await assert.rejects(download('qwen3.5-2b-q8', dir, { fetchFn: failing }), /reset/);
  assert.strictEqual(missingFiles('qwen3.5-2b-q8', dir).length, 2);
  assert.deepStrictEqual(fs.readdirSync(path.dirname(modelPaths('qwen3.5-2b-q8', dir).model)), []);
  await assert.rejects(download('qwen3.5-2b-q8', dir, { fetchFn: async () => ({ ok: false, status: 404 }) }), /HTTP 404/);
});

test('a downloaded vision model runs as a second llama-server the backend looks through', async () => {
  const dir = tmp();
  const fileConfig = { llama: { dir, port: 9000 }, settings: { visionModel: 'qwen3.5-4b-q4' } };
  // Not downloaded yet: the main Qwen keeps looking.
  assert.strictEqual(resolveConfig({ fileConfig }).vision, null);
  await download('qwen3.5-4b-q4', dir, { fetchFn: fakeFetch([]) });
  const cfg = resolveConfig({ fileConfig });
  assert.strictEqual(cfg.vision.port, 9001);
  const cmds = serviceCommands(cfg);
  assert.strictEqual(cmds.vision.name, 'vision');
  assert.deepStrictEqual(cmds.vision.args.slice(0, 4), ['-m', modelPaths('qwen3.5-4b-q4', dir).model, '--mmproj', modelPaths('qwen3.5-4b-q4', dir).mmproj]);
  assert.ok(cmds.vision.args.join(' ').includes('--port 9001'));
  const backend = cmds.backend.args.join(' ');
  assert.ok(backend.includes('--qwen-url http://127.0.0.1:9000'));
  assert.ok(backend.includes('--vision-url http://127.0.0.1:9001'));
  assert.ok(backend.includes('--vision-model Qwen3.5-4B-Q4_K_M'));
  assert.ok(!backend.includes('visionModel'));
  // "Same as the main Qwen" runs nothing extra.
  const same = resolveConfig({ fileConfig: { ...fileConfig, settings: { visionModel: 'same' } } });
  assert.strictEqual(same.vision, null);
  assert.ok(!('vision' in serviceCommands(same)));
  assert.ok(!serviceCommands(same).backend.args.includes('--vision-url'));
  const sameArgs = serviceCommands(same).backend.args;
  assert.strictEqual(sameArgs[sameArgs.indexOf('--vision-model') + 1], path.basename(same.llama.model, '.gguf'));
});
