// Vision models the Settings tab offers for looking at the screen (the look
// tool and the situation notes). Routing and answers stay on the main Qwen
// (Qwen3.5 2B Q4_K_M); picking anything but "same" runs a second
// llama-server with the chosen model. Files download from Hugging Face into
// <llama.dir>/models/vision/<id>/ the first time a model is picked.
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const GB = 1e9;

// Sizes are the Hugging Face file sizes. VRAM and speed are estimates scaled
// from the 2B Q4_K_M measured on Arnau's RTX 3090 (2.7 GB, 0.3-1.4 s per
// image question), except where marked measured.
const VISION_MODELS = [
  {
    id: 'same',
    label: 'Same as the main Qwen (2B Q4_K_M)',
    summary: 'No extra download or VRAM. Measured: 0.3-1.4 s per look.',
    files: [],
  },
  {
    id: 'qwen3.8-27b-ud-iq3s',
    label: 'Qwen3.8 27B UD-IQ3_S',
    summary: '12.9 GB download, about +14 GB VRAM. Much slower per look (several seconds); tight on 24 GB with a game running.',
    repo: 'unsloth/Qwen3.8-27B-GGUF',
    files: [['Qwen3.8-27B-UD-IQ3_S.gguf', 12.0 * GB], ['mmproj-F16.gguf', 0.928 * GB]],
  },
  {
    id: 'qwen3.5-9b-ud-q6kxl',
    label: 'Qwen3.5 9B UD-Q6_K_XL',
    summary: '9.7 GB download, about +10.5 GB VRAM. About 3-4x slower per look; watch game FPS.',
    repo: 'unsloth/Qwen3.5-9B-GGUF',
    files: [['Qwen3.5-9B-UD-Q6_K_XL.gguf', 8.76 * GB], ['mmproj-F16.gguf', 0.918 * GB]],
  },
  {
    id: 'qwen3.5-4b-ud-q8kxl',
    label: 'Qwen3.5 4B UD-Q8_K_XL',
    summary: '6.6 GB download, about +7.3 GB VRAM. About 2-2.5x slower per look.',
    repo: 'unsloth/Qwen3.5-4B-GGUF',
    files: [['Qwen3.5-4B-UD-Q8_K_XL.gguf', 5.95 * GB], ['mmproj-F16.gguf', 0.672 * GB]],
  },
  {
    id: 'qwen3.5-9b-q3km',
    label: 'Qwen3.5 9B Q3_K_M',
    summary: '5.6 GB download, about +6.3 GB VRAM. About 3x slower per look.',
    repo: 'unsloth/Qwen3.5-9B-GGUF',
    files: [['Qwen3.5-9B-Q3_K_M.gguf', 4.67 * GB], ['mmproj-F16.gguf', 0.918 * GB]],
  },
  {
    id: 'glm-4.6v-flash-ud-q6kxl',
    label: 'GLM-4.6V-Flash UD-Q6_K_XL',
    summary: '10.7 GB download, about +11.5 GB VRAM. 9B, about 3-4x slower per look; watch game FPS.',
    repo: 'unsloth/GLM-4.6V-Flash-GGUF',
    files: [['GLM-4.6V-Flash-UD-Q6_K_XL.gguf', 8.89 * GB], ['mmproj-F16.gguf', 1.79 * GB]],
  },
  {
    id: 'gemma-4-26b-a4b-ud-iq2m',
    label: 'Gemma 4 26B-A4B UD-IQ2_M',
    summary: '11.2 GB download, about +12 GB VRAM. Mixture of experts with 4B active, so about as fast per look as a 4B.',
    repo: 'unsloth/gemma-4-26B-A4B-it-GGUF',
    files: [['gemma-4-26B-A4B-it-UD-IQ2_M.gguf', 10.0 * GB], ['mmproj-F16.gguf', 1.19 * GB]],
  },
];

const BY_ID = Object.fromEntries(VISION_MODELS.map((m) => [m.id, m]));

function folder(llamaDir, id) {
  return path.join(llamaDir, 'models', 'vision', id);
}

// { model, mmproj } paths for a model, or null for "same".
function modelPaths(id, llamaDir) {
  const m = BY_ID[id];
  if (!m || !m.files.length) return null;
  const dir = folder(llamaDir, id);
  return { model: path.join(dir, m.files[0][0]), mmproj: path.join(dir, m.files[1][0]) };
}

// Files still to download: [{ name, url, dest, bytes }].
function missingFiles(id, llamaDir) {
  const m = BY_ID[id];
  if (!m) throw new Error(`unknown vision model "${id}"`);
  return m.files
    .map(([name, bytes]) => ({
      name, bytes, url: `https://huggingface.co/${m.repo}/resolve/main/${name}`, dest: path.join(folder(llamaDir, id), name),
    }))
    .filter((f) => !fs.existsSync(f.dest));
}

// Downloads what is missing, reporting { file, done, total } (bytes over all
// missing files). Each file goes to <dest>.part first, so an interrupted
// download is never mistaken for a finished one.
async function download(id, llamaDir, { onProgress = () => {}, fetchFn = fetch, signal } = {}) {
  const files = missingFiles(id, llamaDir);
  const total = files.reduce((n, f) => n + f.bytes, 0);
  let done = 0;
  for (const f of files) {
    fs.mkdirSync(path.dirname(f.dest), { recursive: true });
    const res = await fetchFn(f.url, { redirect: 'follow', signal });
    if (!res.ok || !res.body) throw new Error(`${f.name}: download failed (HTTP ${res.status})`);
    const part = `${f.dest}.part`;
    const out = fs.createWriteStream(part);
    let last = 0;
    try {
      for await (const chunk of res.body) {
        if (!out.write(chunk)) await new Promise((r) => out.once('drain', r));
        done += chunk.length;
        if (done - last > 8e6) {
          last = done;
          onProgress({ file: f.name, done, total });
        }
      }
      await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
    } catch (err) {
      out.destroy();
      fs.rmSync(part, { force: true });
      throw err;
    }
    fs.renameSync(part, f.dest);
    onProgress({ file: f.name, done, total });
  }
  return { downloaded: files.map((f) => f.name) };
}

module.exports = { VISION_MODELS, download, missingFiles, modelPaths };
