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
    id: 'qwen3.5-2b-q8',
    label: 'Qwen3.5 2B Q8_0',
    summary: '2.7 GB download, about +3.5 GB VRAM. About the same speed; small quality gain over Q4.',
    repo: 'unsloth/Qwen3.5-2B-GGUF',
    files: [['Qwen3.5-2B-Q8_0.gguf', 2.01 * GB], ['mmproj-F16.gguf', 0.668 * GB]],
  },
  {
    id: 'qwen3.5-4b-q4',
    label: 'Qwen3.5 4B Q4_K_M (recommended step up)',
    summary: '3.4 GB download, about +4.5 GB VRAM. About 1.5-2x slower per look; the natural step up.',
    repo: 'unsloth/Qwen3.5-4B-GGUF',
    files: [['Qwen3.5-4B-Q4_K_M.gguf', 2.74 * GB], ['mmproj-F16.gguf', 0.672 * GB]],
  },
  {
    id: 'qwen3.5-4b-q8',
    label: 'Qwen3.5 4B Q8_0',
    summary: '5.2 GB download, about +6.5 GB VRAM. About 2-2.5x slower per look.',
    repo: 'unsloth/Qwen3.5-4B-GGUF',
    files: [['Qwen3.5-4B-Q8_0.gguf', 4.48 * GB], ['mmproj-F16.gguf', 0.672 * GB]],
  },
  // Community distill of Qwen3.8 into the Qwen3.5 2B architecture. It ships
  // without a vision file, so it borrows the stock 2B's mmproj: experimental,
  // since the distill was trained on text only.
  {
    id: 'qwen3.8-2b-distill-q4',
    label: 'Qwen3.8 2B Distill Q4_K_M (experimental)',
    summary: '2.0 GB download, about +2.7 GB VRAM. Same speed as the 2B. Text-only distill using the 2B\'s vision file, so image answers may be worse.',
    repo: 'empero-ai/Qwen3.8-2B-Distill-GGUF',
    files: [['Qwen3.8-2B-Q4_K_M.gguf', 1.31 * GB], ['mmproj-F16.gguf', 0.668 * GB, 'unsloth/Qwen3.5-2B-GGUF']],
  },
  {
    id: 'qwen3.8-2b-distill-q8',
    label: 'Qwen3.8 2B Distill Q8_0 (experimental)',
    summary: '2.7 GB download, about +3.5 GB VRAM. Same speed as the 2B. Text-only distill using the 2B\'s vision file, so image answers may be worse.',
    repo: 'empero-ai/Qwen3.8-2B-Distill-GGUF',
    files: [['Qwen3.8-2B-Q8_0.gguf', 2.08 * GB], ['mmproj-F16.gguf', 0.668 * GB, 'unsloth/Qwen3.5-2B-GGUF']],
  },
  {
    id: 'qwen3.5-9b-q4',
    label: 'Qwen3.5 9B Q4_K_M (best quality)',
    summary: '6.6 GB download, about +8 GB VRAM. About 3-4x slower per look; watch game FPS.',
    repo: 'unsloth/Qwen3.5-9B-GGUF',
    files: [['Qwen3.5-9B-Q4_K_M.gguf', 5.68 * GB], ['mmproj-F16.gguf', 0.918 * GB]],
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

// Files still to download: [{ name, url, dest, bytes }]. A file may name its
// own repo (a vision file borrowed from another model).
function missingFiles(id, llamaDir) {
  const m = BY_ID[id];
  if (!m) throw new Error(`unknown vision model "${id}"`);
  return m.files
    .map(([name, bytes, repo = m.repo]) => ({
      name, bytes, url: `https://huggingface.co/${repo}/resolve/main/${name}`, dest: path.join(folder(llamaDir, id), name),
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
