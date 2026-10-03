// Settings edited in the panel's Settings tab. They live in the "settings"
// section of gvision.config.json and become command-line flags for the
// Python backend (applied on its next restart), except the ones marked live,
// which reach the running backend as a config_changed message.
'use strict';

const fs = require('node:fs');

const SPEC = [
  {
    key: 'pttKey', group: 'Voice', label: 'Push-to-talk key', type: 'hotkey', default: 'alt+3', flag: '--ptt-key',
    help: 'Hold it while you ask. A key with optional Alt, Ctrl or Shift.',
  },
  {
    key: 'asr', group: 'Voice', label: 'Speech-to-text', type: 'choice', default: 'whisper', flag: '--asr',
    choices: [['whisper', 'Whisper'], ['nemotron', 'Nemotron']],
    help: 'Whisper copes better with accents; Nemotron is faster.',
  },
  {
    key: 'whisperModel', group: 'Voice', label: 'Whisper model', type: 'choice', default: 'medium', flag: '--whisper-model',
    choices: [['small', 'Small'], ['medium', 'Medium'], ['large-v3-turbo', 'Large v3 turbo'], ['large-v3', 'Large v3']],
    help: 'Bigger hears better and uses more VRAM.',
  },
  {
    key: 'voice', group: 'Voice', label: 'Voice', type: 'choice', default: 'af_heart', flag: '--voice',
    choices: [
      ['af_heart', 'Heart (US)'], ['af_bella', 'Bella (US)'], ['af_nicole', 'Nicole (US)'],
      ['am_michael', 'Michael (US)'], ['am_adam', 'Adam (US)'], ['bf_emma', 'Emma (UK)'], ['bm_george', 'George (UK)'],
    ],
    help: 'Kokoro voice that speaks the answers.',
  },
  {
    key: 'speak', group: 'Voice', label: 'Speak answers', type: 'toggle', default: true, offFlag: '--no-tts',
    help: 'Off: answers only appear here, in the Log.',
  },
  {
    key: 'readText', group: 'Vision', label: 'Read on-screen text', type: 'toggle', default: true, offFlag: '--no-text',
    help: 'Signs, quests and menus, with RapidOCR on the CPU.',
  },
  {
    key: 'sceneMemory', group: 'Vision', label: 'Scene memory', type: 'toggle', default: true, offFlag: '--no-memory',
    help: 'Remembers the last minute for "what just hit me?".',
  },
  {
    key: 'narrateEvery', group: 'Vision', label: 'Situation notes every', type: 'number', default: 25,
    min: 0, max: 300, step: 5, unit: 's', flag: '--narrate-every',
    help: 'How often Qwen sums up what is going on. 0 turns it off.',
  },
  {
    key: 'dimStrength', group: 'Overlay', label: 'Dim strength', type: 'range', default: 0.6,
    min: 0, max: 0.9, step: 0.05, live: true,
    help: 'How dark the screen gets around what is highlighted.',
  },
];

const BY_KEY = Object.fromEntries(SPEC.map((s) => [s.key, s]));

function defaults() {
  return Object.fromEntries(SPEC.map((s) => [s.key, s.default]));
}

// A clean value for one setting, or an Error saying why it was refused.
function check(key, value) {
  const s = BY_KEY[key];
  if (!s) return new Error(`unknown setting "${key}"`);
  switch (s.type) {
    case 'toggle':
      return typeof value === 'boolean' ? value : new Error(`${s.label}: expected on or off`);
    case 'choice':
      return s.choices.some(([v]) => v === value) ? value : new Error(`${s.label}: unknown choice "${value}"`);
    case 'number':
    case 'range': {
      const n = Number(value);
      if (typeof value === 'boolean' || !Number.isFinite(n) || n < s.min || n > s.max) {
        return new Error(`${s.label}: expected a number from ${s.min} to ${s.max}`);
      }
      return n;
    }
    case 'hotkey': {
      const parts = String(value).toLowerCase().split('+').map((p) => p.trim()).filter(Boolean);
      const key = parts.pop();
      if (!key || parts.some((m) => !['alt', 'ctrl', 'shift'].includes(m)) || ['alt', 'ctrl', 'shift'].includes(key)) {
        return new Error(`${s.label}: expected a key with optional alt, ctrl or shift, e.g. alt+3`);
      }
      return [...parts, key].join('+');
    }
    default:
      return new Error(`${s.label}: unsupported type`);
  }
}

// Defaults <- the file's settings section. A bad value in the file falls back
// to the default rather than stopping the app.
function resolveSettings(fileSettings = {}) {
  const out = defaults();
  for (const [k, v] of Object.entries(fileSettings || {})) {
    const value = check(k, v);
    if (!(value instanceof Error)) out[k] = value;
  }
  return out;
}

// Backend flags for these settings. Flags already in python.args win, so a
// hand-written config keeps working.
function settingsArgs(settings, explicitArgs = []) {
  const args = [];
  for (const s of SPEC) {
    const value = settings[s.key];
    if (s.flag && !explicitArgs.includes(s.flag)) args.push(s.flag, String(value));
    if (s.offFlag && value === false && !explicitArgs.includes(s.offFlag)) args.push(s.offFlag);
  }
  return args;
}

function readFile(file) {
  if (!fs.existsSync(file)) return {};
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

// Merges `changes` into the settings section of the config file, keeping
// everything else in it. Throws on an invalid value; nothing is written then.
function saveSettings(file, changes) {
  const clean = {};
  for (const [k, v] of Object.entries(changes)) {
    const value = check(k, v);
    if (value instanceof Error) throw value;
    clean[k] = value;
  }
  const raw = readFile(file);
  raw.settings = { ...(raw.settings || {}), ...clean };
  fs.writeFileSync(file, `${JSON.stringify(raw, null, 2)}\n`);
  return resolveSettings(raw.settings);
}

module.exports = { SPEC, check, defaults, readFile, resolveSettings, saveSettings, settingsArgs };
