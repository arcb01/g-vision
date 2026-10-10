// Settings edited in the panel's Settings tab. They live in the "settings"
// section of gvision.config.json and become command-line flags for the
// Python backend (applied on its next restart), except the ones marked live,
// which reach the running backend as a config_changed message.
//
// In two-PC mode each setting lives on one PC: `side: 'server'` ones are the
// AI server's (the backend runs there) and the gaming PC edits them over the
// network; `side: 'gaming'` ones only matter on the PC that plays. Network
// settings are each PC's own.
'use strict';

const fs = require('node:fs');
const { VISION_MODELS } = require('./models');

const ROLES = [['standalone', 'Standalone'], ['gaming', 'Gaming PC'], ['server', 'AI server']];

const SPEC = [
  {
    key: 'role', group: 'Network', label: 'This PC is', type: 'choice', default: 'standalone', side: 'local', relaunch: true,
    choices: ROLES,
    details: {
      standalone: 'Plays and thinks on this PC, as usual.',
      gaming: 'Plays here: the screen, your voice and the answers go to the AI server PC, which runs the models.',
      server: 'Runs the models for a gaming PC on the network. No overlay here. Allow G-VISION through Windows Firewall when asked.',
    },
    help: 'Use two PCs so the models don\'t share the GPU with the game. Changing it restarts G-VISION.',
  },
  {
    key: 'serverHost', group: 'Network', label: 'AI server address', type: 'text', default: '', side: 'local', relaunch: true,
    placeholder: '192.168.1.20',
    help: 'The AI server PC\'s IP address or name. Its Home page shows it.',
  },
  {
    key: 'pttKey', group: 'Voice', side: 'gaming', label: 'Push-to-talk key', type: 'hotkey', default: 'alt+3', flag: '--ptt-key',
    help: 'Hold it while you ask. A key with optional Alt, Ctrl or Shift.',
  },
  {
    key: 'asr', side: 'server', group: 'Voice', label: 'Speech-to-text', type: 'choice', default: 'whisper', flag: '--asr',
    choices: [['whisper', 'Whisper'], ['nemotron', 'Nemotron']],
    help: 'Whisper copes better with accents; Nemotron is faster.',
  },
  {
    key: 'whisperModel', side: 'server', group: 'Voice', label: 'Whisper model', type: 'choice', default: 'medium', flag: '--whisper-model',
    choices: [['small', 'Small'], ['medium', 'Medium'], ['large-v3-turbo', 'Large v3 turbo'], ['large-v3', 'Large v3']],
    help: 'Bigger hears better and uses more VRAM.',
  },
  {
    key: 'voice', side: 'server', group: 'Voice', label: 'Voice', type: 'choice', default: 'af_heart', flag: '--voice',
    choices: [
      ['af_heart', 'Heart (US)'], ['af_bella', 'Bella (US)'], ['af_nicole', 'Nicole (US)'],
      ['am_michael', 'Michael (US)'], ['am_adam', 'Adam (US)'], ['bf_emma', 'Emma (UK)'], ['bm_george', 'George (UK)'],
    ],
    help: 'Kokoro voice that speaks the answers.',
  },
  {
    key: 'speak', side: 'server', group: 'Voice', label: 'Speak answers', type: 'toggle', default: true, offFlag: '--no-tts',
    help: 'Off: answers only appear here, in the Log.',
  },
  {
    key: 'readText', side: 'server', group: 'Vision', label: 'Read on-screen text', type: 'toggle', default: true, offFlag: '--no-text',
    help: 'Signs, quests and menus, with RapidOCR on the CPU.',
  },
  {
    key: 'sceneMemory', side: 'server', group: 'Vision', label: 'Scene memory', type: 'toggle', default: true, offFlag: '--no-memory',
    help: 'Remembers the last minute for "what just hit me?".',
  },
  {
    key: 'visionModel', side: 'server', group: 'Vision', label: 'Vision model', type: 'choice', default: 'same',
    choices: VISION_MODELS.map((m) => [m.id, m.label]),
    details: Object.fromEntries(VISION_MODELS.map((m) => [m.id, m.summary])),
    help: 'The model that looks at the screen (look back, situation notes). Questions are still routed by the 2B. A new pick downloads once, then the vision server and backend restart on their own.',
  },
  {
    key: 'lookSize', side: 'server', group: 'Vision', label: 'Image resolution', type: 'choice', default: '1600', flag: '--look-size',
    choices: [['640', '640 px (fastest)'], ['1024', '1024 px'], ['1280', '1280 px'], ['1600', '1600 px'], ['1920', '1920 px'], ['full', 'Full screen']],
    help: 'How sharp the screen is when the vision model answers a question about right now. Sharper reads small icons and numbers better but takes longer.',
  },
  {
    key: 'visionReasoning', side: 'server', group: 'Vision', label: 'Reasoning', type: 'toggle', default: false,
    onFlag: '--vision-reasoning',
    help: 'The vision model thinks before it answers. Can help with counting and telling items apart, but answers take several seconds longer. Needs a vision model other than "Same as the main Qwen".',
  },
  {
    key: 'visionOnly', side: 'server', group: 'Vision', label: 'Vision only (testing)', type: 'toggle', default: false,
    onFlag: '--vision-only',
    help: 'Skip routing: every question goes straight to the vision model, which answers. No highlights or text reading.',
  },
  {
    key: 'narrateEvery', side: 'server', group: 'Vision', label: 'Situation notes every', type: 'number', default: 25,
    min: 0, max: 300, step: 5, unit: 's', flag: '--narrate-every',
    help: 'How often Qwen sums up what is going on. 0 turns it off.',
  },
  {
    key: 'dimStrength', side: 'gaming', group: 'Overlay', label: 'Dim strength', type: 'range', default: 0.6,
    min: 0, max: 0.9, step: 0.05, live: true,
    help: 'How dark the screen gets around what is highlighted.',
  },
];

const BY_KEY = Object.fromEntries(SPEC.map((s) => [s.key, s]));
// An IPv4 address or a host name; no scheme or port.
const HOST = /^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$/;

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
    case 'text': {
      const text = String(value).trim();
      if (key === 'serverHost' && text && !HOST.test(text)) {
        return new Error(`${s.label}: expected an IP address or a computer name, e.g. 192.168.1.20`);
      }
      return text;
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
    if (s.onFlag && value === true && !explicitArgs.includes(s.onFlag)) args.push(s.onFlag);
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

module.exports = { ROLES, SPEC, check, defaults, readFile, resolveSettings, saveSettings, settingsArgs };
