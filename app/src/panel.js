// Control panel: Home (what is happening now, services, performance), Log
// (every question with the screen at that moment and the answer) and
// Settings (saved to gvision.config.json by the main process).
'use strict';

const $ = (sel) => document.querySelector(sel);
const counts = new Map();
let answer = null;
let log = [];
let settings = null;
let services = { managed: false, services: [] };

const TOOL_NAMES = {
  set_watch: 'Find and highlight',
  clear_watch: 'Stop highlighting',
  query_state: 'Check the scene',
  read_text: 'Read text',
  recent_text: 'Recent text',
  look: 'Look back',
};
const VOICE_LABELS = { idle: 'Ready', listening: 'Listening', thinking: 'Thinking', speaking: 'Speaking' };

function msg(type, fields = {}) {
  return { v: 1, ts: Date.now() / 1000, type, ...fields };
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? '' : v);
  }
  node.append(...children.flat().filter((c) => c != null));
  return node;
}

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.appendChild(use);
  return svg;
}

function toast(text, bad = false) {
  const t = $('#toast');
  t.textContent = text;
  t.className = `toast${bad ? ' bad' : ''}`;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, bad ? 5000 : 1800);
}

// --- Tabs ---------------------------------------------------------------------

function showTab(name) {
  for (const b of document.querySelectorAll('.nav-item')) b.classList.toggle('active', b.dataset.tab === name);
  for (const p of document.querySelectorAll('.page')) p.classList.toggle('active', p.dataset.page === name);
  $('.main').scrollTop = 0;
}
for (const b of document.querySelectorAll('[data-tab]')) b.addEventListener('click', () => showTab(b.dataset.tab));
for (const b of document.querySelectorAll('[data-goto]')) b.addEventListener('click', () => showTab(b.dataset.goto));

// --- Formatting ---------------------------------------------------------------

const pad = (n) => String(n).padStart(2, '0');
function clock(ts) {
  const d = new Date(ts * 1000);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
function dayLabel(ts) {
  const d = new Date(ts * 1000);
  const today = new Date();
  const yesterday = new Date(today.getTime() - 86400000);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' });
}
function ago(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return dayLabel(ts);
}
function seconds(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}
function hotkeyParts(key) {
  const names = { alt: 'Alt', ctrl: 'Ctrl', shift: 'Shift', space: 'Space', caps_lock: 'Caps Lock' };
  return String(key).split('+').map((p) => names[p] || (p.length === 1 ? p.toUpperCase() : p.replace(/^f(\d+)$/, 'F$1')));
}
function kbdList(key) {
  return hotkeyParts(key).flatMap((p, i) => (i ? [' + ', el('kbd', { text: p })] : [el('kbd', { text: p })]));
}

// --- Connection and services ----------------------------------------------------

function setConnection({ connected }) {
  $('#conn').classList.toggle('ok', connected);
  $('.conn-text').textContent = connected ? 'Connected to the backend' : 'Waiting for the backend';
}

function renderServices(state) {
  services = state;
  const box = $('#services');
  const mini = $('#mini-services');
  if (state.error) {
    box.replaceChildren(el('div', { class: 'service' },
      el('span', { class: 'status-dot failed' }),
      el('div', {}, el('div', { class: 'service-name', text: 'Config error' }), el('div', { class: 'service-detail', text: state.error })),
      el('span')));
    mini.replaceChildren();
    return;
  }
  if (!state.managed) {
    box.replaceChildren(el('div', { class: 'empty-note', text: 'Not managed by the app: start llama-server and Python by hand.' }));
    mini.replaceChildren();
    return;
  }
  box.replaceChildren(...state.services.map((s) => {
    const busy = s.state === 'stopping';
    const running = s.state === 'ready' || s.state === 'starting' || busy;
    const actions = el('div', { class: 'service-actions' });
    const button = (text, iconName, onClick) => actions.append(el('button', {
      class: 'btn btn-small', disabled: busy,
      onclick: () => {
        for (const b of actions.querySelectorAll('button')) b.disabled = true; // feedback until the next update
        onClick();
      },
    }, icon(iconName), text));
    button(running ? 'Restart' : 'Start', 'refresh', () => window.gvision.restartService(s.name));
    if (running) button('Stop', 'x', () => window.gvision.stopService(s.name));
    return el('div', { class: 'service' },
      el('span', { class: `status-dot ${s.state}`, title: s.state }),
      el('div', {},
        el('div', { class: 'service-name' }, s.label, ' ', el('span', { class: `state-label ${s.state}`, text: `· ${s.state}` })),
        s.detail ? el('div', { class: 'service-detail', text: s.detail }) : null),
      actions);
  }));
  mini.replaceChildren(...state.services.map((s) => el('div', { class: 'mini' },
    el('span', { class: `status-dot ${s.state}` }), s.label.replace(/ \(.*\)$/, ''))));
}

// --- Home: now, performance -----------------------------------------------------

let voiceState = 'idle';

function renderVoice({ state, transcript }) {
  if (state === voiceState && transcript == null) return; // loudness ticks, ~20 a second
  voiceState = state;
  $('#orb').className = `orb ${state}`;
  const eyebrow = $('#voice-state');
  eyebrow.textContent = VOICE_LABELS[state] || state;
  eyebrow.className = `eyebrow ${state}`;
  const heard = $('#heard');
  if (state === 'listening') {
    heard.className = 'heard';
    heard.textContent = 'Listening…';
    answer = null;
    renderAnswer(null);
  }
  if (transcript != null) {
    heard.className = transcript ? 'heard said' : 'heard';
    heard.textContent = transcript || 'Nothing heard. Try again, a little closer to the mic.';
  }
}

function renderIdleHint() {
  const key = settings ? settings.values.pttKey : 'alt+3';
  $('#heard').className = 'heard';
  $('#heard').replaceChildren('Hold ', ...kbdList(key), ' and ask about what’s on screen.');
}

function renderAnswer(activeSegment) {
  const box = $('#answer');
  if (!answer) {
    box.replaceChildren();
    return;
  }
  box.replaceChildren(...answer.segments.map((seg) =>
    el('span', { class: `segment${seg.id === activeSegment ? ' active' : ''}`, text: `${seg.text} ` })));
}

function renderStatus(s) {
  const tiles = [];
  if (s.perception_fps != null) tiles.push(['Perception', s.perception_fps.toFixed(0), 'fps']);
  if (s.game_fps != null) tiles.push(['Game', s.game_fps.toFixed(0), 'fps']);
  if (s.vram_used_mb != null) tiles.push(['VRAM', (s.vram_used_mb / 1024).toFixed(1), 'GB']);
  for (const [k, v] of Object.entries(s.latency_ms)) tiles.push([`${k[0].toUpperCase()}${k.slice(1)} latency`, v.toFixed(0), 'ms']);
  $('#metrics').replaceChildren(...tiles.slice(0, 6).map(([label, value, unit]) => el('div', { class: 'metric' },
    el('div', { class: 'metric-value' }, value, el('small', { text: unit })),
    el('div', { class: 'metric-label', text: label }))));
  $('#components').replaceChildren(...Object.entries(s.components).map(([name, st]) =>
    el('span', { class: `chip ${st}`, title: st }, el('i'), name)));
}

function renderEmptyStatus() {
  $('#metrics').replaceChildren(el('div', { class: 'empty-note', text: 'No numbers yet. They appear once the backend is running.' }));
}

// --- Log ------------------------------------------------------------------------

function shot(entry, cls) {
  if (!entry.imageUrl) {
    return el('div', { class: `${cls} thumb none` }, icon('image'), cls === 'shot' ? 'No screenshot' : null);
  }
  return el('button', {
    class: `${cls} thumb`, title: 'Show the screen at that moment',
    onclick: (e) => { e.stopPropagation(); openLightbox(entry); },
  }, el('img', { src: entry.imageUrl, alt: '', loading: 'lazy' }));
}

function highlight(text, query) {
  if (!query) return [text];
  const out = [];
  const lower = text.toLowerCase();
  let i = 0;
  for (let j = lower.indexOf(query, i); j !== -1; j = lower.indexOf(query, i)) {
    out.push(text.slice(i, j), el('mark', { text: text.slice(j, j + query.length) }));
    i = j + query.length;
  }
  out.push(text.slice(i));
  return out;
}

function exchangeCard(entry, query, fresh) {
  const total = Object.values(entry.latencyMs || {}).reduce((a, b) => a + b, 0);
  const voice = entry.via === 'voice';
  return el('article', { class: `exchange${fresh ? ' fresh' : ''}`, id: `ex-${entry.id}` },
    shot(entry, 'shot'),
    el('div', { class: 'ex-body' },
      el('div', { class: 'ex-meta' },
        el('time', { text: clock(entry.askedTs), title: new Date(entry.askedTs * 1000).toLocaleString() }),
        el('span', { class: 'via' }, icon(voice ? 'mic' : 'keyboard'), voice ? 'Spoken' : 'Typed')),
      el('div', { class: 'bubble q' }, el('span', { class: 'who', text: 'You' }), ...highlight(entry.question, query)),
      el('div', { class: 'bubble a' }, el('span', { class: 'who', text: 'G-VISION' }), ...highlight(entry.answer, query)),
      el('div', { class: 'ex-foot' },
        ...(entry.tools || []).map((t) => el('span', { class: 'chip', title: t }, icon('tool'), TOOL_NAMES[t] || t)),
        (entry.tools || []).length ? null : el('span', { class: 'chip', text: 'No tools' }),
        total ? el('span', { class: 'latency', text: `answered in ${seconds(total)}`, title: Object.entries(entry.latencyMs).map(([k, v]) => `${k}: ${seconds(v)}`).join('\n') }) : null)));
}

function renderLog(freshId = null) {
  const query = $('#log-search').value.trim().toLowerCase();
  const shown = log.filter((e) => !query || e.question.toLowerCase().includes(query) || e.answer.toLowerCase().includes(query));
  const feed = $('#feed');
  if (!log.length) {
    feed.replaceChildren(el('div', { class: 'empty' }, icon('log'),
      el('strong', { text: 'No questions yet' }),
      el('div', {}, 'Hold ', ...kbdList(settings ? settings.values.pttKey : 'alt+3'), ' in game and ask something. Each question shows up here with what was on screen.')));
  } else if (!shown.length) {
    feed.replaceChildren(el('div', { class: 'empty' }, icon('search'), el('strong', { text: 'Nothing matches' }), `No question or answer contains “${query}”.`));
  } else {
    const items = [];
    let day = null;
    for (const e of [...shown].reverse()) { // newest first
      const d = dayLabel(e.askedTs);
      if (d !== day) items.push(el('div', { class: 'day', text: (day = d) }));
      items.push(exchangeCard(e, query, e.id === freshId));
    }
    feed.replaceChildren(...items);
  }
  const count = $('#log-count');
  count.hidden = !log.length;
  count.textContent = String(log.length);
  renderRecent();
}

function renderRecent() {
  const recent = log.slice(-4).reverse();
  $('#recent').replaceChildren(...(recent.length ? recent.map((e) => el('button', {
    class: 'recent-item',
    onclick: () => {
      showTab('log');
      const card = document.getElementById(`ex-${e.id}`);
      if (card) card.scrollIntoView({ block: 'center' });
    },
  }, shot(e, 'thumb'), el('div', { style: 'min-width:0' },
    el('div', { class: 'recent-q', text: e.question }),
    el('div', { class: 'recent-a', text: e.answer })),
  el('span', { class: 'recent-time', text: ago(e.askedTs) }))) : [el('div', { class: 'empty-note', text: 'Your questions and the answers will show up here.' })]));
}

function openLightbox(entry) {
  $('#lightbox-img').src = entry.imageUrl;
  $('#lightbox-cap').textContent = `${dayLabel(entry.askedTs)}, ${clock(entry.askedTs)}: “${entry.question}”`;
  $('#lightbox').hidden = false;
}
$('#lightbox').addEventListener('click', () => { $('#lightbox').hidden = true; });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') $('#lightbox').hidden = true;
});

$('#log-search').addEventListener('input', () => renderLog());
$('#clear-log').addEventListener('click', async () => {
  if (!log.length) return;
  if (!window.confirm(`Delete all ${log.length} questions and their screenshots?`)) return;
  await window.gvision.clearLog();
  log = [];
  renderLog();
  toast('Log cleared');
});

// --- Settings -------------------------------------------------------------------

let restartNeeded = false;

function setRestartNeeded(on) {
  restartNeeded = on;
  const backend = services.managed && services.services.some((s) => s.name === 'backend');
  $('#restart-banner').hidden = !on;
  $('#restart-backend').hidden = !backend;
  $('#settings-dot').hidden = !on;
}

async function save(key, value, tick) {
  const spec = settings.spec.find((s) => s.key === key);
  const r = await window.gvision.saveSettings({ [key]: value });
  if (!r.ok) {
    toast(r.error, true);
    renderSettings();
    return;
  }
  settings.values = r.values;
  if (tick) {
    tick.classList.add('show');
    clearTimeout(tick.timer);
    tick.timer = setTimeout(() => tick.classList.remove('show'), 1400);
  }
  if (!spec.live) setRestartNeeded(true);
  if (key === 'pttKey') renderIdleHint();
  syncDependent();
}

// Fields that only matter when another one is set (Whisper model vs Nemotron,
// situation notes without scene memory) are dimmed.
function syncDependent() {
  const v = settings.values;
  const dep = { whisperModel: v.asr === 'whisper', voice: v.speak, narrateEvery: v.sceneMemory };
  for (const [key, on] of Object.entries(dep)) {
    const f = document.querySelector(`[data-field="${key}"]`);
    if (f) f.classList.toggle('disabled', !on);
  }
}

function control(spec, value, tick) {
  switch (spec.type) {
    case 'toggle': {
      const sw = el('button', { class: 'switch', role: 'switch', 'aria-checked': String(value), 'aria-label': spec.label });
      sw.addEventListener('click', () => {
        const on = sw.getAttribute('aria-checked') !== 'true';
        sw.setAttribute('aria-checked', String(on));
        save(spec.key, on, tick);
      });
      return sw;
    }
    case 'choice': {
      const sel = el('select', { 'aria-label': spec.label }, ...spec.choices.map(([v, label]) => el('option', { value: v, text: label })));
      sel.value = value;
      sel.addEventListener('change', () => save(spec.key, sel.value, tick));
      return sel;
    }
    case 'number': {
      const input = el('input', { class: 'num', type: 'number', min: spec.min, max: spec.max, step: spec.step, value, 'aria-label': spec.label });
      input.addEventListener('change', () => save(spec.key, Number(input.value), tick));
      return el('div', { class: 'num-wrap' }, input, spec.unit || '');
    }
    case 'range': {
      const input = el('input', { type: 'range', min: spec.min, max: spec.max, step: spec.step, value, 'aria-label': spec.label });
      const out = el('output');
      const show = () => {
        out.textContent = `${Math.round(Number(input.value) * 100)}%`;
        input.style.setProperty('--fill', `${(Number(input.value) - spec.min) / (spec.max - spec.min) * 100}%`);
      };
      show();
      input.addEventListener('input', show);
      input.addEventListener('change', () => save(spec.key, Number(input.value), tick));
      return el('div', { class: 'range-wrap' }, input, out);
    }
    case 'hotkey':
      return hotkeyRecorder(spec, value, tick);
    default:
      return el('span', { text: String(value) });
  }
}

// pynput key names for KeyboardEvent.code values.
function keyFromCode(code) {
  let m;
  if ((m = /^Digit(\d)$/.exec(code))) return m[1];
  if ((m = /^Key([A-Z])$/.exec(code))) return m[1].toLowerCase();
  if ((m = /^F(\d{1,2})$/.exec(code))) return `f${m[1]}`;
  const named = {
    Space: 'space', Tab: 'tab', CapsLock: 'caps_lock', Backquote: '`', Insert: 'insert', Home: 'home', End: 'end',
    PageUp: 'page_up', PageDown: 'page_down', Pause: 'pause', ScrollLock: 'scroll_lock',
  };
  return named[code] || null;
}

function hotkeyRecorder(spec, value, tick) {
  const btn = el('button', { class: 'hotkey', title: 'Click, then press the new key combination' }, ...kbdList(value));
  let recording = false;
  const stop = (key) => {
    recording = false;
    btn.classList.remove('recording');
    window.removeEventListener('keydown', onKey, true);
    btn.replaceChildren(...kbdList(key || settings.values.pttKey));
    if (key && key !== settings.values.pttKey) save(spec.key, key, tick);
  };
  const onKey = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') return stop(null);
    const key = keyFromCode(e.code);
    if (!key) return undefined; // a modifier alone, or a key push-to-talk can't use: keep waiting
    const mods = [e.altKey && 'alt', e.ctrlKey && 'ctrl', e.shiftKey && 'shift'].filter(Boolean);
    return stop([...mods, key].join('+'));
  };
  btn.addEventListener('click', () => {
    if (recording) return stop(null);
    recording = true;
    btn.classList.add('recording');
    btn.textContent = 'Press keys… (Esc to cancel)';
    window.addEventListener('keydown', onKey, true);
    return undefined;
  });
  return btn;
}

function renderSettings() {
  if (!settings) return;
  $('#settings-file').textContent = `Saved to ${settings.file}`;
  const err = $('#settings-error');
  err.hidden = !settings.error;
  err.textContent = settings.error ? `Can't read the config file, so these are the defaults: ${settings.error}` : '';
  const groups = new Map();
  for (const spec of settings.spec) {
    if (!groups.has(spec.group)) groups.set(spec.group, []);
    groups.get(spec.group).push(spec);
  }
  $('#settings-form').replaceChildren(...[...groups].map(([name, specs]) => el('section', { class: 'card group' },
    el('h2', { text: name }),
    ...specs.map((spec) => {
      const tick = el('span', { class: 'saved-tick', text: 'Saved' });
      return el('div', { class: 'field', 'data-field': spec.key },
        el('div', {},
          el('div', { class: 'field-label' }, spec.label, tick),
          el('div', { class: 'field-help', text: spec.help })),
        control(spec, settings.values[spec.key], tick));
    }))));
  syncDependent();
}

$('#restart-backend').addEventListener('click', () => {
  window.gvision.restartService('backend');
  setRestartNeeded(false);
  toast('Restarting the backend with the new settings');
  showTab('home');
});

// --- Wiring ---------------------------------------------------------------------

window.gvision.onConnection(setConnection);
window.gvision.getConnection().then(setConnection);
window.gvision.onServices(renderServices);
window.gvision.getServices().then(renderServices);
window.gvision.getSettings().then((s) => {
  settings = s;
  renderSettings();
  renderIdleHint();
  renderLog();
});
window.gvision.getLog().then((entries) => {
  log = entries;
  renderLog();
});
window.gvision.onExchange((entry) => {
  log.push(entry);
  renderLog(entry.id);
});

$('#open-logs').addEventListener('click', () => window.gvision.openLogs());
$('#update').addEventListener('click', (e) => {
  const b = e.currentTarget;
  b.disabled = true;
  b.lastChild.textContent = 'Updating…';
  window.gvision.update();
});
$('#clear').addEventListener('click', () => {
  window.gvision.send(msg('clear', { reason: 'panel button' }));
  toast('Highlights cleared');
});

window.gvision.onMessage((m) => {
  counts.set(m.type, (counts.get(m.type) || 0) + 1);
  switch (m.type) {
    case 'status':
      renderStatus(m);
      break;
    case 'answer':
      answer = m;
      renderAnswer(null);
      break;
    case 'segment_started':
      if (answer && answer.answer_id === m.answer_id) renderAnswer(m.segment_id);
      break;
    case 'answer_finished':
      renderAnswer(null);
      break;
    case 'voice':
      renderVoice(m);
      break;
    default:
      break;
  }
});

renderEmptyStatus();
renderRecent();
setInterval(() => {
  $('#counts').replaceChildren(...[...counts].sort().map(([k, v]) => el('span', { class: 'chip', text: `${k} ${v}` })));
}, 1000);
setInterval(renderRecent, 30000); // keep "3 min ago" honest
