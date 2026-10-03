// Control panel shell: connection state, dashboard metrics, push-to-talk state, the current
// answer (karaoke-style segment highlight) and a couple of controls.
'use strict';

const counts = new Map();
let answer = null;

function msg(type, fields = {}) {
  return { v: 1, ts: Date.now() / 1000, type, ...fields };
}

function setConnection({ connected, url }) {
  const el = document.getElementById('conn');
  el.textContent = connected ? `connected to ${url}` : 'disconnected';
  el.classList.toggle('ok', connected);
}

function renderRows(table, rows) {
  table.replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement('tr');
      for (const text of [k, v]) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.appendChild(td);
      }
      return tr;
    }),
  );
}

function renderStatus(s) {
  const rows = [];
  if (s.perception_fps != null) rows.push(['Perception FPS', s.perception_fps.toFixed(0)]);
  if (s.game_fps != null) rows.push(['Game FPS', s.game_fps.toFixed(0)]);
  if (s.vram_used_mb != null) rows.push(['VRAM', `${(s.vram_used_mb / 1024).toFixed(1)} GB`]);
  for (const [k, v] of Object.entries(s.latency_ms)) rows.push([`Latency: ${k}`, `${v.toFixed(1)} ms`]);
  for (const [k, v] of Object.entries(s.components)) rows.push([k, v]);
  renderRows(document.getElementById('status'), rows);
}

function renderAnswer(activeSegment) {
  const el = document.getElementById('answer');
  if (!answer) {
    el.textContent = 'No answer yet';
    return;
  }
  el.replaceChildren(
    ...answer.segments.map((seg) => {
      const span = document.createElement('span');
      span.className = 'segment' + (seg.id === activeSegment ? ' active' : '');
      span.textContent = seg.text + ' ';
      return span;
    }),
  );
}

// Processes the app started (Qwen, Python backend): state, why one failed,
// and a restart button each.
function renderServices({ managed, error, services }) {
  const table = document.getElementById('services');
  if (error) {
    renderRows(table, [['Config error', error]]);
    return;
  }
  if (!managed) {
    renderRows(table, [['Not managed by the app', 'start llama-server and Python by hand']]);
    return;
  }
  table.replaceChildren(
    ...services.map((s) => {
      const tr = document.createElement('tr');
      const name = document.createElement('td');
      name.textContent = s.label;
      const state = document.createElement('td');
      const pill = document.createElement('span');
      pill.className = `pill ${s.state}`;
      pill.textContent = s.state;
      state.appendChild(pill);
      const detail = document.createElement('td');
      detail.className = 'detail';
      detail.textContent = s.detail;
      const action = document.createElement('td');
      const busy = s.state === 'stopping';
      const button = (text, onClick) => {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.disabled = busy;
        btn.addEventListener('click', () => {
          // Feedback right away; the next state update re-renders the row.
          for (const b of action.querySelectorAll('button')) b.disabled = true;
          onClick();
        });
        action.appendChild(btn);
      };
      const running = s.state === 'ready' || s.state === 'starting' || busy;
      button(running ? 'Restart' : 'Start', () => window.gvision.restartService(s.name));
      if (running) button('Stop', () => window.gvision.stopService(s.name));
      tr.append(name, state, detail, action);
      return tr;
    }),
  );
}

// Push-to-talk state and what speech-to-text heard, to spot ASR mistakes.
function renderVoice({ state, transcript }) {
  const pill = document.getElementById('voice-state');
  pill.textContent = state;
  pill.className = `pill ${state}`;
  if (state === 'listening') document.getElementById('heard').textContent = '';
  if (transcript != null) document.getElementById('heard').textContent = transcript ? `"${transcript}"` : '(nothing heard)';
}

window.gvision.onConnection(setConnection);
window.gvision.getConnection().then(setConnection);
window.gvision.onServices(renderServices);
window.gvision.getServices().then(renderServices);
document.getElementById('open-logs').addEventListener('click', () => window.gvision.openLogs());

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

setInterval(() => renderRows(document.getElementById('counts'), [...counts].sort()), 500);

document.getElementById('clear').addEventListener('click', () => {
  window.gvision.send(msg('clear', { reason: 'panel button' }));
});

document.getElementById('dim').addEventListener('input', (e) => {
  document.getElementById('dim-value').textContent = `${Math.round(Number(e.target.value) * 100)}%`;
});

document.getElementById('dim').addEventListener('change', (e) => {
  window.gvision.send(msg('config_changed', { changes: { 'visual_effects.dim_strength': Number(e.target.value) } }));
});
