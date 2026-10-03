// Transparent overlay: draws tracked objects, semantic highlights and the
// spotlight dim layer from bridge messages (plan sections 5.5 and 6).
'use strict';

const COLORS = { target: 0xffc83d, danger: 0xff4d4d, info: 0x3de0ff };
const SUBTLE = 0xffffff;
const DIM_FADE_MS = 200;
const MAX_EXTRAPOLATION_S = 0.25;
// Glow: a blurred wide stroke under a dark edge and a bright core (plan 6.3).
const GLOW_WIDTH_PX = 14;
const GLOW_BLUR = 10;
const CORE_WIDTH_PX = { focused: 4, mentioned: 3 };
const PULSE_HZ = 1; // slow, well below flashing (plan 6.3)
const PULSE_DEPTH = 0.2;
// Developer aid until the debug overlay exists: show unhighlighted tracks faintly
// (Python only sends them with --show-all). No text is drawn on screen.
const SHOW_ALL_TRACKS = true;
const LOST_ALPHA = 0.4; // lost tracks stay at their last position, faded (plan 5.4)
// Voice indicator: waves while you talk (cyan) and while G-VISION answers (gold).
const VOICE_COLORS = { listening: 0x3de0ff, thinking: 0xffc83d, speaking: 0xffc83d };
const VOICE_BARS = 28;
const VOICE_FADE_MS = 180;
const VOICE_PILL = { w: 300, h: 64, bottom: 0.1 }; // bottom: share of screen height
const FILL_ALPHA = 0.22; // see-through tint inside highlighted contours

const state = {
  objects: { frameTs: 0, list: [] },
  highlights: new Map(), // ref -> highlight message (with box for text:/region: refs)
  focus: new Set(),
  dim: { on: false, strength: 0.6, alpha: 0 },
  voice: { state: 'idle', level: null, smooth: 0, alpha: 0, shown: 'idle' },
};

function handle(msg) {
  switch (msg.type) {
    case 'objects': {
      state.objects = { frameTs: msg.frame_ts, list: msg.objects };
      // Forget highlights and focus on tracks the tracker has removed.
      const live = new Set(msg.objects.map((o) => o.ref));
      for (const ref of state.highlights.keys()) {
        if (ref.startsWith('obj:') && !live.has(ref)) state.highlights.delete(ref);
      }
      for (const ref of state.focus) {
        if (ref.startsWith('obj:') && !live.has(ref)) state.focus.delete(ref);
      }
      break;
    }
    case 'highlight':
      state.highlights.set(msg.ref, msg);
      break;
    case 'focus':
      state.focus = new Set(msg.refs);
      break;
    case 'dim':
      state.dim.on = msg.on;
      state.dim.strength = msg.strength;
      break;
    case 'voice':
      if (msg.state !== state.voice.state) state.voice.level = null;
      state.voice.state = msg.state;
      if (msg.level != null) state.voice.level = msg.level;
      break;
    case 'clear':
      state.highlights.clear();
      state.focus.clear();
      state.dim.on = false;
      break;
    default:
      break; // answer, badges, status... are handled by the panel for now
  }
}

// Shift boxes and outlines by tracker velocity to hide display delay (plan 5.5).
function extrapolated(obj, nowS) {
  const dt = Math.min(Math.max(nowS - state.objects.frameTs, 0), MAX_EXTRAPOLATION_S);
  const [vx, vy] = obj.velocity;
  const dx = vx * dt;
  const dy = vy * dt;
  return {
    box: { x: obj.box.x + dx, y: obj.box.y + dy, w: obj.box.w, h: obj.box.h },
    outline: obj.outline ? obj.outline.map(([x, y]) => [x + dx, y + dy]) : null,
  };
}

// Everything that can be drawn this frame: tracked objects, plus highlighted
// text blocks and regions, which carry their own box.
function elements(nowS) {
  const out = [];
  const tracked = new Set();
  for (const obj of state.objects.list) {
    tracked.add(obj.ref);
    if (obj.status === 'tentative') continue; // only confirmed objects glow
    out.push({ ref: obj.ref, lost: obj.status === 'lost', ...extrapolated(obj, nowS) });
  }
  for (const [ref, hl] of state.highlights) {
    if (!tracked.has(ref) && hl.box) out.push({ ref, lost: false, box: hl.box, outline: null });
  }
  return out;
}

// Bars whose height follows the voice loudness, tallest in the middle.
function drawWaves(g, cx, cy, width, level, color, alpha, t) {
  const gap = width / VOICE_BARS;
  for (let i = 0; i < VOICE_BARS; i++) {
    const shape = Math.pow(Math.sin((Math.PI * (i + 0.5)) / VOICE_BARS), 0.8);
    const wobble = 0.5 + 0.5 * Math.sin(t * 7 + i * 0.75) * Math.sin(t * 3.1 + i * 1.3);
    const h = 4 + level * 40 * shape * (0.45 + wobble);
    const x = cx - width / 2 + i * gap + gap * 0.2;
    g.roundRect(x, cy - h / 2, gap * 0.6, h, gap * 0.3).fill({ color, alpha });
  }
}

function drawThinking(g, cx, cy, color, alpha, t) {
  for (let i = 0; i < 3; i++) {
    const bounce = Math.max(0, Math.sin(t * 6 - i * 0.9));
    g.circle(cx + (i - 1) * 18, cy - bounce * 8, 5).fill({ color, alpha: alpha * (0.5 + 0.5 * bounce) });
  }
}

async function main() {
  const app = new PIXI.Application();
  await app.init({ resizeTo: window, backgroundAlpha: 0, antialias: true });
  document.body.appendChild(app.canvas);

  const dimLayer = new PIXI.Graphics();
  const glowLayer = new PIXI.Graphics();
  glowLayer.filters = [new PIXI.BlurFilter({ strength: GLOW_BLUR, quality: 4 })];
  const outlineLayer = new PIXI.Graphics();
  const voiceGlow = new PIXI.Graphics();
  voiceGlow.filters = [new PIXI.BlurFilter({ strength: 18, quality: 4 })];
  const voiceLayer = new PIXI.Graphics();
  app.stage.addChild(dimLayer, glowLayer, outlineLayer, voiceGlow, voiceLayer);

  window.gvision.onMessage(handle);

  app.ticker.add((ticker) => {
    const W = app.screen.width;
    const H = app.screen.height;
    const nowS = Date.now() / 1000;
    const step = ticker.deltaMS / DIM_FADE_MS;
    const target = state.dim.on ? state.dim.strength : 0;
    state.dim.alpha += Math.sign(target - state.dim.alpha) * Math.min(Math.abs(target - state.dim.alpha), step * Math.max(state.dim.strength, 0.1));
    const pulse = 1 - PULSE_DEPTH * (0.5 + 0.5 * Math.sin(2 * Math.PI * PULSE_HZ * nowS));

    const els = elements(nowS).map((el) => ({
      ...el,
      px: { x: el.box.x * W, y: el.box.y * H, w: el.box.w * W, h: el.box.h * H },
      poly: el.outline ? el.outline.flatMap(([x, y]) => [x * W, y * H]) : null,
    }));
    // Objects are drawn by their mask contour. Without one (rare: the tracker
    // carries the last contour over), an ellipse rather than a box; text and
    // regions keep their rounded box.
    const shape = (g, { ref, px, poly }) => {
      if (poly) return g.poly(poly, true);
      if (ref.startsWith('obj:')) return g.ellipse(px.x + px.w / 2, px.y + px.h / 2, px.w / 2, px.h / 2);
      return g.roundRect(px.x, px.y, px.w, px.h, 6);
    };

    // Dim layer with cut-outs following the focused elements' contours.
    dimLayer.clear();
    if (state.dim.alpha > 0.001) {
      dimLayer.rect(0, 0, W, H).fill({ color: 0x000000, alpha: state.dim.alpha });
      for (const el of els) {
        if (state.focus.has(el.ref)) shape(dimLayer, el).cut();
      }
    }

    glowLayer.clear();
    outlineLayer.clear();
    for (const el of els) {
      const { ref, lost } = el;
      const hl = state.highlights.get(ref);
      if (!hl && !SHOW_ALL_TRACKS) continue;
      const focused = state.focus.has(ref);
      const color = hl ? COLORS[hl.color_role] : SUBTLE;
      const fade = lost ? LOST_ALPHA : 1;

      let alpha = 0.25 * fade; // unhighlighted debug track
      if (hl) {
        alpha = (focused || state.focus.size === 0 ? 1 : 0.7) * fade;
        const glowAlpha = (focused ? pulse : 0.5) * fade;
        shape(glowLayer, el).stroke({ width: GLOW_WIDTH_PX, color, alpha: glowAlpha, join: 'round' });
        shape(outlineLayer, el).fill({ color, alpha: FILL_ALPHA * (focused ? pulse : 0.7) * fade });
      }
      const core = focused ? CORE_WIDTH_PX.focused : CORE_WIDTH_PX.mentioned;
      shape(outlineLayer, el).stroke({ width: core + 4, color: 0x000000, alpha: alpha * 0.7, join: 'round' });
      shape(outlineLayer, el).stroke({ width: core, color, alpha, join: 'round' });
    }

    drawVoice(W, H, nowS, ticker.deltaMS);
  });

  // Push-to-talk indicator at the bottom center, no text: your voice is cyan,
  // G-VISION's answer is gold, with a soft glow along the screen edge.
  function drawVoice(W, H, t, dtMs) {
    const v = state.voice;
    const active = v.state !== 'idle';
    if (active) v.shown = v.state;
    v.alpha += Math.sign((active ? 1 : 0) - v.alpha) * Math.min(Math.abs((active ? 1 : 0) - v.alpha), dtMs / VOICE_FADE_MS);
    // No level (typed request, or no speech output): a gentle idle wave.
    const target = v.level ?? (v.shown === 'thinking' ? 0 : 0.25);
    v.smooth += (target - v.smooth) * Math.min(1, (dtMs / 1000) * 14);
    voiceGlow.clear();
    voiceLayer.clear();
    if (v.alpha < 0.01) return;

    const color = VOICE_COLORS[v.shown];
    const cx = W / 2;
    const cy = H * (1 - VOICE_PILL.bottom);
    const { w, h } = VOICE_PILL;
    // Edge glow: brighter when louder, so it reads from the corner of the eye.
    const edge = 0.25 + 0.35 * v.smooth;
    voiceGlow.rect(cx - W * 0.3, H - 10, W * 0.6, 40).fill({ color, alpha: edge * v.alpha });
    voiceGlow.roundRect(cx - w / 2, cy - h / 2, w, h, h / 2).stroke({ width: 10, color, alpha: 0.45 * v.alpha });
    voiceLayer.roundRect(cx - w / 2, cy - h / 2, w, h, h / 2).fill({ color: 0x0b0d12, alpha: 0.72 * v.alpha });
    voiceLayer.roundRect(cx - w / 2, cy - h / 2, w, h, h / 2).stroke({ width: 1.5, color, alpha: 0.8 * v.alpha });
    if (v.shown === 'thinking') drawThinking(voiceLayer, cx, cy, color, v.alpha, t);
    else drawWaves(voiceLayer, cx, cy, w - 48, v.smooth, color, v.alpha, t);
  }
}

main();
