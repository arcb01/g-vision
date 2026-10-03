// Transparent overlay: draws tracked objects, semantic highlights and the
// spotlight dim layer from bridge messages (plan sections 5.5 and 6).
'use strict';

const COLORS = { target: 0xffc83d, danger: 0xff4d4d, info: 0x3de0ff };
const SUBTLE = 0xffffff;
const DIM_FADE_MS = 200;
const MAX_EXTRAPOLATION_S = 0.25;
const CUTOUT_PAD_PX = 12;
// Glow: a blurred wide stroke under a dark edge and a bright core (plan 6.3).
const GLOW_WIDTH_PX = 14;
const GLOW_BLUR = 10;
const CORE_WIDTH_PX = { focused: 4, mentioned: 3 };
const PULSE_HZ = 1; // slow, well below flashing (plan 6.3)
const PULSE_DEPTH = 0.2;
// Developer aid until the debug overlay exists: show unhighlighted tracks faintly.
const SHOW_ALL_TRACKS = true;
const LOST_ALPHA = 0.4; // lost tracks stay at their last position, faded (plan 5.4)

const state = {
  objects: { frameTs: 0, list: [] },
  highlights: new Map(), // ref -> highlight message (with box for text:/region: refs)
  focus: new Set(),
  dim: { on: false, strength: 0.6, alpha: 0 },
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
    out.push({ ref: obj.ref, label: obj.label, lost: obj.status === 'lost', ...extrapolated(obj, nowS) });
  }
  for (const [ref, hl] of state.highlights) {
    if (!tracked.has(ref) && hl.box) out.push({ ref, label: null, lost: false, box: hl.box, outline: null });
  }
  return out;
}

async function main() {
  const app = new PIXI.Application();
  await app.init({ resizeTo: window, backgroundAlpha: 0, antialias: true });
  document.body.appendChild(app.canvas);

  const dimLayer = new PIXI.Graphics();
  const glowLayer = new PIXI.Graphics();
  glowLayer.filters = [new PIXI.BlurFilter({ strength: GLOW_BLUR, quality: 4 })];
  const outlineLayer = new PIXI.Graphics();
  const labelLayer = new PIXI.Container();
  app.stage.addChild(dimLayer, glowLayer, outlineLayer, labelLayer);
  const labels = new Map(); // ref -> PIXI.Text

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
    // Mask outline when the detector gave one, otherwise the rounded box.
    const shape = (g, { px, poly }) => (poly ? g.poly(poly, true) : g.roundRect(px.x, px.y, px.w, px.h, 6));

    // Dim layer with cut-outs around focused elements (box-shaped for now).
    dimLayer.clear();
    if (state.dim.alpha > 0.001) {
      dimLayer.rect(0, 0, W, H).fill({ color: 0x000000, alpha: state.dim.alpha });
      for (const { ref, px } of els) {
        if (state.focus.has(ref)) {
          dimLayer.roundRect(px.x - CUTOUT_PAD_PX, px.y - CUTOUT_PAD_PX, px.w + 2 * CUTOUT_PAD_PX, px.h + 2 * CUTOUT_PAD_PX, 12).cut();
        }
      }
    }

    glowLayer.clear();
    outlineLayer.clear();
    const seen = new Set();
    for (const el of els) {
      const { ref, label: text, px, lost } = el;
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
        if (focused) shape(glowLayer, el).fill({ color, alpha: 0.12 * pulse * fade });
      }
      const core = focused ? CORE_WIDTH_PX.focused : CORE_WIDTH_PX.mentioned;
      shape(outlineLayer, el).stroke({ width: core + 4, color: 0x000000, alpha: alpha * 0.7, join: 'round' });
      shape(outlineLayer, el).stroke({ width: core, color, alpha, join: 'round' });

      if (text == null) continue;
      let label = labels.get(ref);
      if (!label) {
        label = new PIXI.Text({ text, style: { fontFamily: 'Segoe UI, sans-serif', fontSize: 14, fill: 0xffffff, stroke: { color: 0x000000, width: 3 } } });
        labelLayer.addChild(label);
        labels.set(ref, label);
      }
      label.text = text;
      label.alpha = alpha;
      label.position.set(px.x, px.y - 22);
      seen.add(ref);
    }
    for (const [ref, label] of labels) {
      if (!seen.has(ref)) {
        label.destroy();
        labels.delete(ref);
      }
    }
  });
}

main();
