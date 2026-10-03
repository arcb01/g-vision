// Transparent overlay: draws tracked objects, semantic highlights and the
// spotlight dim layer from bridge messages (plan sections 5.5 and 6).
'use strict';

const COLORS = { target: 0xffc83d, danger: 0xff4d4d, info: 0x3de0ff };
const SUBTLE = 0xffffff;
const DIM_FADE_MS = 200;
const MAX_EXTRAPOLATION_S = 0.25;
// Developer aid until the debug overlay exists: show unhighlighted tracks faintly.
const SHOW_ALL_TRACKS = true;

const state = {
  objects: { frameTs: 0, list: [] },
  highlights: new Map(), // ref -> {style, color_role, uncertain}
  focus: new Set(),
  dim: { on: false, strength: 0.45, alpha: 0 },
};

function handle(msg) {
  switch (msg.type) {
    case 'objects':
      state.objects = { frameTs: msg.frame_ts, list: msg.objects };
      break;
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

// Shift boxes by tracker velocity to hide display delay (plan 5.5).
function extrapolated(obj, nowS) {
  const dt = Math.min(Math.max(nowS - state.objects.frameTs, 0), MAX_EXTRAPOLATION_S);
  const [vx, vy] = obj.velocity;
  return { x: obj.box.x + vx * dt, y: obj.box.y + vy * dt, w: obj.box.w, h: obj.box.h };
}

async function main() {
  const app = new PIXI.Application();
  await app.init({ resizeTo: window, backgroundAlpha: 0, antialias: true });
  document.body.appendChild(app.canvas);

  const dimLayer = new PIXI.Graphics();
  const outlineLayer = new PIXI.Graphics();
  const labelLayer = new PIXI.Container();
  app.stage.addChild(dimLayer, outlineLayer, labelLayer);
  const labels = new Map(); // ref -> PIXI.Text

  window.gvision.onMessage(handle);

  app.ticker.add((ticker) => {
    const W = app.screen.width;
    const H = app.screen.height;
    const nowS = Date.now() / 1000;
    const step = ticker.deltaMS / DIM_FADE_MS;
    const target = state.dim.on ? state.dim.strength : 0;
    state.dim.alpha += Math.sign(target - state.dim.alpha) * Math.min(Math.abs(target - state.dim.alpha), step * state.dim.strength);

    const boxes = state.objects.list.map((obj) => {
      const b = extrapolated(obj, nowS);
      return { obj, px: { x: b.x * W, y: b.y * H, w: b.w * W, h: b.h * H } };
    });

    // Dim layer with cut-outs around focused elements (box-shaped for now).
    dimLayer.clear();
    if (state.dim.alpha > 0.001) {
      dimLayer.rect(0, 0, W, H).fill({ color: 0x000000, alpha: state.dim.alpha });
      for (const { obj, px } of boxes) {
        if (state.focus.has(obj.ref)) dimLayer.roundRect(px.x - 12, px.y - 12, px.w + 24, px.h + 24, 12).cut();
      }
    }

    // Layered outline: dark outer edge + bright core (plan 6.3).
    outlineLayer.clear();
    const seen = new Set();
    for (const { obj, px } of boxes) {
      if (obj.status === 'tentative') continue; // only confirmed objects glow
      const hl = state.highlights.get(obj.ref);
      if (!hl && !SHOW_ALL_TRACKS) continue;
      const focused = state.focus.has(obj.ref);
      const color = hl ? COLORS[hl.color_role] : SUBTLE;
      const alpha = hl ? (focused || state.focus.size === 0 ? 1 : 0.55) : 0.25;
      const core = focused ? 3 : 2;
      outlineLayer.roundRect(px.x, px.y, px.w, px.h, 6).stroke({ width: core + 4, color: 0x000000, alpha: alpha * 0.6 });
      outlineLayer.roundRect(px.x, px.y, px.w, px.h, 6).stroke({ width: core, color, alpha });

      let label = labels.get(obj.ref);
      if (!label) {
        label = new PIXI.Text({ text: obj.label, style: { fontFamily: 'Segoe UI, sans-serif', fontSize: 14, fill: 0xffffff, stroke: { color: 0x000000, width: 3 } } });
        labelLayer.addChild(label);
        labels.set(obj.ref, label);
      }
      label.text = obj.label;
      label.alpha = alpha;
      label.position.set(px.x, px.y - 20);
      seen.add(obj.ref);
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
