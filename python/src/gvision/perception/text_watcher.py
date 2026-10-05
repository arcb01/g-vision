"""Text watcher with automatic zone discovery (plan 7.1).

A background loop at ~2 Hz that keeps a log of the text on screen, so a
question about a sign, a quest prompt or a menu is answered from text that
was already read:

1. **Cheap change check.** A 320 px sample of each frame is compared tile by
   tile (16 x 9) with the last one; unchanged tiles are skipped.
2. **Detect** text lines with RapidOCR's detection stage only, on the
   full-width band of rows that changed. The whole screen is covered at
   most once a second; learned zones are checked on every tick.
3. **Track** lines like objects (box overlap), each with an ID, and
   **recognize only new or changed lines** on full-resolution crops.
4. **UI or world:** a line that stays put while the scene moves is UI; one
   that moves with the scene belongs to the world.
5. **Group lines into blocks** (``text:<n>`` refs) by proximity, size and
   alignment.
6. **Static text** ("HP", "AMMO") unchanged for 30 s is flagged and sorted
   last.
7. **Learned zones:** cells where text keeps appearing or changing (chat,
   notifications, subtitles) become zones that are checked first.
8. A **game profile** per foreground app (heat map + static text) is saved
   under ``data/profiles/`` so the next session starts with its zones.
9. Blocks that change or disappear go to a **text log** kept for 5 minutes.

The OCR runs on the CPU in a worker thread; the bookkeeping runs on the
asyncio loop, so tools read the blocks without locking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from gvision.bridge import Bridge
from gvision.perception.capture import Frame, FrameSource
from gvision.perception.ocr import OcrEngine
from gvision.protocol import Box, ColorRole, ConfigChangedMsg, DimMsg, FocusMsg, HighlightMsg, Message
from gvision.world import where

log = logging.getLogger(__name__)

GRID_W, GRID_H = 16, 9
RATE_HZ = 2.0
FULL_PERIOD_S = 1.0
"""Text outside learned zones is detected at most this often (plan: 1-2 Hz)."""
TILE_DIFF = 2.0
"""Mean change in gray level (0-255) that marks a screen tile as changed."""
CROP_DIFF = 8.0
"""Same for a line's crop: above it, the line is read again."""
MOVING_SCENE = 0.3
"""Share of changed tiles that means the camera is moving."""
MIN_SCORE = 0.6
"""Recognitions below this are treated as "not text" (textures, icons)."""
GONE_AFTER = 2
"""Detection passes a line can be missed before it counts as gone."""
STATIC_AFTER_S = 30.0
ZONE_HEAT = 4.0
"""Appearances/changes in a cell before it becomes a learned zone."""
LOG_S = 300.0
PROFILE_SAVE_S = 30.0
APP_CHECK_S = 5.0
HIGHLIGHT_PAD = 0.004

# Our own windows and shells: switching to them keeps the game's profile.
OWN_APPS = {"electron", "g-vision", "python", "pythonw", "windowsterminal", "cmd", "powershell",
            "pwsh", "bash", "mintty", "conhost", "explorer"}

NormBox = tuple[float, float, float, float]
"""x0, y0, x1, y1 normalized to 0..1."""


@dataclass
class TextLine:
    id: int
    box: NormBox
    text: str = ""
    score: float = 0.0
    sig: np.ndarray | None = None
    """Tiny sample of the crop, to notice when the text under the box changes."""
    seen: float = 0.0
    changed: float = 0.0
    missed: int = 0
    stayed: int = 0
    """Passes where the scene moved but this line stayed put."""
    moved: int = 0

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class TextBlock:
    ref: str
    box: Box
    text: str
    confidence: float
    kind: str
    """"ui", "world" or "unknown" (not enough camera movement seen yet)."""
    static: bool
    in_zone: bool
    appeared: float
    changed: float
    gone: float | None = None
    crop: np.ndarray | None = field(default=None, repr=False)
    """Full-resolution crop, kept for re-reading it later."""

    def info(self, now: float | None = None) -> dict:
        out = {"ref": self.ref, "text": self.text[:200], "where": where(self.box)}
        if self.kind != "unknown":
            out["kind"] = self.kind
        if self.static:
            out["static"] = True
        if now is not None:
            out["seconds_ago"] = round(now - (self.gone or self.appeared))
            if self.gone is None:
                out["on_screen"] = True
        return out


# --- pure helpers ------------------------------------------------------------


def thumbnail(image: np.ndarray) -> np.ndarray:
    """Gray-ish 320 px sample of a frame; ~1 ms for 1440p."""
    step = max(1, image.shape[1] // 320)
    return image[::step, ::step].mean(axis=2, dtype=np.float32)


def tile_changes(prev: np.ndarray | None, cur: np.ndarray) -> np.ndarray:
    """GRID_H x GRID_W booleans: which tiles changed."""
    if prev is None or prev.shape != cur.shape:
        return np.ones((GRID_H, GRID_W), bool)
    diff = np.abs(cur - prev)
    h, w = diff.shape
    ys = np.linspace(0, h, GRID_H + 1).astype(int)
    xs = np.linspace(0, w, GRID_W + 1).astype(int)
    sums = np.add.reduceat(np.add.reduceat(diff, ys[:-1], axis=0), xs[:-1], axis=1)
    sizes = np.outer(np.diff(ys), np.diff(xs))
    return sums / np.maximum(sizes, 1) > TILE_DIFF


def crop_sig(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    return crop[np.linspace(0, h - 1, 8).astype(int)][:, np.linspace(0, w - 1, 48).astype(int)].mean(
        axis=2, dtype=np.float32)


def iou(a: NormBox, b: NormBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cells(box: NormBox) -> tuple[slice, slice]:
    """Grid rows and columns a box covers."""
    c0, c1 = int(box[0] * GRID_W), int(np.ceil(box[2] * GRID_W))
    r0, r1 = int(box[1] * GRID_H), int(np.ceil(box[3] * GRID_H))
    return slice(max(r0, 0), max(min(r1, GRID_H), r0 + 1)), slice(max(c0, 0), max(min(c1, GRID_W), c0 + 1))


def group_lines(lines: list[TextLine]) -> list[list[TextLine]]:
    """Lines -> blocks by proximity, size and alignment (plan 7.1 step 5)."""
    blocks: list[list[TextLine]] = []
    for ln in sorted(lines, key=lambda ln: (ln.box[1], ln.box[0])):
        for block in blocks:
            if any(_same_block(ln, other) for other in block):
                block.append(ln)
                break
        else:
            blocks.append([ln])
    return blocks


def _same_block(a: TextLine, b: TextLine) -> bool:
    h = min(a.h, b.h)
    if h <= 0 or not 0.6 <= a.h / b.h <= 1.67:
        return False
    v_overlap = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
    if v_overlap > 0.5 * h:  # same row: words split apart by the detector
        gap = max(a.box[0], b.box[0]) - min(a.box[2], b.box[2])
        return gap < 1.5 * h
    gap = max(a.box[1], b.box[1]) - min(a.box[3], b.box[3])
    aligned = (min(a.box[2], b.box[2]) > max(a.box[0], b.box[0])  # overlap horizontally
               or abs(a.box[0] - b.box[0]) < 2 * h)
    return gap < 0.8 * h and aligned


def block_text(lines: list[TextLine]) -> str:
    rows: list[list[TextLine]] = []
    for ln in sorted(lines, key=lambda ln: (ln.box[1], ln.box[0])):
        if rows and min(ln.box[3], rows[-1][0].box[3]) - max(ln.box[1], rows[-1][0].box[1]) > 0.5 * ln.h:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return "\n".join(" ".join(ln.text for ln in sorted(r, key=lambda ln: ln.box[0])) for r in rows)


_WORD = re.compile(r"[a-z0-9]{3,}")
_POSITIONS = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom", "left": "left",
              "right": "right", "center": "center", "middle": "center", "centre": "center"}


def words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def matches_where(box: Box, phrase: str | None) -> bool:
    """'top right' matches a block whose position in words has top and right."""
    wanted = {_POSITIONS[w] for w in re.findall(r"[a-z]+", (phrase or "").lower()) if w in _POSITIONS}
    have = {_POSITIONS.get(w, w) for w in where(box).split()}
    return wanted <= have


def foreground_app() -> str | None:
    """Executable name of the focused window ('javaw' for Minecraft), Windows only."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return re.sub(r"[^a-z0-9_.-]", "_", Path(buf.value).stem.lower()) or None
    finally:
        kernel32.CloseHandle(handle)


# --- the watcher -------------------------------------------------------------


class TextWatcher:
    def __init__(
        self, bridge: Bridge | None, source: FrameSource | None, engine: OcrEngine,
        profiles_dir: str | Path | None = "data/profiles", rate_hz: float = RATE_HZ,
    ) -> None:
        self.bridge = bridge
        self.source = source
        self.engine = engine
        self.rate_hz = rate_hz
        self.profiles_dir = Path(profiles_dir) if profiles_dir else None
        self.lines: list[TextLine] = []
        self.blocks: dict[str, TextBlock] = {}
        self.log: deque[TextBlock] = deque(maxlen=500)
        self.heat = np.zeros((GRID_H, GRID_W), np.float32)
        self.static_texts: set[str] = set()
        self.app = "default"
        self.dim_strength = DimMsg.model_fields["strength"].default
        self._line_ids = 0
        self._block_ids = 0
        self._line_block: dict[int, str] = {}
        self._thumb: np.ndarray | None = None
        self._pending = np.ones((GRID_H, GRID_W), bool)
        self._last_full = 0.0
        self._profile_dirty = False
        self._lock = asyncio.Lock()
        self.timing_ms: dict[str, float] = {}
        if bridge:
            bridge.on_message(self._on_message)

    async def _on_message(self, msg: Message) -> None:
        if isinstance(msg, ConfigChangedMsg):
            strength = msg.changes.get("visual_effects.dim_strength")
            if isinstance(strength, (int, float)) and 0 <= strength <= 1:
                self.dim_strength = float(strength)

    # --- zones and profile ------------------------------------------------

    @property
    def zones(self) -> np.ndarray:
        return self.heat >= ZONE_HEAT

    def _profile_path(self, app: str) -> Path | None:
        return self.profiles_dir / f"{app}.json" if self.profiles_dir else None

    def load_profile(self, app: str) -> None:
        self.app = app
        self.heat = np.zeros((GRID_H, GRID_W), np.float32)
        self.static_texts = set()
        path = self._profile_path(app)
        if path and path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                heat = np.asarray(data.get("heat"), np.float32)
                if heat.shape == self.heat.shape:
                    self.heat = heat
                self.static_texts = set(data.get("static", []))
                log.info("text profile %s: %d zone cells, %d static texts", app, int(self.zones.sum()),
                         len(self.static_texts))
            except (OSError, ValueError) as e:
                log.warning("ignoring unreadable text profile %s: %s", path, e)
        self._profile_dirty = False

    def save_profile(self) -> None:
        path = self._profile_path(self.app)
        if not path or not self._profile_dirty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"app": self.app, "grid": [GRID_W, GRID_H], "heat": np.round(self.heat, 1).tolist(),
                "static": sorted(self.static_texts)[:300]}
        path.write_text(json.dumps(data, indent=1), "utf-8")
        self._profile_dirty = False

    def _switch_app(self, app: str | None) -> None:
        if not app or app == self.app or app in OWN_APPS:
            return
        self.save_profile()
        self.load_profile(app)

    # --- one pass ---------------------------------------------------------

    async def tick(self, frame: Frame, force: bool = False) -> None:
        """Process one frame: detect where it changed, read new lines, update blocks."""
        now = frame.ts
        image = frame.image
        H, W = image.shape[:2]
        thumb = thumbnail(image)
        changed = tile_changes(self._thumb, thumb)
        self._thumb = thumb
        self._pending |= changed
        moving = bool(changed.mean() >= MOVING_SCENE)

        full = force or now - self._last_full >= FULL_PERIOD_S
        rows = (self._pending if full else self._pending & self.zones).any(axis=1)
        if full:
            self._last_full = now
        if not rows.any():
            self._observe_still(moving, changed, band=None)
            self._rebuild_blocks(now, image)
            return

        # Full-width band of the changed rows, one tile of margin: text lines
        # are horizontal, so a band never cuts one in half sideways.
        r = np.flatnonzero(rows)
        r0, r1 = max(int(r[0]) - 1, 0), min(int(r[-1]) + 2, GRID_H)
        y0, y1 = r0 * H // GRID_H, r1 * H // GRID_H
        self._pending[r0:r1] = False
        t0 = time.perf_counter()
        found = await asyncio.to_thread(self.engine.detect, image[y0:y1])
        self.timing_ms["detect"] = (time.perf_counter() - t0) * 1000
        band = (y0 / H, y1 / H)
        edge = 2 / H
        boxes: list[NormBox] = []
        for bx0, by0, bx1, by1 in found:
            box = (bx0 / W, (by0 + y0) / H, bx1 / W, (by1 + y0) / H)
            # Boxes touching a band edge inside the screen may be cut; the
            # next pass with margin around them reads them whole.
            if (box[1] <= band[0] + edge and y0 > 0) or (box[3] >= band[1] - edge and y1 < H):
                continue
            boxes.append(box)

        self._observe_still(moving, changed, band)
        to_read = self._match(boxes, band, now, moving, image)
        if to_read:
            crops = [self._crop(image, ln.box) for ln in to_read]
            t0 = time.perf_counter()
            results = await asyncio.to_thread(self.engine.recognize, crops)
            self.timing_ms["recognize"] = (time.perf_counter() - t0) * 1000
            for ln, crop, (text, score) in zip(to_read, crops, results):
                ln.sig = crop_sig(crop)
                text = " ".join(text.split()) if score >= MIN_SCORE else ""
                if text != ln.text:
                    ln.text, ln.changed = text, now
                    if text:
                        self.heat[cells(ln.box)] += 1
                        self._profile_dirty = True
                ln.score = score
        self._rebuild_blocks(now, image)

    def _crop(self, image: np.ndarray, box: NormBox) -> np.ndarray:
        H, W = image.shape[:2]
        pad = 0.15 * (box[3] - box[1])
        x0, x1 = int(max(box[0] - pad, 0) * W), int(np.ceil(min(box[2] + pad, 1) * W))
        y0, y1 = int(max(box[1] - pad, 0) * H), int(np.ceil(min(box[3] + pad, 1) * H))
        return image[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]

    def _observe_still(self, moving: bool, changed: np.ndarray, band: tuple[float, float] | None) -> None:
        """A line not re-detected whose pixels didn't change while the scene
        moved stayed put on screen: evidence that it is UI."""
        if not moving:
            return
        for ln in self.lines:
            outside = band is None or not (ln.box[1] >= band[0] and ln.box[3] <= band[1])
            if outside and not changed[cells(ln.box)].any():
                ln.stayed += 1

    def _match(
        self, boxes: list[NormBox], band: tuple[float, float], now: float, moving: bool, image: np.ndarray,
    ) -> list[TextLine]:
        """Track detected boxes onto known lines; return the lines to (re)read."""
        inside = [ln for ln in self.lines if ln.box[1] >= band[0] and ln.box[3] <= band[1]]
        pairs = sorted(
            ((iou(ln.box, b), i, j) for i, ln in enumerate(inside) for j, b in enumerate(boxes)), reverse=True)
        used_lines: set[int] = set()
        used_boxes: set[int] = set()
        to_read: list[TextLine] = []
        for score, i, j in pairs:
            if score < 0.3 or i in used_lines or j in used_boxes:
                continue
            used_lines.add(i)
            used_boxes.add(j)
            ln, box = inside[i], boxes[j]
            shift = max(abs((box[0] + box[2]) - (ln.box[0] + ln.box[2])),
                        abs((box[1] + box[3]) - (ln.box[1] + ln.box[3]))) / 2
            if shift > 0.004:
                ln.moved += 1
            elif moving:
                ln.stayed += 1
            ln.box, ln.seen, ln.missed = box, now, 0
            sig = crop_sig(self._crop(image, box))
            if ln.sig is None or ln.sig.shape != sig.shape or np.abs(sig - ln.sig).mean() > CROP_DIFF:
                to_read.append(ln)
        for i, ln in enumerate(inside):
            if i not in used_lines:
                ln.missed += 1
                self._pending[cells(ln.box)] = True  # look again before calling it gone
        gone = {id(ln) for ln in inside if ln.missed >= GONE_AFTER}
        self.lines = [ln for ln in self.lines if id(ln) not in gone]
        for j, box in enumerate(boxes):
            if j not in used_boxes:
                self._line_ids += 1
                ln = TextLine(self._line_ids, box, seen=now, changed=now)
                self.lines.append(ln)
                to_read.append(ln)
        return to_read

    def _rebuild_blocks(self, now: float, image: np.ndarray) -> None:
        readable = [ln for ln in self.lines if ln.text]
        groups = group_lines(readable)
        old = self.blocks
        new: dict[str, TextBlock] = {}
        line_block: dict[int, str] = {}
        for group in groups:
            # Keep the ref of the oldest block these lines belonged to.
            prev = sorted({self._line_block[ln.id] for ln in group if ln.id in self._line_block} - set(new),
                          key=lambda r: int(r.split(":")[1]))
            if prev:
                ref = prev[0]
            else:
                self._block_ids += 1
                ref = f"text:{self._block_ids}"
            x0 = min(ln.box[0] for ln in group)
            y0 = min(ln.box[1] for ln in group)
            x1 = max(ln.box[2] for ln in group)
            y1 = max(ln.box[3] for ln in group)
            box = Box(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
            text = block_text(group)
            changed = max(ln.changed for ln in group)
            stayed, moved = sum(ln.stayed for ln in group), sum(ln.moved for ln in group)
            kind = "world" if moved >= 2 and moved > stayed else "ui" if stayed >= 2 else "unknown"
            before = old.get(ref)
            appeared = before.appeared if before else now
            if before and before.text != text:
                self.log.append(replace(before, gone=now, crop=None))
            crop = before.crop if before and before.text == text else self._crop(image, (x0, y0, x1, y1)).copy()
            static = text.lower() in self.static_texts or (
                now - changed >= STATIC_AFTER_S and now - appeared >= STATIC_AFTER_S)
            if static and text.lower() not in self.static_texts:
                self.static_texts.add(text.lower())
                self._profile_dirty = True
            r, c = cells((x0, y0, x1, y1))
            new[ref] = TextBlock(
                ref=ref, box=box, text=text, confidence=min(ln.score for ln in group), kind=kind,
                static=static, in_zone=bool(self.zones[r, c].any()), appeared=appeared, changed=changed, crop=crop,
            )
            line_block.update({ln.id: ref for ln in group})
        for ref, block in old.items():
            if ref not in new:
                self.log.append(replace(block, gone=now))
        while self.log and now - (self.log[0].gone or now) > LOG_S:
            self.log.popleft()
        self.blocks = new
        self._line_block = line_block

    # --- reading, for the agent's tools ------------------------------------

    def visible(self) -> list[TextBlock]:
        """On-screen blocks, most useful first: changing, in a learned zone, newest."""
        return sorted(self.blocks.values(), key=lambda b: (b.static, not b.in_zone, -b.changed))

    def recent(self, seconds: float, now: float | None = None) -> list[TextBlock]:
        """Blocks seen in the last ``seconds``, gone or not, newest first."""
        now = time.time() if now is None else now
        gone = [b for b in self.log if now - (b.gone or now) <= seconds and not b.static]
        current = [b for b in self.blocks.values() if not b.static]
        return sorted(gone + current, key=lambda b: (b.gone is not None, -(b.gone or b.changed)))

    async def refresh(self) -> None:
        """Read the screen now (full-screen detection), before answering a question."""
        if not self.source:
            return
        async with self._lock:
            frame = await asyncio.to_thread(self.source.latest)
            if frame is not None:
                await self.tick(frame, force=True)

    def show(self, refs: list[str], color_role: ColorRole = "info",
             known: dict[str, TextBlock] | None = None) -> list[str]:
        """Outline the blocks an answer will read, all unlit under the dimmed
        screen; ``light`` then makes the one being read glow. ``known`` holds
        blocks as a tool read them, for ones the watcher has since lost (a
        toast sliding away). Returns the refs shown."""
        known = known or {}
        blocks = [b for b in (self.blocks.get(r) or known.get(r) for r in refs) if b]
        if not self.bridge or not blocks:
            return [b.ref for b in blocks]
        for b in blocks:
            p = HIGHLIGHT_PAD
            box = Box(x=b.box.x - p, y=b.box.y - p, w=b.box.w + 2 * p, h=b.box.h + 2 * p)
            self.bridge.send(HighlightMsg(ref=b.ref, color_role=color_role, box=box))
        self.bridge.send(FocusMsg(refs=[]))
        self.bridge.send(DimMsg(on=True, strength=self.dim_strength))
        return [b.ref for b in blocks]

    def light(self, ref: str, segment_id: int | None = None) -> None:
        """Make the block being read aloud glow; the others stay dimmed."""
        if self.bridge:
            self.bridge.send(FocusMsg(refs=[ref], segment_id=segment_id))
            self.bridge.send(DimMsg(on=True, strength=self.dim_strength))

    # --- background loop ---------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        period = 1.0 / self.rate_hz
        last_app = last_save = 0.0
        self.load_profile(self.app)
        try:
            while not stop.is_set():
                started = time.monotonic()
                if started - last_app >= APP_CHECK_S:
                    self._switch_app(await asyncio.to_thread(foreground_app))
                    last_app = started
                async with self._lock:
                    frame = await asyncio.to_thread(self.source.latest) if self.source else None
                    if frame is not None:
                        try:
                            await self.tick(frame)
                        except Exception:  # never take the backend down for one bad frame
                            log.exception("text watcher pass failed")
                if started - last_save >= PROFILE_SAVE_S:
                    self.save_profile()
                    last_save = started
                await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
        finally:
            self.save_profile()
