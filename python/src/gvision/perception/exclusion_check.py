"""Check that the overlay is excluded from screen capture (plan 5.1, 5.5).

If the detector could see the overlay, it would detect its own glows. The
app sets content protection on the overlay window; this check proves it
works on the actual machine: with the app running, it captures the screen,
makes the overlay draw a heavy dim layer and a gold box, captures again and
looks for either in the second frame.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from gvision.bridge import Bridge
from gvision.perception.capture import FrameSource
from gvision.protocol import Box, ClearMsg, DimMsg, FocusMsg, HighlightMsg

log = logging.getLogger(__name__)

TEST_REF = "region:exclusion_test"
TEST_BOX = Box(x=0.3, y=0.3, w=0.4, h=0.4)
TEST_DIM = 0.9
GOLD_BGR = np.array([61, 200, 255])  # overlay's "target" color, 0xffc83d
GOLD_TOLERANCE = 40


@dataclass
class ExclusionResult:
    brightness_ratio: float | None
    """Mean brightness with the dim layer on / before; ~0.1 if it was captured."""
    gold_before: float
    gold_after: float
    """Fraction of pixels on the test box's edge that are overlay gold."""

    @property
    def overlay_captured(self) -> bool:
        dimmed = self.brightness_ratio is not None and self.brightness_ratio < 0.5
        return dimmed or self.gold_after - self.gold_before > 0.2


def _edge_pixels(image: np.ndarray, box: Box) -> np.ndarray:
    h, w = image.shape[:2]
    x0, x1 = int(box.x * w), int((box.x + box.w) * w)
    y0, y1 = int(box.y * h), int((box.y + box.h) * h)
    parts = []
    for d in (-1, 0, 1):
        parts += [image[y0 + d, x0:x1], image[y1 + d, x0:x1], image[y0:y1, x0 + d], image[y0:y1, x1 + d]]
    return np.concatenate(parts)


def _gold_fraction(image: np.ndarray, box: Box) -> float:
    px = _edge_pixels(image, box).astype(int)
    return float(np.mean(np.all(np.abs(px - GOLD_BGR) <= GOLD_TOLERANCE, axis=1)))


def analyze(before: np.ndarray, after: np.ndarray, box: Box = TEST_BOX) -> ExclusionResult:
    mean_before = float(before.mean())
    ratio = float(after.mean()) / mean_before if mean_before > 15 else None  # too dark to tell
    return ExclusionResult(
        brightness_ratio=ratio,
        gold_before=_gold_fraction(before, box),
        gold_after=_gold_fraction(after, box),
    )


async def _grab(source: FrameSource) -> np.ndarray:
    for _ in range(50):
        frame = await asyncio.to_thread(source.latest)
        if frame is not None:
            return frame.image.copy()
        await asyncio.sleep(0.05)
    raise RuntimeError("no frame from the capture source")


async def run_exclusion_check(bridge: Bridge, stop: asyncio.Event, source: FrameSource, timeout: float = 30.0) -> bool:
    """Returns True if the overlay stayed out of the captured frames."""
    try:
        log.info("waiting for the app to connect (run `npm start` in app/)")
        for _ in range(int(timeout * 10)):
            if bridge.connected:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("the app did not connect")
        await asyncio.sleep(1.5)  # let the overlay window finish loading
        bridge.send(ClearMsg(reason="exclusion check"))
        await asyncio.sleep(0.5)
        before = await _grab(source)

        bridge.send(HighlightMsg(ref=TEST_REF, color_role="target", box=TEST_BOX))
        bridge.send(FocusMsg(refs=[]))
        bridge.send(DimMsg(on=True, strength=TEST_DIM))
        await asyncio.sleep(1.0)  # dim fades in over ~200 ms
        after = await _grab(source)
        bridge.send(DimMsg(on=False, strength=TEST_DIM))
        bridge.send(ClearMsg(reason="exclusion check done"))
        await asyncio.sleep(0.3)  # let the clear reach the app before exiting

        result = analyze(before, after)
        ratio = "n/a (screen too dark)" if result.brightness_ratio is None else f"{result.brightness_ratio:.2f}"
        log.info(
            "brightness after/before: %s, gold on test box edge: %.0f%% -> %.0f%%",
            ratio, 100 * result.gold_before, 100 * result.gold_after,
        )
        if result.overlay_captured:
            log.error("FAIL: the overlay shows up in captured frames")
        else:
            log.info("PASS: the overlay is excluded from capture (you should have seen it dim the screen)")
        return not result.overlay_captured
    finally:
        source.close()
        stop.set()
