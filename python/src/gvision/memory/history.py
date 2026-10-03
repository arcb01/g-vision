"""Frame history (plan 7.2): the last ~60 s of the screen at ~2 fps.

Frames are downscaled to 640 px wide and JPEG-compressed on a worker
thread, so the whole minute is a few MB of host memory and costs the game
nothing on the GPU. Each frame also keeps a 32x18 grayscale thumbnail, and
its difference to the previous one is the frame's ``change`` score: hits,
flashes and sudden camera moves score high, which is how ``pick`` finds
the moments worth showing Qwen for "what just hit me?".
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gvision.perception.capture import Frame

log = logging.getLogger(__name__)

WIDTH = 640
"""Stored width: ~220 image tokens per frame for Qwen, ~40 KB as JPEG."""
JPEG_QUALITY = 70
THUMB = (32, 18)


@dataclass
class StoredFrame:
    ts: float
    """Unix capture time, as on ``Frame``."""
    jpeg: bytes
    thumb: np.ndarray
    change: float = 0.0
    """Mean absolute thumbnail difference to the previous frame, 0-1."""

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.jpeg).decode()


Prepare = Callable[[np.ndarray], tuple[bytes, np.ndarray]]


def prepare(image: np.ndarray) -> tuple[bytes, np.ndarray]:
    """BGR frame -> (JPEG of the downscaled frame, grayscale thumbnail)."""
    import cv2

    h, w = image.shape[:2]
    if w > WIDTH:
        image = cv2.resize(image, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    thumb = cv2.cvtColor(cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return jpeg.tobytes(), thumb


class FrameHistory:
    def __init__(self, seconds: float = 60.0, fps: float = 2.0, prepare_fn: Prepare = prepare) -> None:
        self.seconds = seconds
        self.interval = 1.0 / fps
        self.frames: deque[StoredFrame] = deque(maxlen=max(1, round(seconds * fps)))
        self._prepare = prepare_fn
        self._last_offer = 0.0
        self._busy = False
        self.listeners: list[Callable[[StoredFrame], None]] = []
        """Called on the loop for every stored frame (the event log hooks in here)."""

    # --- writer: the live pipeline, once per captured frame ----------------

    def offer(self, frame: Frame) -> None:
        """Keep the frame if one is due; never blocks the caller's loop."""
        if self._busy or frame.ts - self._last_offer < self.interval:
            return
        self._last_offer = frame.ts
        self._busy = True
        task = asyncio.get_running_loop().run_in_executor(None, self._prepare, frame.image)
        task.add_done_callback(lambda f: self._store(frame.ts, f))

    def _store(self, ts: float, done: asyncio.Future) -> None:
        self._busy = False
        if done.cancelled() or done.exception():
            log.warning("frame history: could not store a frame: %s", None if done.cancelled() else done.exception())
            return
        jpeg, thumb = done.result()
        self.add(StoredFrame(ts, jpeg, thumb))

    def add(self, frame: StoredFrame) -> None:
        if self.frames:
            prev = self.frames[-1].thumb
            if prev.shape == frame.thumb.shape:
                frame.change = float(np.abs(frame.thumb.astype(np.int16) - prev.astype(np.int16)).mean() / 255)
        self.frames.append(frame)
        for listen in self.listeners:
            listen(frame)

    # --- readers: the look tool and the narrator ---------------------------

    def latest(self) -> StoredFrame | None:
        return self.frames[-1] if self.frames else None

    def window(self, seconds: float, now: float | None = None) -> list[StoredFrame]:
        now = time.time() if now is None else now
        return [f for f in self.frames if f.ts >= now - seconds]

    def pick(self, seconds: float, k: int = 4, now: float | None = None) -> list[StoredFrame]:
        """Up to ``k`` frames from the last ``seconds``, oldest first.

        Always the newest frame, plus the ones with the biggest change
        (where something happened), and never two neighbours when others
        are left, so the frames cover the moment rather than repeat it.
        """
        frames = self.window(seconds, now)
        if not frames:
            latest = self.latest()
            return [latest] if latest else []
        chosen = {len(frames) - 1}
        ranked = sorted(range(len(frames) - 1), key=lambda i: frames[i].change, reverse=True)
        for spread in (True, False):
            for i in ranked:
                if len(chosen) >= k:
                    break
                if i not in chosen and (not spread or all(abs(i - j) > 1 for j in chosen)):
                    chosen.add(i)
        return [frames[i] for i in sorted(chosen)]

    def max_change_since(self, ts: float) -> float:
        return max((f.change for f in self.frames if f.ts > ts), default=0.0)
