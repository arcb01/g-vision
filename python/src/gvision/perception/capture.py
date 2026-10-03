"""Frame sources (plan 5.1).

``ScreenSource`` captures the primary display with Desktop Duplication through
``dxcam`` (or its fork ``bettercam``) and only ever hands out the latest
frame, never a queue. The Electron overlay sets content protection
(``WDA_EXCLUDEFROMCAPTURE``), so it is not part of what is captured;
``python -m gvision --check-exclusion`` verifies that on a real machine.

``FileSource`` replays a video or a still image at real-time speed, for
development off Windows and for the test bench later.

Frames are BGR ``uint8`` arrays in host memory for now; keeping them on the
GPU is a later optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Frame:
    image: np.ndarray
    """BGR, H x W x 3."""
    ts: float
    """Unix time the frame was captured, used by the overlay to extrapolate."""


class FrameSource(Protocol):
    def latest(self) -> Frame | None: ...

    def close(self) -> None: ...


class ScreenSource:
    def __init__(self, output_idx: int = 0, target_fps: int = 60) -> None:
        try:
            import dxcam
        except ImportError:
            try:
                import bettercam as dxcam
            except ImportError as e:
                raise RuntimeError(
                    "screen capture needs Windows and dxcam: pip install -e \".[perception]\""
                ) from e
        self._camera = dxcam.create(output_idx=output_idx, output_color="BGR")
        if self._camera is None:
            raise RuntimeError(f"could not open display output {output_idx} for capture")
        # video_mode repeats the last frame when the screen is static, so
        # latest() never blocks for long.
        self._camera.start(target_fps=target_fps, video_mode=True)

    def latest(self) -> Frame | None:
        image = self._camera.get_latest_frame()
        if image is None:
            return None
        return Frame(image=image, ts=time.time())

    def close(self) -> None:
        self._camera.stop()


class FileSource:
    def __init__(self, path: str | Path, loop: bool = True) -> None:
        import cv2

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._still: np.ndarray | None = None
        self._cap = None
        if self.path.suffix.lower() in IMAGE_SUFFIXES:
            self._still = cv2.imread(str(self.path))
            if self._still is None:
                raise ValueError(f"cannot read image {self.path}")
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise ValueError(f"cannot open video {self.path}")
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._pos = -1
        self.loop = loop
        self._t0 = time.monotonic()
        self._image: np.ndarray | None = None

    def latest(self) -> Frame | None:
        if self._still is not None:
            return Frame(image=self._still, ts=time.time())
        # Skip ahead to the frame that should be on screen now, like a live
        # capture would: slow consumers drop frames instead of lagging.
        want = int((time.monotonic() - self._t0) * self._fps)
        while self._pos < want:
            ok = self._cap.grab()
            if not ok:
                if not self.loop:
                    return None
                self._cap.set(1, 0)  # cv2.CAP_PROP_POS_FRAMES
                self._t0 = time.monotonic()
                self._pos, want = -1, 0
                continue
            self._pos += 1
            self._image = None
        if self._image is None:
            ok, self._image = self._cap.retrieve()
            if not ok:
                return None
        return Frame(image=self._image, ts=time.time())

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def open_source(spec: str) -> FrameSource:
    """``screen`` or ``screen:<n>`` for a display, otherwise a file path."""
    if spec == "screen" or spec.startswith("screen:"):
        idx = int(spec.partition(":")[2] or 0)
        return ScreenSource(output_idx=idx)
    return FileSource(spec)
