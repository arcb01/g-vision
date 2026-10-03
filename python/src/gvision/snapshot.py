"""What was on screen when a question was asked, for the panel's conversation log.

``LatestFrame`` sits on the live pipeline's frame listeners and only keeps a
reference to the newest frame; nothing is copied or encoded until a question
is asked. The overlay is excluded from capture, so the picture is the game
without glows.
"""

from __future__ import annotations

import base64
from collections.abc import Callable

import numpy as np

from gvision.perception.capture import Frame

WIDTH = 960
"""Wide enough to read in the panel, ~60-90 KB as JPEG."""
JPEG_QUALITY = 75

Encode = Callable[[np.ndarray], str]


class LatestFrame:
    def __init__(self) -> None:
        self.frame: Frame | None = None

    def offer(self, frame: Frame) -> None:
        self.frame = frame

    def grab(self) -> np.ndarray | None:
        """A copy of the newest frame: the capture may reuse its buffer."""
        return None if self.frame is None else self.frame.image.copy()


def encode(image: np.ndarray) -> str:
    """BGR frame -> ``data:image/jpeg;base64,...``, downscaled to ``WIDTH``."""
    import cv2

    h, w = image.shape[:2]
    if w > WIDTH:
        image = cv2.resize(image, (WIDTH, max(1, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
