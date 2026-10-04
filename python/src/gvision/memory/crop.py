"""Sharp close-ups for questions about one part of the screen.

Frame history keeps 640 px frames: a HUD ammo counter on a 2560 px screen
shrinks to a few blurry pixels there. When the question names a region
("bottom right", "on the left", "top center"), the look tool also sends
Qwen that region of the newest frame at full resolution.
"""

from __future__ import annotations

import base64
import re

import numpy as np

MAX_SIDE = 1280
"""Longest side of the crop sent to Qwen (~1000 image tokens at most)."""
SCREEN_SIDE = 1600
"""Default longest side of the whole newest frame for "right now" looks (~1400 image tokens).
Settings > Vision > Image resolution changes it."""
JPEG_QUALITY = 90

_ROW = {"top": "top", "upper": "top", "bottom": "bottom", "lower": "bottom"}
_CENTER = {"center", "centre", "middle"}
_SIDE_BEFORE = {"the", "my", "your", "top", "bottom", "upper", "lower", "far", "very"}
_SIDE_AFTER = {"side", "corner", "edge", "half", "part", "hand"}

# Fractions of the screen (x0, y0, x1, y1) for each row/column word.
_ROWS = {"top": (0.0, 0.4), "bottom": (0.6, 1.0), "middle": (0.25, 0.75), None: (0.0, 1.0)}
_COLS = {"left": (0.0, 0.4), "right": (0.6, 1.0), "center": (0.3, 0.7), None: (0.0, 1.0)}


def region_of(question: str) -> tuple[str, tuple[float, float, float, float]] | None:
    """('bottom right', box) when the question names a part of the screen.

    'left' and 'right' only count as places when they read like one ("on the
    left", "top right", "right side"), not in "how many do I have left" or
    "right now".
    """
    tokens = re.findall(r"[a-z]+", question.lower())
    row = col = None
    center = False
    for i, t in enumerate(tokens):
        before = tokens[i - 1] if i else ""
        after = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in _ROW:
            row = _ROW[t]
        elif t in _CENTER:
            center = True
        elif t in ("left", "right") and (before in _SIDE_BEFORE or after in _SIDE_AFTER):
            col = t
    if center:
        if row and not col:
            col = "center"
        elif col and not row:
            row = "middle"
        elif not row and not col:
            row, col = "middle", "center"
    if not row and not col:
        return None
    name = " ".join(w for w in (row, col) if w)
    name = {"middle center": "center", "middle left": "left", "middle right": "right"}.get(name, name)
    (y0, y1), (x0, x1) = _ROWS[row], _COLS[col]
    return name, (x0, y0, x1, y1)


def crop_url(
    image: np.ndarray, box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0), max_side: int = MAX_SIDE
) -> tuple[str, tuple[int, int]]:
    """BGR frame -> (JPEG data URL of the region at full resolution, its size)."""
    import cv2

    h, w = image.shape[:2]
    x0, y0, x1, y1 = box
    part = image[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
    ph, pw = part.shape[:2]
    if max(ph, pw) > max_side:
        scale = max_side / max(ph, pw)
        part = cv2.resize(part, (max(1, round(pw * scale)), max(1, round(ph * scale))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode(), (part.shape[1], part.shape[0])
