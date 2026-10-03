"""RapidOCR, split into its two stages for the text watcher (plan 7.1).

The watcher runs detection often and recognition rarely, so the two stages
are exposed separately instead of RapidOCR's one-shot call:

- ``detect`` finds text lines on a downscaled copy of the image (the long
  side capped at ``det_side`` px) and returns pixel boxes in the original
  image's coordinates;
- ``recognize`` reads a batch of full-resolution line crops.

Both run on the CPU through onnxruntime with a few threads, so the game
keeps the whole GPU. On a 4-core cloud VM a 1280 px detection pass takes
~350 ms and reading three lines ~60 ms.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

PixelBox = tuple[int, int, int, int]
"""x0, y0, x1, y1 in pixels."""


class OcrEngine(Protocol):
    """What the text watcher needs; tests use a fake."""

    def detect(self, image: np.ndarray) -> list[PixelBox]: ...

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


class RapidOcrEngine:
    def __init__(self, det_side: int = 1280, threads: int = 4) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as e:
            raise RuntimeError('reading text needs RapidOCR: pip install -e ".[ocr]"') from e
        import cv2

        self._cv2 = cv2
        self.det_side = det_side
        self._ocr = RapidOCR(params={
            "Global.log_level": "warning",
            # The watcher resizes before detection, so RapidOCR must not
            # upscale again (its default grows the short side to 736 px).
            "Det.limit_type": "max",
            "Det.limit_side_len": det_side,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        })
        from rapidocr.ch_ppocr_rec import TextRecInput

        self._rec_input = TextRecInput

    def detect(self, image: np.ndarray) -> list[PixelBox]:
        h, w = image.shape[:2]
        scale = min(1.0, self.det_side / max(h, w))
        small = image
        if scale < 1.0:
            small = self._cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=self._cv2.INTER_AREA)
        out = self._ocr.text_det(small)
        if out.boxes is None or len(out.boxes) == 0:
            return []
        boxes = []
        for quad in np.asarray(out.boxes) / scale:
            x0, y0 = quad.min(axis=0)
            x1, y1 = quad.max(axis=0)
            boxes.append((max(int(x0), 0), max(int(y0), 0), min(int(x1) + 1, w), min(int(y1) + 1, h)))
        return [b for b in boxes if b[2] > b[0] and b[3] > b[1]]

    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        out = self._ocr.text_rec(self._rec_input(img=crops))
        return [(str(t), float(s)) for t, s in zip(out.txts, out.scores)]
