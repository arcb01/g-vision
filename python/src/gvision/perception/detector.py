"""Open-vocabulary object detector: YOLOE with text prompts (plan 5.2).

YOLOE is prompted with class names ("person", "explosive barrel") instead of
being trained on them. The segmentation variant also returns masks, which
are simplified into small outline polygons for the overlay (plan 6.3).

Visual exemplars and icon matching come later; ``ultralytics`` is imported
lazily so the rest of the package works without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "yoloe-26s-seg.pt"
MODELS_DIR = Path("models")
MAX_OUTLINE_POINTS = 48


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    """Normalized xyxy."""
    confidence: float
    label: str
    outline: list[tuple[float, float]] | None = None
    """Simplified mask polygon, normalized."""


def simplify_polygon(points: np.ndarray, epsilon: float, max_points: int = MAX_OUTLINE_POINTS) -> np.ndarray:
    """Ramer-Douglas-Peucker on a closed polygon, then thin to ``max_points``."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 3:
        return pts
    # Split the closed ring at the point farthest from the first one.
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[far] = True
    stack = [(0, far), (far, len(pts))]  # second span wraps back to index 0
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = pts[a], pts[b % len(pts)]
        seg = pts[a + 1 : b]
        d = q - p
        norm = np.hypot(*d)
        if norm < 1e-12:
            dist = np.linalg.norm(seg - p, axis=1)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / norm
        i = int(np.argmax(dist))
        if dist[i] > epsilon:
            mid = a + 1 + i
            keep[mid] = True
            stack += [(a, mid), (mid, b)]
    out = pts[keep]
    if len(out) > max_points:
        out = out[np.linspace(0, len(out) - 1, max_points).astype(int)]
    return out


def _outline(mask: np.ndarray, orig_shape: tuple[int, int]) -> list[tuple[float, float]] | None:
    """Normalized outline polygon of a mask, or ``None`` if the mask is empty.

    Uses the largest contour only: Ultralytics' own polygons join every piece
    of a split mask (an object occluded in the middle) with bridging lines.
    The overlay only draws contours, so a fragmented mask still gives its
    largest piece rather than falling back to the box.
    """
    import cv2
    from ultralytics.utils import ops

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = np.array([cv2.contourArea(c) for c in contours])
    i = int(areas.argmax())
    if len(contours[i]) < 3:
        return None
    xyn = ops.scale_coords(mask.shape, contours[i].reshape(-1, 2).astype(np.float32), orig_shape, normalize=True)
    h, w = orig_shape
    # Simplify in pixels so the tolerance is the same on both axes.
    px = simplify_polygon(xyn * (w, h), epsilon=max(1.5, 0.002 * max(w, h)))
    return [(float(x / w), float(y / h)) for x, y in px]


class YoloeDetector:
    def __init__(
        self,
        classes: list[str],
        model: str = DEFAULT_MODEL,
        conf: float = 0.1,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLOE, settings

        # Everything stays local (plan 16): opt out of Ultralytics analytics.
        if settings.get("sync"):
            settings.update({"sync": False})

        path = Path(model)
        if not path.parent.parts:
            # Bare model name: keep downloaded weights out of the source tree.
            MODELS_DIR.mkdir(exist_ok=True)
            path = MODELS_DIR / model
        self.model = YOLOE(str(path))
        self.classes: list[str] = []
        self.set_classes(classes)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        on_gpu = device.startswith("cuda") if device else torch.cuda.is_available()
        self.quantize = 16 if on_gpu else None  # fp16 on the GPU: faster, half the VRAM
        log.info("YOLOE %s ready, prompts: %s", path.name, ", ".join(self.classes))

    def set_classes(self, classes: list[str]) -> None:
        """Change the text prompts, e.g. when the agent starts a new watch.
        Encodes the new prompts with YOLOE's text encoder (tens of ms)."""
        self.model.set_classes(list(classes))
        self.classes = list(classes)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect in one BGR frame. ``conf`` is low on purpose: the tracker
        uses low-confidence detections to keep existing tracks alive."""
        r = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, device=self.device, quantize=self.quantize, verbose=False
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxyn.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy().astype(np.uint8) if r.masks is not None else [None] * len(boxes)
        out = []
        for box, c, k, mask in zip(boxes, confs, cls, masks):
            outline = _outline(mask, image.shape[:2]) if mask is not None else None
            out.append(Detection(tuple(map(float, box)), float(c), r.names[int(k)], outline))
        return out
