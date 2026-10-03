"""ByteTrack-style multi-object tracker (plan 5.4).

Associates detections across frames in normalized screen coordinates, keeps
a constant-velocity Kalman filter per track and runs the track life cycle:

    tentative -> confirmed (allowed to glow) -> lost (last position kept) -> removed

ByteTrack's idea is a second association pass with low-confidence
detections, so a partly occluded object keeps its track instead of
flickering out. Re-identification with appearance embeddings is left for
later; for now a lost track is recovered by position only.

Time is passed in explicitly (seconds), so the filter works at whatever rate
the detector runs and tests are deterministic.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from gvision.perception.detector import Detection


@dataclass
class TrackerConfig:
    high_conf: float = 0.5
    """Detections at or above this take part in the first association."""
    low_conf: float = 0.1
    """Detections between this and ``high_conf`` only extend existing tracks."""
    new_track_conf: float = 0.6
    """Minimum confidence to start a new track."""
    match_iou: float = 0.2
    """Minimum IoU for the first (high-confidence) association."""
    low_match_iou: float = 0.5
    tentative_match_iou: float = 0.3
    min_hits: int = 3
    """Consecutive hits before a track is confirmed (hysteresis, plan 5.5)."""
    max_lost_s: float = 1.0
    """How long a lost track is kept before it is removed."""


class _Kalman:
    """Constant-velocity filter over (cx, cy, w, h) with time-based steps."""

    # Noise relative to object height, as in ByteTrack, but per second.
    STD_POS = 1 / 20
    STD_VEL = 1 / 2

    def __init__(self, z: np.ndarray) -> None:
        self.x = np.concatenate([z, np.zeros(4)])
        h = max(z[3], 1e-3)
        self.P = np.diag(np.r_[np.full(4, 2 * self.STD_POS * h), np.full(4, 10 * self.STD_VEL * h)] ** 2)

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(8)
        F[:4, 4:] = dt * np.eye(4)
        h = max(self.x[3], 1e-3)
        q = np.r_[np.full(4, self.STD_POS * h * dt), np.full(4, self.STD_VEL * h * dt)] ** 2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag(q)
        self.x[2:4] = np.maximum(self.x[2:4], 1e-4)

    def update(self, z: np.ndarray) -> None:
        H = np.eye(4, 8)
        h = max(self.x[3], 1e-3)
        R = np.diag(np.full(4, self.STD_POS * h) ** 2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])


def _cxcywh_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, w, h = s[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N, 4) and ``b`` (M, 4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


@dataclass
class Track:
    ref: str
    label: str
    confidence: float
    kf: _Kalman
    status: str = "tentative"
    hits: int = 1
    last_seen: float = 0.0
    outline: list[tuple[float, float]] | None = None
    last_box: np.ndarray = field(default_factory=lambda: np.zeros(4))
    """Last observed xyxy box, shown while the track is lost."""

    @property
    def box(self) -> np.ndarray:
        """Current xyxy box: the filter estimate, or the last observation if lost."""
        return self.last_box if self.status == "lost" else _cxcywh_to_xyxy(self.kf.x)

    @property
    def velocity(self) -> tuple[float, float]:
        if self.status == "lost":
            return (0.0, 0.0)
        return (float(self.kf.x[4]), float(self.kf.x[5]))


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)
        self._t: float | None = None

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        """Advance to time ``t`` with this frame's detections; returns live tracks."""
        cfg = self.config
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        for tr in self.tracks:
            tr.kf.predict(dt)

        high = [d for d in detections if d.confidence >= cfg.high_conf]
        low = [d for d in detections if cfg.low_conf <= d.confidence < cfg.high_conf]
        active = [tr for tr in self.tracks if tr.status in ("confirmed", "lost")]
        tentative = [tr for tr in self.tracks if tr.status == "tentative"]

        # 1. High-confidence detections vs confirmed and lost tracks.
        m1, active_left, high_left = self._associate(active, high, cfg.match_iou)
        # 2. Low-confidence detections keep the remaining confirmed tracks alive.
        still_visible = [tr for tr in active_left if tr.status == "confirmed"]
        m2, unmatched_confirmed, _ = self._associate(still_visible, low, cfg.low_match_iou)
        # 3. Leftover high-confidence detections vs tentative tracks.
        m3, tentative_left, high_left = self._associate(tentative, high_left, cfg.tentative_match_iou)

        for tr, det in (*m1, *m2, *m3):
            self._apply(tr, det, t)
        for tr in unmatched_confirmed:
            tr.status = "lost"
        removed = {id(tr) for tr in tentative_left}
        removed |= {id(tr) for tr in self.tracks if tr.status == "lost" and t - tr.last_seen > cfg.max_lost_s}
        self.tracks = [tr for tr in self.tracks if id(tr) not in removed]

        for det in high_left:
            if det.confidence >= cfg.new_track_conf:
                self.tracks.append(self._new_track(det, t))
        return list(self.tracks)

    def _associate(
        self, tracks: list[Track], dets: list[Detection], min_iou: float
    ) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        if not tracks or not dets:
            return [], list(tracks), list(dets)
        iou = iou_matrix(np.array([tr.box for tr in tracks]), np.array([d.box for d in dets]))
        same_label = np.array([[tr.label == d.label for d in dets] for tr in tracks])
        iou = np.where(same_label, iou, 0.0)
        rows, cols = linear_sum_assignment(-iou)
        matches = [(tracks[r], dets[c]) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]
        matched_t = {id(tr) for tr, _ in matches}
        matched_d = {id(d) for _, d in matches}
        return (
            matches,
            [tr for tr in tracks if id(tr) not in matched_t],
            [d for d in dets if id(d) not in matched_d],
        )

    def _apply(self, tr: Track, det: Detection, t: float) -> None:
        z = _xyxy_to_cxcywh(np.asarray(det.box, dtype=float))
        if tr.status == "lost":
            # Re-found after a gap: restart velocity from this observation.
            tr.kf = _Kalman(z)
            tr.status = "confirmed"
        else:
            tr.kf.update(z)
        tr.hits += 1
        tr.confidence = det.confidence
        tr.outline = det.outline
        tr.last_box = np.asarray(det.box, dtype=float)
        tr.last_seen = t
        if tr.status == "tentative" and tr.hits >= self.config.min_hits:
            tr.status = "confirmed"

    def _new_track(self, det: Detection, t: float) -> Track:
        box = np.asarray(det.box, dtype=float)
        tr = Track(
            ref=f"obj:{next(self._ids)}",
            label=det.label,
            confidence=det.confidence,
            kf=_Kalman(_xyxy_to_cxcywh(box)),
            last_seen=t,
            outline=det.outline,
            last_box=box,
        )
        if self.config.min_hits <= 1:
            tr.status = "confirmed"
        return tr
