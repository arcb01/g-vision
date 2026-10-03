import copy

import numpy as np
import pytest

from gvision.perception.detector import Detection, simplify_polygon
from gvision.perception.tracker import ByteTracker, TrackerConfig, iou_matrix


def det(x, y, w=0.1, h=0.2, conf=0.9, label="person"):
    return Detection(box=(x, y, x + w, y + h), confidence=conf, label=label)


def run(tracker, frames, dt=0.1):
    out = []
    for i, dets in enumerate(frames):
        # Tracks are updated in place; snapshot each step.
        out.append(copy.deepcopy(tracker.update(dets, i * dt)))
    return out


def test_iou_matrix():
    a = np.array([[0, 0, 1, 1]], dtype=float)
    b = np.array([[0, 0, 1, 1], [0.5, 0, 1.5, 1], [2, 2, 3, 3]], dtype=float)
    assert iou_matrix(a, b) == pytest.approx(np.array([[1.0, 1 / 3, 0.0]]))


def test_confirmed_after_min_hits_with_stable_ref():
    tracker = ByteTracker(TrackerConfig(min_hits=3))
    steps = run(tracker, [[det(0.1 + 0.01 * i, 0.3)] for i in range(5)])
    assert [s[0].status for s in steps] == ["tentative", "tentative", "confirmed", "confirmed", "confirmed"]
    assert {s[0].ref for s in steps} == {"obj:1"}


def test_velocity_follows_motion():
    tracker = ByteTracker()
    steps = run(tracker, [[det(0.1 + 0.02 * i, 0.3)] for i in range(15)])  # 0.2 units/s right
    vx, vy = steps[-1][0].velocity
    assert vx == pytest.approx(0.2, abs=0.03)
    assert vy == pytest.approx(0.0, abs=0.03)


def test_tentative_dies_on_first_miss():
    tracker = ByteTracker()
    steps = run(tracker, [[det(0.1, 0.1)], []])
    assert steps[1] == []


def test_lost_then_recovered_keeps_ref():
    tracker = ByteTracker(TrackerConfig(max_lost_s=1.0))
    frames = [[det(0.4, 0.4)]] * 4 + [[]] * 3 + [[det(0.41, 0.4)]]
    steps = run(tracker, frames)
    assert steps[4][0].status == "lost"
    assert steps[4][0].velocity == (0.0, 0.0)
    assert steps[-1][0].status == "confirmed"
    assert steps[-1][0].ref == "obj:1"


def test_lost_track_removed_after_timeout():
    tracker = ByteTracker(TrackerConfig(max_lost_s=0.5))
    steps = run(tracker, [[det(0.4, 0.4)]] * 4 + [[]] * 8)
    assert steps[-1] == []


def test_low_confidence_detection_keeps_track_alive():
    tracker = ByteTracker()
    frames = [[det(0.4, 0.4)]] * 4 + [[det(0.4, 0.4, conf=0.3)]] * 3
    steps = run(tracker, frames)
    assert [tr.status for tr in steps[-1]] == ["confirmed"]


def test_low_confidence_detection_never_starts_a_track():
    tracker = ByteTracker()
    steps = run(tracker, [[det(0.4, 0.4, conf=0.3)]] * 5)
    assert steps[-1] == []


def test_labels_do_not_swap():
    tracker = ByteTracker()
    frames = [[det(0.4, 0.4, label="person"), det(0.41, 0.4, label="dog")]] * 4
    steps = run(tracker, frames)
    assert sorted((tr.ref, tr.label) for tr in steps[-1]) == [("obj:1", "person"), ("obj:2", "dog")]


def test_simplify_polygon_keeps_corners():
    # A dense square: 400 points on its perimeter.
    t = np.linspace(0, 4, 400, endpoint=False)
    side, f = np.divmod(t, 1)
    corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]], dtype=float)
    pts = corners[side.astype(int)] + f[:, None] * (corners[side.astype(int) + 1] - corners[side.astype(int)])
    out = simplify_polygon(pts, epsilon=1.0)
    assert len(out) == 4
    assert {tuple(p) for p in out.round()} == {(0, 0), (100, 0), (100, 100), (0, 100)}


def test_outline_carried_over_when_a_frame_has_no_mask():
    from gvision.perception.detector import Detection
    from gvision.perception.tracker import ByteTracker

    tracker = ByteTracker()
    outline = [(0.1, 0.1), (0.2, 0.1), (0.15, 0.3)]
    tracker.update([Detection((0.1, 0.1, 0.2, 0.3), 0.9, "cow", outline)], 0.0)
    (tr,) = tracker.update([Detection((0.12, 0.1, 0.22, 0.3), 0.9, "cow")], 0.1)  # moved right, no mask
    assert np.ravel(tr.outline) == pytest.approx([0.12, 0.1, 0.22, 0.1, 0.17, 0.3])
