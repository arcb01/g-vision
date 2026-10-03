import asyncio

import numpy as np

from gvision.perception.detector import Detection
from gvision.perception.exclusion_check import TEST_BOX, analyze
from gvision.perception.live import LivePipeline, LiveSettings
from gvision.protocol import ClearMsg, DimMsg, FocusMsg, HighlightMsg, ObjectsMsg, dump, parse


class FakeBridge:
    def __init__(self):
        self.handlers = []
        self.sent = []

    def on_message(self, handler):
        self.handlers.append(handler)

    def send(self, msg):
        self.sent.append(msg)


def make(watch=("person",), spotlight=False):
    bridge = FakeBridge()
    pipe = LivePipeline(bridge, source=None, detector=None, settings=LiveSettings(watch=set(watch), spotlight=spotlight))
    return bridge, pipe


DETS = [
    Detection((0.1, 0.1, 0.2, 0.4), 0.9, "person", outline=[(0.1, 0.1), (0.2, 0.1), (0.15, 0.4)]),
    Detection((0.6, 0.6, 0.7, 0.7), 0.9, "car"),
]


def steps(pipe, n, dets=DETS):
    return [pipe.step(i * 0.1, dets) for i in range(n)]


def test_messages_are_valid_and_only_confirmed_watched_tracks_glow():
    _, pipe = make()
    out = steps(pipe, 3)
    for msgs in out:
        for m in msgs:
            assert parse(dump(m)) == m
    assert [type(m) for m in out[0]] == [ObjectsMsg]
    highlights = [m for m in out[2] if isinstance(m, HighlightMsg)]
    assert [(h.ref, h.color_role) for h in highlights] == [("obj:1", "target")]
    person = out[2][0].objects[0]
    assert person.outline is not None and person.status == "confirmed"


def test_spotlight_dims_and_focuses_targets():
    _, pipe = make(spotlight=True)
    out = steps(pipe, 4)
    dims = [m for msgs in out for m in msgs if isinstance(m, DimMsg)]
    assert [d.on for d in dims] == [True, True]  # from confirmation on, resent each step
    assert [m.refs for m in out[3] if isinstance(m, FocusMsg)] == [["obj:1"]]
    lost = pipe.step(0.4, [])  # person lost: spotlight held through the grace period
    assert any(isinstance(m, DimMsg) and m.on for m in lost)
    gone = pipe.step(1.6, [])  # track removed: no targets left
    assert any(isinstance(m, DimMsg) and not m.on for m in gone)


def test_dismiss_stops_current_glows():
    bridge, pipe = make()
    steps(pipe, 3)
    asyncio.run(bridge.handlers[0](ClearMsg(reason="dismiss hotkey")))
    out = pipe.step(0.3, DETS)
    assert not any(isinstance(m, HighlightMsg) for m in out)


def _screen(value):
    return np.full((200, 300, 3), value, dtype=np.uint8)


def test_exclusion_analysis_detects_a_captured_overlay():
    before = _screen(120)
    after = (before * 0.1).astype(np.uint8)
    assert analyze(before, after).overlay_captured

    gold = _screen(120)
    x0, y0 = int(TEST_BOX.x * 300), int(TEST_BOX.y * 200)
    x1, y1 = int((TEST_BOX.x + TEST_BOX.w) * 300), int((TEST_BOX.y + TEST_BOX.h) * 200)
    for img in (gold,):
        img[y0 - 2 : y0 + 3, x0:x1] = img[y1 - 2 : y1 + 3, x0:x1] = (61, 200, 255)
        img[y0:y1, x0 - 2 : x0 + 3] = img[y0:y1, x1 - 2 : x1 + 3] = (61, 200, 255)
    assert analyze(before, gold).overlay_captured


def test_exclusion_analysis_passes_when_nothing_changes():
    result = analyze(_screen(120), _screen(118))
    assert not result.overlay_captured
    assert analyze(_screen(3), _screen(3)).brightness_ratio is None
