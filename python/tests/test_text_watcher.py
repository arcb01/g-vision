import asyncio
import json

import numpy as np
from scipy import ndimage

from gvision.agent.agent import Agent
from gvision.agent.qwen import Reply
from gvision.agent.tools import TEXT_TOOLS, TOOLS, ToolExecutor
from gvision.perception.capture import Frame
from gvision.perception.text_watcher import (
    STATIC_AFTER_S,
    ZONE_HEAT,
    TextLine,
    TextWatcher,
    group_lines,
    matches_where,
)
from gvision.protocol import Box, DimMsg, FocusMsg, HighlightMsg, dump, parse
from gvision.world import WorldState
from test_agent import FakeQwen, tool_reply
from test_live import FakeBridge

W, H = 1280, 720


class FakeOcr:
    """Text is drawn as solid rectangles whose gray level encodes the string."""

    def __init__(self):
        self.codes: dict[int, str] = {}
        self.detect_calls = 0
        self.read: list[str] = []

    def paint(self, image, x, y, w, h, text):
        known = {t: c for c, t in self.codes.items()}
        code = known.get(text, 100 + (37 * len(self.codes)) % 150)
        self.codes[code] = text
        image[y:y + h, x:x + w] = code

    def detect(self, image):
        self.detect_calls += 1
        labels, _ = ndimage.label(image[..., 0] >= 100)
        return [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(labels)]

    def recognize(self, crops):
        out = [(self.codes.get(int(c.max()), ""), 0.95) for c in crops]
        self.read += [t for t, _ in out]
        return out


def scene(ocr, texts, background=20):
    image = np.full((H, W, 3), background, np.uint8)
    for x, y, w, h, text in texts:
        ocr.paint(image, x, y, w, h, text)
    return image


QUEST = [(900, 40, 300, 24, "Quest: find the old lighthouse"), (900, 70, 220, 24, "Talk to the blacksmith")]
HP = (20, 680, 60, 22, "HP 100")


def run(watcher, frames, start=1000.0):
    async def go():
        for i, image in enumerate(frames):
            await watcher.tick(Frame(image, start + 1.5 * i))

    asyncio.run(go())


def test_reads_groups_and_skips_unchanged_frames():
    ocr = FakeOcr()
    watcher = TextWatcher(None, None, ocr, profiles_dir=None)
    image = scene(ocr, QUEST + [HP])
    run(watcher, [image, image.copy(), image.copy()])
    texts = {b.text for b in watcher.visible()}
    assert texts == {"Quest: find the old lighthouse\nTalk to the blacksmith", "HP 100"}
    assert ocr.detect_calls == 1  # nothing changed after the first pass
    assert len(ocr.read) == 3  # each line read once
    quest = next(b for b in watcher.visible() if "Quest" in b.text)
    assert matches_where(quest.box, "top right") and not matches_where(quest.box, "left")


def test_changed_text_is_reread_and_logged_and_gone_text_is_recent():
    ocr = FakeOcr()
    watcher = TextWatcher(None, None, ocr, profiles_dir=None)
    msg = (40, 300, 280, 24, "The gate is open")
    first = scene(ocr, [HP, msg])
    run(watcher, [first])
    ref = next(b.ref for b in watcher.visible() if b.text == "The gate is open")
    reads = len(ocr.read)
    changed = scene(ocr, [HP, (40, 300, 280, 24, "The gate is closed")])
    empty = scene(ocr, [HP])
    run(watcher, [changed, changed, empty, empty, empty], start=1001.5)
    assert {b.text for b in watcher.visible()} == {"HP 100"}
    assert ocr.read[reads:] == ["The gate is closed"]  # unchanged lines are never read again
    recent = watcher.recent(60, now=1010.0)
    assert [b.text for b in recent if b.gone] == ["The gate is closed", "The gate is open"]
    assert all(b.ref == ref for b in recent if b.gone)


def test_ui_text_stays_put_while_world_text_moves():
    ocr = FakeOcr()
    watcher = TextWatcher(None, None, ocr, profiles_dir=None)
    frames = []
    for i in range(5):  # the camera pans: background and the sign shift every frame
        frames.append(scene(ocr, [HP, (300 + 40 * i, 300, 200, 24, "Old Mill")], background=10 + 15 * (i % 2)))
    run(watcher, frames)
    kinds = {b.text: b.kind for b in watcher.visible()}
    assert kinds == {"HP 100": "ui", "Old Mill": "world"}


def test_static_text_zones_and_profile(tmp_path):
    ocr = FakeOcr()
    watcher = TextWatcher(None, None, ocr, profiles_dir=tmp_path)
    watcher.load_profile("game")
    frames = [scene(ocr, [HP, (40, 600, 300, 24, f"chat message {i}")]) for i in range(int(ZONE_HEAT) + 1)]
    run(watcher, frames)
    chat = next(b for b in watcher.visible() if b.text.startswith("chat"))
    assert chat.in_zone and not chat.static
    # HP never changes: after 30 s it is static and sorted last.
    asyncio.run(watcher.tick(Frame(frames[-1], 1000.0 + STATIC_AFTER_S + 10)))
    assert watcher.visible()[-1].text == "HP 100" and watcher.visible()[-1].static
    watcher.save_profile()
    data = json.loads((tmp_path / "game.json").read_text())
    assert "hp 100" in data["static"]

    again = TextWatcher(None, None, FakeOcr(), profiles_dir=tmp_path)
    again.load_profile("game")
    assert again.zones.sum() == watcher.zones.sum() > 0 and "hp 100" in again.static_texts


def test_group_lines_keeps_columns_apart():
    def line(i, x0, y0, x1, y1):
        return TextLine(i, (x0, y0, x1, y1), text=str(i))

    lines = [line(1, 0.1, 0.10, 0.3, 0.13), line(2, 0.1, 0.135, 0.25, 0.165),  # a paragraph
             line(3, 0.7, 0.10, 0.9, 0.13),  # same row, far right
             line(4, 0.1, 0.5, 0.2, 0.53)]  # far below
    assert sorted(sorted(ln.id for ln in g) for g in group_lines(lines)) == [[1, 2], [3], [4]]


class FakeSource:
    def __init__(self, image):
        self.image = image

    def latest(self):
        return Frame(self.image, 2000.0)

    def close(self):
        pass


def test_read_text_tool_answers_and_highlights_the_quoted_block():
    ocr = FakeOcr()
    bridge = FakeBridge()
    image = scene(ocr, QUEST + [HP, (40, 300, 280, 24, "Press E to open")])
    watcher = TextWatcher(bridge, FakeSource(image), ocr, profiles_dir=None)
    world = WorldState()
    tools = ToolExecutor(world, text=watcher)
    qwen = FakeQwen(tool_reply("read_text", about="quest"), Reply("Your quest is to find the old lighthouse."))
    answer = asyncio.run(Agent(qwen, world, tools).handle("what's my quest?"))

    assert qwen.calls[0]["tools"] == TOOLS + TEXT_TOOLS
    assert "call read_text" in qwen.calls[0]["messages"][0]["content"]
    result = json.loads(qwen.calls[1]["messages"][-1]["content"])
    assert [b["text"] for b in result["text"]] == ["Quest: find the old lighthouse\nTalk to the blacksmith"]
    quest_ref = result["text"][0]["ref"]
    assert answer.refs == [quest_ref]
    highlights = [m for m in bridge.sent if isinstance(m, HighlightMsg)]
    assert [(h.ref, h.color_role) for h in highlights] == [(quest_ref, "info")]
    assert highlights[0].box.x < 900 / W  # padded around the text
    assert any(isinstance(m, FocusMsg) and m.refs == [quest_ref] for m in bridge.sent)
    assert any(isinstance(m, DimMsg) and m.on for m in bridge.sent)
    for m in bridge.sent:
        assert parse(dump(m)) == m


def test_read_text_filters_by_place_and_falls_back_to_everything():
    ocr = FakeOcr()
    image = scene(ocr, QUEST + [HP])
    tools = ToolExecutor(WorldState(), text=TextWatcher(None, FakeSource(image), ocr, profiles_dir=None))

    async def go():
        return (await tools.run("read_text", {"where": "bottom left"}),
                await tools.run("read_text", {"about": "sign"}),
                await tools.run("recent_text", {"seconds": "soon"}))

    corner, unknown, recent = asyncio.run(go())
    assert [b["text"] for b in corner.content["text"]] == ["HP 100"]
    assert "no text mentions 'sign'" in unknown.content["note"] and len(unknown.content["text"]) == 2
    assert all(b["on_screen"] for b in recent.content["text"])


def test_without_watcher_text_tools_are_not_offered():
    tools = ToolExecutor(WorldState())
    assert tools.specs is TOOLS
    assert "error" in asyncio.run(tools.run("read_text", {})).content
