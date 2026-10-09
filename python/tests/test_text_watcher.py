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
from gvision.protocol import Box, ClearMsg, DimMsg, FocusMsg, HighlightMsg, dump, parse
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


def test_read_text_glow_follows_the_voice(monkeypatch):
    import gvision.assistant as assistant_module
    from gvision.assistant import Assistant

    monkeypatch.setattr(assistant_module, "CHARS_PER_S", 2000.0)  # no voice: walk the cues fast
    monkeypatch.setattr(assistant_module, "TEXT_LINGER_S", 0.0)
    ocr = FakeOcr()
    bridge = FakeBridge()
    door = (40, 300, 280, 24, "Press E to open the door")
    image = scene(ocr, QUEST + [HP, door])
    watcher = TextWatcher(bridge, FakeSource(image), ocr, profiles_dir=None)
    world = WorldState()
    tools = ToolExecutor(world, text=watcher)
    qwen = FakeQwen(tool_reply("read_text"),
                    Reply("Your quest: find the old lighthouse. The sign says press E to open the door."))
    answer = asyncio.run(Assistant(bridge, world, Agent(qwen, world, tools)).ask("what does it say?"))

    assert qwen.calls[0]["tools"] == TOOLS + TEXT_TOOLS
    assert "call read_text" in qwen.calls[0]["messages"][0]["content"]
    result = {b["text"]: b["ref"] for b in json.loads(qwen.calls[1]["messages"][-1]["content"])["text"]}
    quest = result["Quest: find the old lighthouse\nTalk to the blacksmith"]
    press = result["Press E to open the door"]
    assert [ref for _, ref in answer.text_cues] == [quest, press]  # in the order they are spoken
    assert [(s.kind, s.ok) for s in answer.steps] == [("llm", True), ("ocr", True), ("llm", True)]
    assert "read_text(" in answer.steps[1].detail and "lighthouse" in answer.steps[1].detail
    assert answer.text_cues[0][0] < 0.3 < answer.text_cues[1][0]
    highlights = [m for m in bridge.sent if isinstance(m, HighlightMsg)]
    assert {(h.ref, h.color_role) for h in highlights} == {(quest, "info"), (press, "info")}  # not HP
    assert all(h.box.w > 0 for h in highlights)
    # Both are outlined unlit first, then each glows only while it is being read.
    assert [m.refs for m in bridge.sent if isinstance(m, FocusMsg)] == [[], [quest], [press]]
    # Once read, the outlines and dimming are cleared.
    assert isinstance(bridge.sent[-1], ClearMsg) and bridge.sent[-1].reason == "text read"
    assert any(isinstance(m, DimMsg) and m.on for m in bridge.sent)
    for m in bridge.sent:
        assert parse(dump(m)) == m


def test_single_text_result_is_lit_even_without_shared_words():
    ocr = FakeOcr()
    image = scene(ocr, [HP])
    tools = ToolExecutor(WorldState(), text=TextWatcher(None, FakeSource(image), ocr, profiles_dir=None))
    result = asyncio.run(tools.run("read_text", {}))
    assert tools.text_cues("Your health is full.", [result]) == [(0.0, result.text_refs[0])]
    assert tools.text_cues("Nothing.", []) == []


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


# --- regressions from Arnau's logs (2026-10-05): the outline never showed ---

SIGN = (300, 340, 260, 24, "Hello my friend")
COORDS = (20, 20, 160, 22, "34, 68, 54")


def text_and_look(bridge, image, ocr, said):
    """An executor with the real watcher and a fake vision tool that says ``said``."""
    from gvision.agent.tools import ToolResult
    from gvision.memory.look import SCHEMA

    world = WorldState()
    watcher = TextWatcher(bridge, FakeSource(image), ocr, profiles_dir=None)
    tools = ToolExecutor(world, text=watcher)
    looked = []

    async def look(question="", seconds=10):
        looked.append(question)
        return ToolResult({"seen": said}, speak=said)

    tools.register(SCHEMA, look)
    return world, watcher, tools, looked


def test_vision_answer_still_outlines_the_text_it_quotes(monkeypatch):
    """Qwen asks read_text about 'sign', a word not on the sign; vision answers.
    The sign's text was read, so it must still be outlined."""
    import gvision.assistant as assistant_module
    from gvision.assistant import Assistant

    monkeypatch.setattr(assistant_module, "CHARS_PER_S", 2000.0)
    ocr = FakeOcr()
    bridge = FakeBridge()
    image = scene(ocr, [SIGN, COORDS])
    world, watcher, tools, looked = text_and_look(bridge, image, ocr, 'The sign on the left says "Hello my friend".')
    qwen = FakeQwen(tool_reply("read_text", about="sign", where="left"))
    answer = asyncio.run(Assistant(bridge, world, Agent(qwen, world, tools, route_first=True)).ask("What does the sign on the left say?"))

    assert looked and answer.tool_calls == ["read_text", "look"]
    sign = next(b.ref for b in watcher.visible() if b.text == "Hello my friend")
    assert [ref for _, ref in answer.text_cues] == [sign]  # not the coordinates
    assert [m.ref for m in bridge.sent if isinstance(m, HighlightMsg)] == [sign]
    assert [m.refs for m in bridge.sent if isinstance(m, FocusMsg)] == [[], [sign]]


def test_vision_answer_without_a_text_tool_outlines_quoted_text():
    """Vision-only mode or a direct look: match the answer against what is on screen."""
    ocr = FakeOcr()
    image = scene(ocr, [SIGN, COORDS, (900, 340, 120, 24, "Yes sir")])
    world, watcher, tools, _ = text_and_look(FakeBridge(), image, ocr, 'The sign says "Yes sir".')
    asyncio.run(watcher.refresh())
    answer = asyncio.run(Agent(FakeQwen(), world, tools, vision_only=True).handle("what does this say?"))
    yes = next(b.ref for b in watcher.visible() if b.text == "Yes sir")
    assert [ref for _, ref in answer.text_cues] == [yes]


def test_vision_answer_about_something_else_outlines_nothing():
    ocr = FakeOcr()
    image = scene(ocr, [COORDS])
    world, _, tools, _ = text_and_look(FakeBridge(), image, ocr, "That is a spider.")
    qwen = FakeQwen(tool_reply("read_text", about="spider"))
    answer = asyncio.run(Agent(qwen, world, tools).handle("What is this?"))
    assert answer.text_cues == []  # the lone coordinates block is not what it read


def test_text_that_left_the_screen_is_still_outlined():
    """A toast read by read_text can be gone (or re-grouped) by the time the answer is spoken."""
    ocr = FakeOcr()
    bridge = FakeBridge()
    image = scene(ocr, [(900, 40, 300, 24, "New Recipes Unlocked!")])
    world = WorldState()
    watcher = TextWatcher(bridge, FakeSource(image), ocr, profiles_dir=None)
    tools = ToolExecutor(world, text=watcher)
    result = asyncio.run(tools.run("read_text", {"where": "top right"}))
    watcher.blocks = {}  # the toast slid away
    cues = tools.text_cues('The top right shows a "New Recipes Unlocked!" message.', [result])
    assert [ref for _, ref in cues] == result.text_refs
    assert watcher.show(result.text_refs, known=tools.text_known([result])) == result.text_refs
    assert any(isinstance(m, HighlightMsg) for m in bridge.sent)


def test_look_first_answer_outlines_the_text_it_quotes(monkeypatch):
    """Look answers (no read_text first) and quotes the sign: the sign is outlined."""
    import gvision.assistant as assistant_module
    from gvision.assistant import Assistant

    monkeypatch.setattr(assistant_module, "CHARS_PER_S", 2000.0)
    ocr = FakeOcr()
    bridge = FakeBridge()
    image = scene(ocr, [SIGN, COORDS])
    world, watcher, tools, looked = text_and_look(bridge, image, ocr, 'The sign on the left says "Hello my friend".')
    asyncio.run(watcher.refresh())
    answer = asyncio.run(Assistant(bridge, world, Agent(FakeQwen(Reply("look")), world, tools)).ask(
        "What does the sign on the left say?"))

    assert looked and answer.tool_calls == ["look"]
    sign = next(b.ref for b in watcher.visible() if b.text == "Hello my friend")
    assert [ref for _, ref in answer.text_cues] == [sign]
