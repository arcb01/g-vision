import asyncio
import json

import numpy as np
import pytest

from gvision.agent.agent import Agent
from gvision.agent.qwen import Reply
from gvision.agent.tools import ToolExecutor, ToolResult
from gvision.memory import EventLog, FrameHistory, LookTool, Narrator, Preempted, SharedQwen, StoredFrame
from gvision.memory.look import HINT, SCHEMA
from gvision.perception.capture import Frame
from gvision.world import Situation, WorldState
from test_agent import FakeQwen, obj, tool_reply


def stored(ts, level=0, change=None):
    f = StoredFrame(ts, b"jpeg%d" % ts, np.full((18, 32), level, np.uint8))
    if change is not None:
        f.change = change
    return f


def fake_prepare(image):
    return b"jpeg", np.full((18, 32), int(image[0, 0, 0]), np.uint8)


def test_history_keeps_one_minute_and_scores_change():
    h = FrameHistory(seconds=10, fps=2)
    for i in range(30):
        h.add(stored(i * 0.5, level=200 if i == 25 else 0))
    assert len(h.frames) == 20
    assert h.frames[0].ts == 5.0
    assert h.frames[-5].change == pytest.approx(200 / 255)  # the flash
    assert h.frames[-4].change == pytest.approx(200 / 255)  # and back
    assert h.frames[-1].change == 0
    assert h.max_change_since(14.0) == 0


def test_offer_samples_at_the_history_rate_off_the_loop():
    async def run():
        h = FrameHistory(seconds=60, fps=2, prepare_fn=fake_prepare)
        for i in range(10):  # 10 frames at 10 Hz -> 2 stored at 2 fps
            h.offer(Frame(np.full((4, 4, 3), i, np.uint8), ts=100 + i * 0.1))
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        return h

    h = asyncio.run(run())
    assert [round(f.ts, 1) for f in h.frames] == [100.0, 100.5]


def test_real_jpeg_encoding_downscales():
    cv2 = pytest.importorskip("cv2")
    from gvision.memory.history import prepare

    jpeg, thumb = prepare(np.zeros((1440, 2560, 3), np.uint8))
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (360, 640, 3)
    assert thumb.shape == (18, 32)


def test_pick_prefers_eventful_frames_and_always_the_newest():
    h = FrameHistory()
    for i, change in enumerate([0, 0.01, 0.5, 0.02, 0, 0.3, 0.01, 0, 0, 0.01]):
        h.add(stored(float(i)))
        h.frames[-1].change = change
    picked = h.pick(seconds=100, k=3, now=9.0)
    assert [f.ts for f in picked] == [2.0, 5.0, 9.0]
    assert [f.ts for f in h.pick(seconds=2.5, k=4, now=9.0)] == [7.0, 8.0, 9.0]  # fewer frames than k: all of them
    assert [f.ts for f in h.pick(seconds=0, k=1, now=50.0)] == [9.0]  # nothing that recent: the newest


def test_event_log_logs_appear_and_gone_once():
    world = WorldState()
    log = EventLog(world)
    world.set_objects(1.0, [obj("obj:1", "zombie"), obj("obj:2", "cow", x=0.8, status="tentative")])
    log.update(10.0)
    world.set_objects(2.0, [obj("obj:1", "zombie", status="lost")])  # occluded: not gone yet
    log.update(11.0)
    world.set_objects(3.0, [])
    log.update(12.0)
    assert [(e.ts, e.text) for e in log.events] == [(10.0, "zombie appeared on the left"), (12.0, "zombie gone")]
    assert log.describe(11.0, now=15.0) == ["3 s ago: zombie gone"]


class SlowQwen:
    def __init__(self):
        self.started = []

    async def chat(self, messages, tools=None, max_tokens=200, **kwargs):
        self.started.append(messages[0]["content"])
        await asyncio.sleep(0.2)
        return Reply(messages[0]["content"])


def test_player_request_preempts_the_narrator():
    async def run():
        shared = SharedQwen(SlowQwen(), quiet_s=0.0)
        background = asyncio.create_task(shared.background_chat([{"role": "user", "content": "narrate"}]))
        await asyncio.sleep(0.05)
        reply = await shared.chat([{"role": "user", "content": "player"}])
        with pytest.raises(Preempted):
            await background
        return reply

    assert asyncio.run(run()).content == "player"


def test_narrator_waits_until_the_player_is_done():
    async def run():
        qwen = SlowQwen()
        shared = SharedQwen(qwen, quiet_s=0.1)
        player = asyncio.create_task(shared.chat([{"role": "user", "content": "player"}]))
        await asyncio.sleep(0.01)
        background = await shared.background_chat([{"role": "user", "content": "narrate"}])
        await player
        return qwen.started, background

    started, background = asyncio.run(run())
    assert started == ["player", "narrate"]
    assert background.content == "narrate"


def test_a_vision_model_takes_the_look_tool_and_the_narrator():
    async def run():
        main, vision = SlowQwen(), SlowQwen()
        shared = SharedQwen(main, quiet_s=0.0, vision=vision)
        background = asyncio.create_task(shared.background_chat([{"role": "user", "content": "narrate"}]))
        await asyncio.sleep(0.05)
        # A look preempts the narrator like any player request.
        await shared.looking.chat([{"role": "user", "content": "look"}])
        with pytest.raises(Preempted):
            await background
        await shared.chat([{"role": "user", "content": "route"}])
        return main.started, vision.started

    main, vision = asyncio.run(run())
    assert main == ["route"]
    assert vision == ["narrate", "look"]


class JsonQwen:
    def __init__(self, notes):
        self.notes = notes
        self.calls = []

    async def chat(self, messages, tools=None, max_tokens=200, **kwargs):
        self.calls.append((messages, kwargs))
        return Reply(json.dumps(self.notes))


def test_narrator_writes_the_situation_into_the_snapshot():
    async def run():
        world = WorldState()
        history = FrameHistory()
        events = EventLog(world)
        qwen = JsonQwen({"situation": "Fighting a zombie at night.", "player": "Low health.",
                         "objective": "unknown", "summary": "Mined iron, then night fell."})
        narrator = Narrator(SharedQwen(qwen, quiet_s=0), history, events, world)
        assert not narrator.due()  # no frames yet
        history.add(stored(100.0))
        events.add("zombie appeared on the left", ts=99.0)
        assert narrator.due()
        await narrator.narrate()
        assert not narrator.due()  # nothing new since
        history.add(stored(101.0))
        assert not narrator.due()  # a new frame, but the same scene
        history.add(stored(102.0, level=120))
        assert narrator.due()
        return world, qwen

    world, qwen = asyncio.run(run())
    messages, kwargs = qwen.calls[0]
    content = messages[0]["content"]
    assert "zombie appeared" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert kwargs["response_format"]["type"] == "json_schema"
    snap = world.snapshot()["situation"]
    assert snap["situation"] == "Fighting a zombie at night."
    assert "objective" not in snap  # "unknown" is left out


def test_narrator_keeps_plain_text_replies():
    from gvision.memory.narrator import parse

    assert parse("A quiet forest.", 1.0).situation == "A quiet forest."
    assert parse("", 1.0) is None


def test_what_just_hit_me_end_to_end():
    async def run():
        world = WorldState()
        history = FrameHistory()
        now = __import__("time").time()
        for i in range(20):
            history.add(stored(now - 10 + i * 0.5))
        history.frames[-6].change = 0.4
        events = EventLog(world)
        events.add("skeleton appeared top right", ts=now - 4)
        qwen = FakeQwen(tool_reply("look", question="what just hit me?", seconds=10),
                        Reply("An arrow from the skeleton on your right."))
        tools = ToolExecutor(world)
        tools.register(SCHEMA, LookTool(qwen, history, events), HINT)
        answer = await Agent(qwen, world, tools).handle("what just hit me?")
        return answer, qwen

    answer, qwen = asyncio.run(run())
    assert answer.text == "An arrow from the skeleton on your right."
    assert answer.tool_calls == ["look"]
    assert len(qwen.calls) == 2  # tool call + look; no third call for the answer
    first, look = qwen.calls
    assert "call look" in first["messages"][0]["content"]
    assert "look back in time" not in first["messages"][0]["content"]
    assert "look" in {t["function"]["name"] for t in first["tools"]}
    parts = look["messages"][0]["content"]
    assert "skeleton appeared top right" in parts[0]["text"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 4


def test_look_without_frames_reports_an_error():
    result = asyncio.run(LookTool(FakeQwen(), FrameHistory())("what was that", seconds="soon"))
    assert result.content == {"error": "no frames captured yet"}
    assert result.speak is None


def test_situation_notes_skip_unknowns():
    s = Situation(ts=1.0, situation="In a cave.", player="unknown", objective="", summary="Explored.")
    assert s.notes() == {"situation": "In a cave.", "summary": "Explored."}


def test_look_and_text_tools_offered_together():
    tools = ToolExecutor(WorldState(), text=object())
    tools.register(SCHEMA, LookTool(FakeQwen(), FrameHistory()), HINT)
    names = [t["function"]["name"] for t in tools.specs]
    assert {"read_text", "recent_text", "look"} <= set(names)
    abilities = Agent(FakeQwen(), WorldState(), tools).abilities()
    assert "call read_text" in abilities and "call look" in abilities
    assert "can't" not in abilities


def test_regions_named_in_questions():
    from gvision.memory.crop import region_of

    assert region_of("How many bullets are there shown on the right bottom?") == ("bottom right", (0.6, 0.6, 1.0, 1.0))
    assert region_of("what's in the bottom left corner")[0] == "bottom left"
    assert region_of("what does the top center say")[0] == "top center"
    assert region_of("what is on my left")[0] == "left"
    assert region_of("what is in the middle of the screen")[0] == "center"
    assert region_of("read the upper right")[0] == "top right"
    # "left" and "right" that aren't places
    assert region_of("How many bullets do I have left?") is None
    assert region_of("what is that right now") is None
    assert region_of("what just hit me?") is None


def test_crop_is_full_resolution_and_capped():
    pytest.importorskip("cv2")
    from gvision.memory.crop import crop_url

    image = np.zeros((1440, 2560, 3), np.uint8)
    url, size = crop_url(image, (0.6, 0.6, 1.0, 1.0))
    assert url.startswith("data:image/jpeg;base64,") and size == (1024, 576)
    _, size = crop_url(image, (0.0, 0.6, 1.0, 1.0))  # the whole bottom band: scaled to 1280 wide
    assert size == (1280, 288)


def test_look_adds_a_sharp_crop_when_the_question_names_a_region():
    pytest.importorskip("cv2")
    from gvision.snapshot import LatestFrame

    screen = LatestFrame()
    screen.offer(Frame(ts=0.0, image=np.zeros((1440, 2560, 3), np.uint8)))
    history = FrameHistory()
    history.add(stored(__import__("time").time()))

    async def ask(question):
        qwen = FakeQwen(Reply("You have 10 bullets."))
        tools = ToolExecutor(WorldState())
        tools.register(SCHEMA, LookTool(qwen, history, screen=screen), HINT)
        result = await tools.run("look", {"question": question, "seconds": 0})
        return result, tools.step("look", {"question": question}, result, 1.0), qwen.calls[0]["messages"][0]["content"]

    result, step, parts = asyncio.run(ask("How many bullets are on the bottom right?"))
    assert len([p for p in parts if p["type"] == "image_url"]) == 2  # overview + crop
    assert "close-up of the bottom right" in parts[0]["text"]
    assert result.content["crop"] == {"region": "bottom right", "size": "1024x576"}
    assert "Sharp crop of the bottom right: 1024x576 px" in step.detail

    result, _, parts = asyncio.run(ask("How many bullets do I have left?"))
    assert len([p for p in parts if p["type"] == "image_url"]) == 1 and "crop" not in result.content


def test_the_look_step_names_the_vision_model():
    tools = ToolExecutor(WorldState())
    result = ToolResult({"answer": "a zombie"})
    assert tools.step("look", {"question": "what hit me?"}, result, 1.0).title == "Qwen vision: look at recent frames"
    tools.vision_model = "Qwen3.5-4B-Q4_K_M"
    step = tools.step("look", {"question": "what hit me?"}, result, 1.0)
    assert step.title == "Qwen vision: look at recent frames · Qwen3.5-4B-Q4_K_M"
    assert step.detail.startswith("Model: Qwen3.5-4B-Q4_K_M\n")
    assert "·" not in tools.step("query_state", {}, result, 1.0).title
