import asyncio
import time

import numpy as np

from gvision.agent.agent import Agent, Answer
from gvision.agent.qwen import Reply
from gvision.assistant import Assistant
from gvision.perception.detector import Detection
from gvision.perception.live import LivePipeline, LiveSettings
from gvision.protocol import ClearMsg, DimMsg, ExchangeMsg, FocusMsg, HighlightMsg, ObjectsMsg, dump, parse
from gvision.snapshot import LatestFrame
from gvision.world import WorldState
from test_agent import FakeQwen, tool_reply
from test_live import FakeBridge


class FakeDetector:
    """Only 'sees' the classes it is prompted with, like YOLOE."""

    def __init__(self, classes):
        self.classes = list(classes)
        self.history = []

    def set_classes(self, classes):
        self.classes = list(classes)
        self.history.append(list(classes))

    def detect(self, image):
        scene = {"cow": (0.1, 0.4, 0.2, 0.5), "person": (0.6, 0.1, 0.7, 0.4)}
        return [Detection(box, 0.9, label) for label, box in scene.items() if label in self.classes]


class FakeSource:
    def __init__(self):
        self.ts = 0.0

    def latest(self):
        self.ts += 0.1
        return type("Frame", (), {"ts": self.ts, "image": np.zeros((4, 4, 3), np.uint8)})()

    def close(self):
        pass


def test_pipeline_follows_world_watches():
    world = WorldState()
    bridge = FakeBridge()
    det = FakeDetector(["person"])
    pipe = LivePipeline(bridge, FakeSource(), det, LiveSettings(), world=world)
    assert pipe.prompts() == ["person"]
    world.set_watch("cow", "danger")
    assert pipe.prompts() == ["person", "cow"]
    out = []
    for i in range(4):
        out += pipe.step(i * 0.1, det.detect(None) + [Detection((0.1, 0.4, 0.2, 0.5), 0.9, "cow")])
    assert isinstance(out[0], ClearMsg)  # the watch change clears old glows once
    assert sum(isinstance(m, ClearMsg) for m in out) == 1
    hl = [m for m in out if isinstance(m, HighlightMsg)]
    assert hl and {m.color_role for m in hl} == {"danger"}
    assert [o.label for o in world.confirmed()] == ["person", "cow"]
    for m in out:
        assert parse(dump(m)) == m


def test_spoken_request_end_to_end():
    """Typed request -> Qwen set_watch -> detector reprompted -> glow, spotlight, answer."""

    async def run():
        world = WorldState()
        bridge = FakeBridge()
        det = FakeDetector(["person"])
        settings = LiveSettings(rate_hz=50, spotlight=True, show_all=False)
        pipe = LivePipeline(bridge, FakeSource(), det, settings, world=world)
        qwen = FakeQwen(tool_reply("set_watch", targets=["cow"]), Reply("The cow is on your left."))
        assistant = Assistant(bridge, world, Agent(qwen, world))
        stop = asyncio.Event()
        loop = asyncio.create_task(pipe.run(stop))
        answer = await assistant.submit("where's the cow")
        await asyncio.sleep(0.2)
        stop.set()
        await loop
        return bridge.sent, det, answer

    sent, det, answer = asyncio.run(run())
    assert det.history == [["person", "cow"]]
    assert answer.text == "The cow is on your left." and answer.refs
    types = [m.type for m in sent]
    assert "voice" in types and "answer" in types and "answer_finished" in types
    cow_ref = answer.refs[0]
    assert any(isinstance(m, HighlightMsg) and m.ref == cow_ref and m.color_role == "target" for m in sent)
    assert any(isinstance(m, FocusMsg) and m.refs == [cow_ref] for m in sent)
    # The spotlight stays on for as long as the cow is tracked, not just while speaking.
    last_answer = max(i for i, m in enumerate(sent) if m.type == "answer_finished")
    assert any(isinstance(m, DimMsg) and m.on for m in sent[last_answer:])
    assert not any(isinstance(m, DimMsg) and not m.on for m in sent)
    # Only the cow reaches the overlay, although YOLOE also tracks the person.
    labels = {o.label for m in sent if isinstance(m, ObjectsMsg) for o in m.objects}
    assert labels == {"cow"}


def test_each_answer_goes_to_the_conversation_log_with_the_screen():
    async def run():
        world = WorldState()
        bridge = FakeBridge()
        screen = LatestFrame()
        screen.offer(FakeSource().latest())
        qwen = FakeQwen(tool_reply("set_watch", targets=["cow"]), Reply("The cow is on your left."))
        encoded = []

        def encode(image):
            encoded.append(image.shape)
            return "data:image/jpeg;base64,AAAA"

        assistant = Assistant(bridge, world, Agent(qwen, world), screen=screen, encode_fn=encode)
        before = time.time()
        await assistant.submit("where's the cow")
        await asyncio.sleep(0.05)
        return bridge.sent, encoded, before

    sent, encoded, before = asyncio.run(run())
    (ex,) = [m for m in sent if isinstance(m, ExchangeMsg)]
    assert ex.question == "where's the cow" and ex.answer == "The cow is on your left."
    assert ex.via == "typed" and ex.tools == ["set_watch"] and "llm_tool_call" in ex.latency_ms
    assert ex.screenshot == "data:image/jpeg;base64,AAAA" and encoded == [(4, 4, 3)]
    assert before <= ex.asked_ts <= ex.ts
    assert parse(dump(ex)) == ex
    assert [(s.kind, s.title) for s in ex.steps] == [
        ("llm", "Qwen chooses what to do"), ("detector", "Detector: find and highlight"), ("llm", "Qwen writes the answer"),
    ]
    assert "set_watch" in ex.steps[0].detail and '"target": "cow"' in ex.steps[1].detail


def test_conversation_log_without_a_screen():
    async def run():
        bridge = FakeBridge()
        world = WorldState()
        assistant = Assistant(bridge, world, Agent(FakeQwen(Reply("Hello.")), world))
        await assistant.ask("hi")
        await asyncio.sleep(0.01)
        return bridge.sent

    (ex,) = [m for m in asyncio.run(run()) if isinstance(m, ExchangeMsg)]
    assert ex.answer == "Hello." and ex.screenshot is None and ex.tools == []


class FakeTTS:
    voice = "af_heart"

    def synthesize(self, text):
        return np.zeros(12000, np.float32), 24000

    def play(self, samples, rate):
        pass

    def stop(self):
        pass


class FakeASR:
    name = "Whisper medium"

    def transcribe(self, audio):
        return "hello there"


class FakeRecorder:
    def start(self):
        pass

    def stop(self):
        return np.zeros(16000, np.float32)

    def level(self):
        return 0.0


def test_log_steps_cover_speech_qwen_and_voice():
    async def run():
        bridge = FakeBridge()
        world = WorldState()
        assistant = Assistant(bridge, world, Agent(FakeQwen(Reply("Hi!")), world),
                              asr=FakeASR(), tts=FakeTTS(), recorder=FakeRecorder())
        assistant.ptt_down()
        assistant.ptt_up()
        await assistant._task
        await asyncio.sleep(0.01)
        return bridge.sent

    (ex,) = [m for m in asyncio.run(run()) if isinstance(m, ExchangeMsg)]
    assert ex.via == "voice" and ex.question == "hello there"
    assert [s.kind for s in ex.steps] == ["asr", "llm", "tts"]
    assert ex.steps[0].title == "Speech-to-text: Whisper medium" and "1.0 s clip" in ex.steps[0].detail
    assert ex.steps[2].detail == "0.5 s of speech"


def test_log_shows_a_failed_qwen_request():
    import httpx

    class DownQwen:
        async def chat(self, *a, **k):
            raise httpx.ConnectError("refused")

    async def run():
        bridge = FakeBridge()
        world = WorldState()
        await Assistant(bridge, world, Agent(DownQwen(), world)).ask("hi")
        await asyncio.sleep(0.01)
        return bridge.sent

    (ex,) = [m for m in asyncio.run(run()) if isinstance(m, ExchangeMsg)]
    assert [(s.title, s.ok) for s in ex.steps] == [("Qwen request failed", False)]
    assert "refused" in ex.steps[0].detail


def test_dismiss_cancels_and_clears_watches():
    async def run():
        world = WorldState()
        bridge = FakeBridge()

        class SlowAgent:
            async def handle(self, text):
                await asyncio.sleep(10)
                return Answer("late")

        assistant = Assistant(bridge, world, SlowAgent())
        world.set_watch("cow")
        task = assistant.submit("find the pig")
        world.set_watch("pig")
        await asyncio.sleep(0.01)
        for h in bridge.handlers:
            await h(ClearMsg(reason="dismiss hotkey"))
        await asyncio.sleep(0.01)
        return task, world

    task, world = asyncio.run(run())
    assert task.cancelled()
    assert not world.watches


def test_spotlight_survives_brief_loss_and_lifts_when_cleared():
    world = WorldState()
    det = FakeDetector(["cow"])
    pipe = LivePipeline(FakeBridge(), FakeSource(), det, LiveSettings(spotlight=True, show_all=False), world=world)
    world.set_watch("cow")
    cow = [Detection((0.1, 0.4, 0.2, 0.5), 0.9, "cow")]
    for i in range(4):
        pipe.step(i * 0.1, cow)
    gap = pipe.step(0.45, [])  # occluded for one frame: track is lost, not gone
    assert any(isinstance(m, DimMsg) and m.on for m in gap)
    world.clear_watch()
    after = pipe.step(0.5, cow)
    assert any(isinstance(m, DimMsg) and not m.on for m in after)
    assert not any(isinstance(m, ObjectsMsg) and m.objects for m in after)


def test_voice_levels():
    from gvision.audio.mic import envelope, loudness

    assert loudness(np.zeros(100, np.float32)) == 0.0
    assert 0.3 < loudness(np.full(100, 0.1, np.float32)) < 1.0
    tone = np.sin(np.linspace(0, 400, 24000)).astype(np.float32) * 0.2
    assert len(envelope(tone, 24000, 20)) == 20
