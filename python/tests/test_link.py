"""Two-PC mode: the wire format, the server end and a real round trip."""

from __future__ import annotations

import asyncio
import json
import socket

import numpy as np
import pytest

from gvision import edge as edge_mod
from gvision.bridge import Bridge
from gvision.edge import Edge
from gvision.link import (
    CLIP, FRAME, SPEECH, ClockOffset, EdgeLink, NetworkSource, RemoteVoice, from_pcm, pack, to_pcm, unpack,
)
from gvision.perception.capture import Frame


def test_pack_round_trip():
    data = pack(FRAME, 1234.5, b"jpeg")
    assert unpack(data) == (FRAME, 1234.5, b"jpeg")
    with pytest.raises(ValueError):
        unpack(b"\x01")


def test_pcm_round_trip_is_close():
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.0], np.float32)
    back = from_pcm(to_pcm(samples))
    assert np.allclose(back, np.clip(samples, -1, 1), atol=1e-4)


def test_clock_offset_takes_the_quickest_trip():
    clock = ClockOffset()
    assert clock.offset == 0.0
    # The gaming PC's clock is 5 s behind; trips take 3 to 20 ms.
    for sent, trip in [(100.0, 0.020), (100.1, 0.003), (100.2, 0.010)]:
        clock.add(sent, sent + 5.0 + trip)
    assert clock.offset == pytest.approx(5.003)


def test_network_source_stamps_frames_in_this_pcs_time(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((20, 30, 3), np.uint8)
    image[:, :, 2] = 200
    ok, buf = cv2.imencode(".jpg", image)
    src = NetworkSource()
    assert src.latest() is None
    now = 1_000_000.0
    monkeypatch.setattr("gvision.link.time.time", lambda: now)
    src.put(buf.tobytes(), sent=now - 10.0 - 0.004, received=now)
    frame = src.latest()
    assert frame.image.shape == (20, 30, 3)
    assert frame.ts == pytest.approx(now)
    assert src.latest() is frame  # decoded once
    now += 5.0  # the gaming PC went quiet
    assert src.latest() is None


def test_ptt_down_then_clip_calls_the_assistant():
    link = EdgeLink()
    calls = []
    link.on_ptt_down = lambda: calls.append("down")
    link.on_ptt_up = lambda: calls.append(("up", len(link.recorder.stop())))
    link.handle(json.dumps({"t": "ptt"}))
    link.handle(json.dumps({"t": "level", "v": 0.7}))
    assert link.recorder.level() == pytest.approx(0.7)
    link.handle(pack(CLIP, 16000.0, to_pcm(np.zeros(8000, np.float32))))
    assert calls == ["down", ("up", 8000)]
    link.handle(pack(CLIP, 16000.0, b""))  # a clip without a key press is ignored
    assert len(calls) == 2


def test_remote_voice_sends_speech_and_stops():
    class Link:
        sent = []

        def send(self, data):
            self.sent.append(data)

    class Tts:
        voice, device = "af_heart", "GPU"

        def synthesize(self, text):
            return np.zeros(240, np.float32), 24000

    link = Link()
    voice = RemoteVoice(Tts(), link)
    voice.start()
    samples, rate = voice.synthesize("hi")
    voice.play(samples, rate)
    kind, value, payload = unpack(link.sent[0])
    assert (kind, value, len(payload)) == (SPEECH, 24000.0, 480)
    voice.stop()
    assert json.loads(link.sent[-1]) == {"t": "stop"}
    voice.play(samples, rate)  # stopped: nothing more is sent
    assert len(link.sent) == 2 and voice.stopped


class FakeSource:
    def __init__(self):
        self.closed = False

    def latest(self):
        return Frame(image=np.zeros((4, 4, 3), np.uint8), ts=500.0)

    def close(self):
        self.closed = True


class FakeRecorder:
    def start(self):
        pass

    def level(self):
        return 0.5

    def stop(self):
        return np.full(4000, 0.25, np.float32)


class FakePlayer:
    def __init__(self):
        self.played, self.stops = [], 0

    def add(self, samples, rate):
        self.played.append((len(samples), rate))

    def stop(self):
        self.stops += 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _until(cond, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.02)


def test_round_trip_through_the_bridge(monkeypatch):
    monkeypatch.setattr(edge_mod, "encode_jpeg", lambda image, quality: b"fake-jpeg")

    async def scenario():
        port = _free_port()
        bridge = Bridge("127.0.0.1", port)
        link = EdgeLink()
        bridge.edge = link.serve
        heard = []
        link.on_ptt_up = lambda: heard.append(link.recorder.stop())
        stop = asyncio.Event()
        server = asyncio.create_task(bridge.run(stop))
        await asyncio.sleep(0.2)
        player = FakePlayer()
        edge = Edge(f"ws://127.0.0.1:{port}/edge", FakeSource(), None, fps=20,
                    recorder=FakeRecorder(), player=player)
        client = asyncio.create_task(edge.run(stop))
        await _until(lambda: link.source._jpeg == b"fake-jpeg")
        assert link.connected and not bridge.connected  # the gaming PC is not an app client

        edge.ptt_down()
        await asyncio.sleep(0.1)
        assert link.recorder.level() == pytest.approx(0.5)
        edge.ptt_up()
        await _until(lambda: heard)
        assert len(heard[0]) == 4000 and heard[0][0] == pytest.approx(0.25, abs=1e-3)

        link.send(pack(SPEECH, 24000.0, to_pcm(np.zeros(2400, np.float32))))
        await _until(lambda: player.played)
        assert player.played == [(2400, 24000)]
        stops = player.stops
        link.send(json.dumps({"t": "stop"}))
        await _until(lambda: player.stops > stops)

        stop.set()
        await asyncio.wait_for(asyncio.gather(server, client), 5)

    asyncio.run(scenario())


def test_bridge_refuses_a_gaming_pc_without_edge_mode():
    async def scenario():
        from websockets.asyncio.client import connect

        port = _free_port()
        bridge = Bridge("127.0.0.1", port)
        stop = asyncio.Event()
        server = asyncio.create_task(bridge.run(stop))
        await asyncio.sleep(0.2)
        async with connect(f"ws://127.0.0.1:{port}/edge") as ws:
            await ws.wait_closed()
            assert ws.close_code == 1008
        stop.set()
        await server

    asyncio.run(scenario())
