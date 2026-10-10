"""The link between the gaming PC and the AI server PC (two-PC mode).

The gaming PC runs ``python -m gvision --edge ws://<server>:8765/edge``
(``gvision.edge``): it captures the screen, records push-to-talk questions
and plays the answers. The AI server runs the backend with
``--source edge``: frames, questions and answers go through the
:class:`EdgeLink` here, on the bridge's ``/edge`` path, so a single port is
opened on the server.

On the wire, binary messages are a 9-byte header, ``kind`` (1 byte) and a
float64 ``value``, then the payload:

- ``FRAME``: value = capture time on the gaming PC, payload = JPEG;
- ``CLIP``: value = sample rate, payload = int16 mono PCM, sent when the
  push-to-talk key comes up;
- ``SPEECH``: value = sample rate, payload = int16 mono PCM, one sentence of
  the answer to play.

Text messages are small JSON objects: ``{"t": "ptt"}`` (key down),
``{"t": "level", "v": 0.4}`` (mic loudness while held) and ``{"t": "stop"}``
(cut the answer short).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from gvision.perception.capture import Frame

log = logging.getLogger(__name__)

PATH = "/edge"
FRAME, CLIP, SPEECH = 1, 2, 3
_HEADER = struct.Struct("<Bd")

STALE_S = 2.0
"""A frame older than this is not handed out: the gaming PC stopped sending."""
OFFSET_WINDOW = 300
"""Frames over which the clock offset is estimated (about 30 s at 10 fps)."""


def pack(kind: int, value: float, payload: bytes) -> bytes:
    return _HEADER.pack(kind, value) + payload


def unpack(data: bytes) -> tuple[int, float, bytes]:
    if len(data) < _HEADER.size:
        raise ValueError("short message")
    kind, value = _HEADER.unpack_from(data)
    return kind, value, data[_HEADER.size:]


def to_pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def from_pcm(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767


class ClockOffset:
    """Gaming PC time -> this PC's time. The smallest (arrival - capture) gap
    seen recently is the clock difference plus the quickest trip, a few ms."""

    def __init__(self, window: int = OFFSET_WINDOW) -> None:
        self._gaps: deque[float] = deque(maxlen=window)

    def add(self, sent: float, received: float) -> None:
        self._gaps.append(received - sent)

    @property
    def offset(self) -> float:
        return min(self._gaps) if self._gaps else 0.0


class NetworkSource:
    """A ``FrameSource`` fed by the gaming PC. Like the screen source it only
    ever hands out the latest frame; it is decoded once, when first asked for."""

    def __init__(self) -> None:
        self.clock = ClockOffset()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._ts = 0.0
        self._received = 0.0
        self._frame: Frame | None = None

    def put(self, jpeg: bytes, sent: float, received: float | None = None) -> None:
        received = time.time() if received is None else received
        self.clock.add(sent, received)
        with self._lock:
            self._jpeg, self._ts, self._received = jpeg, sent + self.clock.offset, received
            self._frame = None

    def latest(self) -> Frame | None:
        with self._lock:
            if self._jpeg is None or time.time() - self._received > STALE_S:
                return None
            if self._frame is None:
                import cv2

                image = cv2.imdecode(np.frombuffer(self._jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    return None
                self._frame = Frame(image=image, ts=self._ts)
            return self._frame

    def close(self) -> None:
        pass


class RemoteRecorder:
    """The ``Recorder`` the assistant uses, with the microphone on the gaming PC."""

    def __init__(self) -> None:
        self._level = 0.0
        self._clip = np.zeros(0, np.float32)

    def start(self) -> None:
        self._level = 0.0

    def level(self) -> float:
        return self._level

    def stop(self) -> np.ndarray:
        clip, self._clip = self._clip, np.zeros(0, np.float32)
        return clip


class RemoteVoice:
    """Kokoro on this PC, speaking through the gaming PC's speakers. ``play``
    blocks for as long as the sentence lasts, like local playback does."""

    def __init__(self, tts, link: EdgeLink) -> None:
        self.tts = tts
        self.link = link
        self.voice = getattr(tts, "voice", "")
        self.device = getattr(tts, "device", "")
        self._stop = threading.Event()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return self.tts.synthesize(text)

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._stop.is_set():
            return
        self.link.send(pack(SPEECH, float(sample_rate), to_pcm(samples)))
        self._stop.wait(len(samples) / sample_rate)

    def start(self) -> None:
        self._stop.clear()

    def stop(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self.link.send(json.dumps({"t": "stop"}))

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


class EdgeLink:
    """The server end: one gaming PC at a time, the newest one wins."""

    def __init__(self) -> None:
        self.source = NetworkSource()
        self.recorder = RemoteRecorder()
        self.on_ptt_down: Callable[[], None] | None = None
        self.on_ptt_up: Callable[[], None] | None = None
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._held = False

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def send(self, data: bytes | str) -> None:
        """Send to the gaming PC from any thread; dropped when none is connected."""
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send(ws, data), loop)

    @staticmethod
    async def _send(ws, data) -> None:
        try:
            await ws.send(data)
        except Exception as e:  # closed meanwhile: the next connection takes over
            log.debug("edge send failed: %s", e)

    async def serve(self, ws) -> None:
        old = self._ws
        self._ws, self._loop = ws, asyncio.get_running_loop()
        if old is not None:
            await old.close()
        log.info("gaming PC connected from %s", ws.remote_address)
        try:
            async for data in ws:
                self.handle(data)
        except Exception as e:
            log.info("gaming PC link closed: %s", e)
        finally:
            if self._ws is ws:
                self._ws = None
                if self._held:  # the key was down when the link went: drop the question
                    self.handle(pack(CLIP, 16000.0, b""))
                log.info("gaming PC disconnected")

    def handle(self, data: bytes | str) -> None:
        if isinstance(data, str):
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                return
            t = msg.get("t") if isinstance(msg, dict) else None
            if t == "ptt":
                self._held = True
                if self.on_ptt_down:
                    self.on_ptt_down()
            elif t == "level":
                try:
                    self.recorder._level = min(max(float(msg.get("v", 0.0)), 0.0), 1.0)
                except (TypeError, ValueError):
                    pass
            return
        try:
            kind, value, payload = unpack(data)
        except ValueError:
            return
        if kind == FRAME:
            self.source.put(payload, value)
        elif kind == CLIP:
            if not self._held:
                return
            self._held = False
            audio = from_pcm(payload)
            rate = int(value) or 16000
            if rate != 16000 and len(audio):  # the recorder always sends 16 kHz; just in case
                audio = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32)
            self.recorder._clip = audio
            if self.on_ptt_up:
                self.on_ptt_up()
