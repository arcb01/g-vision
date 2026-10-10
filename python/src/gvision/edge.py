"""The gaming PC's end of two-PC mode: ``python -m gvision --edge ws://<server>:8765/edge``.

Captures the screen and streams it as JPEG, records push-to-talk questions
and plays the answers, while the AI server PC does all the thinking (see
``gvision.link`` for the wire format). Reconnects on its own when the server
restarts or the network drops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading
import time

import numpy as np

from gvision.link import CLIP, FRAME, SPEECH, pack, to_pcm, from_pcm, unpack

log = logging.getLogger(__name__)

RECONNECT_S = 1.0
LEVEL_HZ = 20


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class Player:
    """Plays answer sentences one after another; ``stop`` drops the rest."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, samples: np.ndarray, sample_rate: int) -> None:
        self._stop.clear()
        self._queue.put((samples, sample_rate))

    def stop(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()

    def _run(self) -> None:
        import sounddevice as sd

        while True:
            samples, rate = self._queue.get()
            if self._stop.is_set():
                continue
            sd.play(samples, rate)
            end = time.monotonic() + len(samples) / rate
            while time.monotonic() < end and not self._stop.wait(0.02):
                pass
            if self._stop.is_set():
                sd.stop()


class Edge:
    def __init__(self, url: str, source, ptt_key: str | None, fps: float = 10.0, quality: int = 85,
                 recorder=None, player: Player | None = None) -> None:
        self.url = url
        self.source = source
        self.ptt_key = ptt_key
        self.fps = fps
        self.quality = quality
        self.recorder = recorder
        self.player = player
        self._ws = None
        self._meter: asyncio.Task | None = None
        self._held = False

    async def run(self, stop: asyncio.Event) -> None:
        from websockets.asyncio.client import connect

        ptt = None
        if self.recorder is not None and self.ptt_key:
            from gvision.audio.ptt import PushToTalk

            ptt = PushToTalk(self.ptt_key, self.ptt_down, self.ptt_up)
            ptt.start(asyncio.get_running_loop())
        try:
            while not stop.is_set():
                try:
                    async with connect(self.url, max_size=None, open_timeout=5) as ws:
                        self._ws = ws
                        log.info("edge: connected to %s, streaming the screen at %.0f fps", self.url, self.fps)
                        await self._session(ws, stop)
                except (OSError, asyncio.TimeoutError) as e:
                    log.info("edge: can't reach %s (%s), retrying", self.url, e)
                except Exception as e:
                    log.info("edge: link to %s lost (%s), reconnecting", self.url, e)
                finally:
                    self._ws = None
                    self._held = False
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), RECONNECT_S)
        finally:
            if ptt:
                ptt.stop()
            self.source.close()

    async def _session(self, ws, stop: asyncio.Event) -> None:
        frames = asyncio.create_task(self._stream(ws))
        incoming = asyncio.create_task(self._receive(ws))
        stopped = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({frames, incoming, stopped}, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t is not stopped and t.exception():
                    raise t.exception()
        finally:
            for t in (frames, incoming, stopped):
                t.cancel()

    async def _stream(self, ws) -> None:
        period = 1.0 / self.fps
        while True:
            started = time.monotonic()
            frame = await asyncio.to_thread(self.source.latest)
            if frame is not None:
                jpeg = await asyncio.to_thread(encode_jpeg, frame.image, self.quality)
                await ws.send(pack(FRAME, frame.ts, jpeg))
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _receive(self, ws) -> None:
        async for data in ws:
            if isinstance(data, str):
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(data).get("t") == "stop" and self.player:
                        self.player.stop()
                continue
            with contextlib.suppress(ValueError):
                kind, value, payload = unpack(data)
                if kind == SPEECH and self.player:
                    self.player.add(from_pcm(payload), int(value))

    # --- push-to-talk (called on the loop by PushToTalk) -------------------

    def _send(self, data) -> None:
        ws = self._ws
        if ws is not None:
            asyncio.create_task(self._send_now(ws, data))

    @staticmethod
    async def _send_now(ws, data) -> None:
        with contextlib.suppress(Exception):
            await ws.send(data)

    def ptt_down(self) -> None:
        if self._ws is None:
            log.info("edge: push-to-talk ignored, not connected to the AI server")
            return
        if self.player:
            self.player.stop()
        self._held = True
        self.recorder.start()
        self._send(json.dumps({"t": "ptt"}))
        self._meter = asyncio.create_task(self._levels())

    async def _levels(self) -> None:
        while True:
            await asyncio.sleep(1 / LEVEL_HZ)
            self._send(json.dumps({"t": "level", "v": round(self.recorder.level(), 3)}))

    def ptt_up(self) -> None:
        if not self._held:
            return
        self._held = False
        if self._meter:
            self._meter.cancel()
        from gvision.audio.mic import SAMPLE_RATE

        audio = self.recorder.stop()
        self._send(pack(CLIP, float(SAMPLE_RATE), to_pcm(audio)))
