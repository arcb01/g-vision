"""Slow path wiring for "find X" (plan build step 5).

Hold push-to-talk -> record -> speech-to-text -> Qwen with tools ->
highlight -> answer in the panel, spotlight on what was found, and Kokoro
speaks it. A new push-to-talk or the dismiss hotkey cancels everything.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import sys
import threading
import time

import httpx

from gvision.agent.agent import Agent, Answer
from gvision.audio.asr import SpeechToText
from gvision.audio.mic import SAMPLE_RATE, Recorder, envelope
from gvision.audio.tts import KokoroTTS
from gvision.bridge import Bridge
from gvision.protocol import (
    AnswerFinishedMsg,
    AnswerMsg,
    ClearMsg,
    Message,
    Segment,
    SegmentStartedMsg,
    VoiceMsg,
)
from gvision.world import WorldState

log = logging.getLogger(__name__)

MIN_CLIP_S = 0.3
LEVEL_HZ = 20
"""How often voice loudness goes to the overlay's wave animation."""
CHARS_PER_S = 15.0
"""Speaking rate assumed when there is no voice (--no-tts), for text cues."""


class Assistant:
    def __init__(
        self, bridge: Bridge, world: WorldState, agent: Agent,
        asr: SpeechToText | None = None, tts: KokoroTTS | None = None, recorder: Recorder | None = None,
    ) -> None:
        self.bridge = bridge
        self.world = world
        self.agent = agent
        self.asr = asr
        self.tts = tts
        self.recorder = recorder
        self._task: asyncio.Task | None = None
        self._meter: asyncio.Task | None = None
        self._ids = itertools.count(1)
        bridge.on_message(self._on_message)

    async def _on_message(self, msg: Message) -> None:
        if isinstance(msg, ClearMsg):  # dismiss hotkey or panel button
            self.cancel()
            self.world.clear_watch()
            self.bridge.send(VoiceMsg(state="idle"))

    def cancel(self) -> None:
        if self.tts:
            self.tts.stop()
        if self._meter:
            self._meter.cancel()
        if self._task and not self._task.done():
            self._task.cancel()

    # --- push-to-talk (called on the loop by PushToTalk) -------------------

    def ptt_down(self) -> None:
        self.cancel()
        self.world.clear_watch()
        self.bridge.send(ClearMsg(reason="push-to-talk"))
        self.bridge.send(VoiceMsg(state="listening"))
        if self.recorder:
            self.recorder.start()
            self._meter = asyncio.create_task(self._send_mic_levels())

    async def _send_mic_levels(self) -> None:
        while True:
            await asyncio.sleep(1 / LEVEL_HZ)
            self.bridge.send(VoiceMsg(state="listening", level=self.recorder.level()))

    def ptt_up(self) -> None:
        if not self.recorder:
            return
        if self._meter:
            self._meter.cancel()
        audio = self.recorder.stop()
        self._task = asyncio.create_task(self._from_audio(audio))

    async def _from_audio(self, audio) -> None:
        if len(audio) < MIN_CLIP_S * SAMPLE_RATE or self.asr is None:
            self.bridge.send(VoiceMsg(state="idle"))
            return
        self.bridge.send(VoiceMsg(state="thinking"))
        t0 = time.perf_counter()
        text = await asyncio.to_thread(self.asr.transcribe, audio)
        log.info("heard %r (%.0f ms, %.1f s clip)", text, (time.perf_counter() - t0) * 1000, len(audio) / SAMPLE_RATE)
        if not text:
            self.bridge.send(VoiceMsg(state="idle", transcript=""))
            return
        await self.ask(text)

    # --- one request -------------------------------------------------------

    def submit(self, text: str) -> asyncio.Task:
        """Typed request: same path as speech, minus the microphone."""
        self.ptt_down()
        self._task = asyncio.create_task(self.ask(text))
        return self._task

    async def ask(self, text: str) -> Answer:
        self.bridge.send(VoiceMsg(state="thinking", transcript=text))
        try:
            answer = await self.agent.handle(text)
        except httpx.HTTPError as e:
            log.error("Qwen request failed: %s", e)
            answer = Answer("I can't reach the language model right now.")
        log.info("answer %r refs=%s tools=%s latency=%s", answer.text, answer.refs, answer.tool_calls,
                 {k: round(v) for k, v in answer.latency_ms.items()})
        await self.present(answer)
        return answer

    async def present(self, answer: Answer) -> None:
        # Glow and spotlight are the live pipeline's job: they stay on for as
        # long as the watched target is tracked, not just while speaking.
        answer_id = f"a{next(self._ids)}"
        self.bridge.send(AnswerMsg(answer_id=answer_id, segments=[Segment(id=0, text=answer.text, refs=answer.refs)]))
        audio = None
        if self.tts:
            audio = await asyncio.to_thread(self.tts.synthesize, answer.text)
        text = self.agent.tools.text
        if text and answer.text_cues:
            text.show([ref for _, ref in answer.text_cues])
        self.bridge.send(SegmentStartedMsg(answer_id=answer_id, segment_id=0))
        self.bridge.send(VoiceMsg(state="speaking"))
        duration = len(audio[0]) / audio[1] if audio is not None else len(answer.text) / CHARS_PER_S
        cues = asyncio.create_task(self._light_text(answer.text_cues, duration)) if text and answer.text_cues else None
        try:
            if audio is not None:
                samples, rate = audio
                levels = asyncio.create_task(self._send_voice_levels(envelope(samples, rate, LEVEL_HZ)))
                try:
                    await asyncio.to_thread(self.tts.play, samples, rate)
                finally:
                    levels.cancel()
            elif cues:
                await cues  # no voice: still walk through the text at reading pace
        finally:
            if cues:
                cues.cancel()
        self.bridge.send(AnswerFinishedMsg(answer_id=answer_id))
        self.bridge.send(VoiceMsg(state="idle"))

    async def _light_text(self, cues: list[tuple[float, str]], duration: float) -> None:
        """Light each text block as the voice reaches the words that quote it."""
        start = time.monotonic()
        for at, ref in cues:
            await asyncio.sleep(max(0.0, start + at * duration - time.monotonic()))
            self.agent.tools.text.light(ref, segment_id=0)

    async def _send_voice_levels(self, levels: list[float]) -> None:
        start = time.monotonic()
        for i, level in enumerate(levels):
            await asyncio.sleep(max(0.0, start + i / LEVEL_HZ - time.monotonic()))
            self.bridge.send(VoiceMsg(state="speaking", level=level))

    async def read_stdin(self, stop: asyncio.Event) -> None:
        """Type requests instead of speaking them (testing without a mic)."""
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[str] = asyncio.Queue()

        def reader() -> None:  # daemon thread: input() must not block exit
            for line in sys.stdin:
                loop.call_soon_threadsafe(lines.put_nowait, line)

        threading.Thread(target=reader, daemon=True).start()
        print("type a request, e.g. 'where is the person?'", flush=True)
        while not stop.is_set():
            line = (await lines.get()).strip()
            if line:
                with contextlib.suppress(asyncio.CancelledError):
                    await self.submit(line)
