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
from gvision.audio.mic import SAMPLE_RATE, Recorder
from gvision.audio.tts import KokoroTTS
from gvision.bridge import Bridge
from gvision.protocol import (
    AnswerFinishedMsg,
    AnswerMsg,
    ClearMsg,
    DimMsg,
    FocusMsg,
    Message,
    Segment,
    SegmentStartedMsg,
    VoiceMsg,
)
from gvision.world import WorldState

log = logging.getLogger(__name__)

MIN_CLIP_S = 0.3
SPOTLIGHT_HOLD_S = 2.0
"""Dimming stays this long after the answer is spoken (plan 6.1)."""


class Assistant:
    def __init__(
        self, bridge: Bridge, world: WorldState, agent: Agent,
        asr: SpeechToText | None = None, tts: KokoroTTS | None = None, recorder: Recorder | None = None,
        dim_strength: float = DimMsg.model_fields["strength"].default, hold_s: float = SPOTLIGHT_HOLD_S,
    ) -> None:
        self.bridge = bridge
        self.world = world
        self.agent = agent
        self.asr = asr
        self.tts = tts
        self.recorder = recorder
        self.dim_strength = dim_strength
        self.hold_s = hold_s
        self._task: asyncio.Task | None = None
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

    def ptt_up(self) -> None:
        if not self.recorder:
            return
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
        answer_id = f"a{next(self._ids)}"
        self.bridge.send(AnswerMsg(answer_id=answer_id, segments=[Segment(id=0, text=answer.text, refs=answer.refs)]))
        if answer.refs:
            self.bridge.send(FocusMsg(refs=answer.refs, segment_id=0))
            self.bridge.send(DimMsg(on=True, strength=self.dim_strength))
        audio = None
        if self.tts:
            audio = await asyncio.to_thread(self.tts.synthesize, answer.text)
        self.bridge.send(SegmentStartedMsg(answer_id=answer_id, segment_id=0))
        self.bridge.send(VoiceMsg(state="speaking", transcript=None))
        if audio is not None:
            await asyncio.to_thread(self.tts.play, *audio)
        self.bridge.send(AnswerFinishedMsg(answer_id=answer_id))
        self.bridge.send(VoiceMsg(state="idle"))
        if answer.refs:
            await asyncio.sleep(self.hold_s)
            self.bridge.send(DimMsg(on=False, strength=self.dim_strength))
            self.bridge.send(FocusMsg(refs=[]))

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
