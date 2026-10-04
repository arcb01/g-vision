"""Slow path wiring for "find X" (plan build step 5).

Hold push-to-talk -> record -> speech-to-text -> Qwen with tools ->
highlight -> answer in the panel, spotlight on what was found, and Kokoro
speaks it. A new push-to-talk or the dismiss hotkey cancels everything.
Every answered question also goes to the panel's conversation log, with
what was on screen when it was asked.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import sys
import threading
import time
from collections.abc import Callable

import httpx

from gvision.agent.agent import Agent, Answer
from gvision.audio.asr import SpeechToText
from gvision.audio.mic import SAMPLE_RATE, Recorder, envelope
from gvision.audio.tts import KokoroTTS, split_sentences
from gvision.bridge import Bridge
from gvision.protocol import (
    AnswerFinishedMsg,
    AnswerMsg,
    ClearMsg,
    ExchangeMsg,
    Message,
    Segment,
    SegmentStartedMsg,
    Step,
    VoiceMsg,
)
from gvision.snapshot import Encode, LatestFrame, encode
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
        screen: LatestFrame | None = None, encode_fn: Encode = encode,
        on_listen: Callable[[], None] | None = None,
    ) -> None:
        self.bridge = bridge
        self.world = world
        self.agent = agent
        self.asr = asr
        self.tts = tts
        self.recorder = recorder
        self.screen = screen
        self._encode = encode_fn
        self._on_listen = on_listen
        """Called on push-to-talk press: look reads the screen ahead."""
        self._asked: tuple[float, str, object] | None = None
        """(time, "voice" or "typed", screen image) of the question being answered."""
        self._logging: set[asyncio.Task] = set()
        self._asr_step: Step | None = None
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
        if self._on_listen:
            self._on_listen()
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
        self._remember_question("voice")
        self._task = asyncio.create_task(self._from_audio(audio))

    async def _from_audio(self, audio) -> None:
        if len(audio) < MIN_CLIP_S * SAMPLE_RATE or self.asr is None:
            self.bridge.send(VoiceMsg(state="idle"))
            return
        self.bridge.send(VoiceMsg(state="thinking"))
        t0 = time.perf_counter()
        text = await asyncio.to_thread(self.asr.transcribe, audio)
        ms = (time.perf_counter() - t0) * 1000
        log.info("heard %r (%.0f ms, %.1f s clip)", text, ms, len(audio) / SAMPLE_RATE)
        self._asr_step = Step(
            kind="asr", title=f"Speech-to-text: {getattr(self.asr, 'name', type(self.asr).__name__)}",
            detail=f"heard {text!r} in a {len(audio) / SAMPLE_RATE:.1f} s clip", ms=round(ms, 1), ok=bool(text),
        )
        if not text:
            self.bridge.send(VoiceMsg(state="idle", transcript=""))
            return
        await self.ask(text)

    # --- one request -------------------------------------------------------

    def submit(self, text: str) -> asyncio.Task:
        """Typed request: same path as speech, minus the microphone."""
        self.ptt_down()
        self._remember_question("typed")
        self._task = asyncio.create_task(self.ask(text))
        return self._task

    async def ask(self, text: str) -> Answer:
        self.bridge.send(VoiceMsg(state="thinking", transcript=text))
        steps = [self._asr_step] if self._asr_step else []
        self._asr_step = None
        try:
            answer = await self.agent.handle(text)
        except httpx.HTTPError as e:
            log.error("Qwen request failed: %s", e)
            answer = Answer("I can't reach the language model right now.", steps=[
                Step(kind="llm", title="Qwen request failed", detail=f"{type(e).__name__}: {e}", ok=False)])
        log.info("answer %r refs=%s tools=%s latency=%s", answer.text, answer.refs, answer.tool_calls,
                 {k: round(v) for k, v in answer.latency_ms.items()})
        steps += answer.steps
        logged = False

        def log_now(extra: list[Step]) -> None:  # once the voice is ready, or if cut short before
            nonlocal logged
            if not logged:
                logged = True
                self._log_exchange(text, answer, steps + extra)

        try:
            await self.present(answer, log_now)
        finally:
            log_now([])
        return answer

    # --- conversation log ----------------------------------------------------

    def _remember_question(self, via: str) -> None:
        self._asked = (time.time(), via, self.screen.grab() if self.screen else None)

    def _log_exchange(self, question: str, answer: Answer, steps: list[Step]) -> None:
        asked_ts, via, image = self._asked or (time.time(), "typed", None)
        self._asked = None
        exchange_id = f"x{int(asked_ts * 1000)}"
        task = asyncio.create_task(self._send_exchange(exchange_id, asked_ts, via, image, question, answer, steps))
        self._logging.add(task)  # keep a reference until it is sent
        task.add_done_callback(self._logging.discard)

    async def _send_exchange(self, exchange_id, asked_ts, via, image, question: str, answer: Answer,
                             steps: list[Step]) -> None:
        screenshot = None
        if image is not None:
            try:
                screenshot = await asyncio.to_thread(self._encode, image)
            except Exception as e:  # never lose the entry over its picture
                log.warning("conversation log: no screenshot: %s", e)
        self.bridge.send(ExchangeMsg(
            exchange_id=exchange_id, asked_ts=asked_ts, question=question, answer=answer.text, via=via,
            tools=answer.tool_calls, latency_ms={k: round(v, 1) for k, v in answer.latency_ms.items()},
            steps=steps, screenshot=screenshot,
        ))

    async def present(self, answer: Answer, ready: Callable[[list[Step]], None] | None = None) -> None:
        # Glow and spotlight are the live pipeline's job: they stay on for as
        # long as the watched target is tracked, not just while speaking.
        answer_id = f"a{next(self._ids)}"
        self.bridge.send(AnswerMsg(answer_id=answer_id, segments=[Segment(id=0, text=answer.text, refs=answer.refs)]))
        # The voice is made one sentence at a time and starts after the first,
        # while the next one is synthesized.
        pieces = split_sentences(answer.text) if self.tts else []
        audio: asyncio.Queue = asyncio.Queue()
        synth = asyncio.create_task(self._synthesize(pieces, audio)) if pieces else None
        tts_steps: list[Step] = []
        first = None
        try:
            if synth:
                if hasattr(self.tts, "start"):
                    self.tts.start()
                t0 = time.perf_counter()
                first = await audio.get()
                if first is not None:
                    speech = f"{len(first[0]) / first[1]:.1f} s of speech"
                    device = getattr(self.tts, "device", "")
                    tts_steps.append(Step(
                        kind="tts", title=" ".join(filter(None, ["Voice: Kokoro", getattr(self.tts, "voice", ""),
                                                                  device and f"on {device}"])),
                        detail=speech if len(pieces) == 1 else f"first of {len(pieces)} sentences: {speech}",
                        ms=round((time.perf_counter() - t0) * 1000, 1),
                    ))
            if ready:
                ready(tts_steps)
            text = self.agent.tools.text
            cues = answer.text_cues if text else []
            if cues:
                text.show([ref for _, ref in cues])
            self.bridge.send(SegmentStartedMsg(answer_id=answer_id, segment_id=0))
            self.bridge.send(VoiceMsg(state="speaking"))
            if first is not None:
                await self._speak(first, pieces, audio, cues, len(answer.text))
            elif cues:  # no voice: still walk through the text at reading pace
                duration = len(answer.text) / CHARS_PER_S
                await self._light_text([(at * duration, ref) for at, ref in cues])
        finally:
            if synth:
                synth.cancel()
        self.bridge.send(AnswerFinishedMsg(answer_id=answer_id))
        self.bridge.send(VoiceMsg(state="idle"))

    async def _synthesize(self, pieces: list[str], out: asyncio.Queue) -> None:
        try:
            for piece in pieces:
                await out.put(await asyncio.to_thread(self.tts.synthesize, piece))
        except Exception as e:  # say what is ready rather than nothing
            log.error("voice failed: %s", e)
        finally:
            out.put_nowait(None)

    async def _speak(self, first, pieces: list[str], audio: asyncio.Queue, cues: list[tuple[float, str]],
                     length: int) -> None:
        """Play each sentence as it is ready, lighting text when its words come."""
        tasks: list[asyncio.Task] = []
        start = 0  # where the sentence starts in the answer text
        chunk = first
        try:
            for i, piece in enumerate(pieces):
                if chunk is None or getattr(self.tts, "stopped", False):
                    break
                samples, rate = chunk
                duration = len(samples) / rate
                end = start + len(piece) if i < len(pieces) - 1 else max(length, start + 1)
                mine = [((at * length - start) / max(end - start, 1) * duration, ref)
                        for at, ref in cues if start <= at * length < end]
                tasks.append(asyncio.create_task(self._light_text(mine)))
                levels = asyncio.create_task(self._send_voice_levels(envelope(samples, rate, LEVEL_HZ)))
                try:
                    await asyncio.to_thread(self.tts.play, samples, rate)
                finally:
                    levels.cancel()
                start = end + 1
                if i < len(pieces) - 1:
                    chunk = await audio.get()
        finally:
            for t in tasks:
                t.cancel()

    async def _light_text(self, cues: list[tuple[float, str]]) -> None:
        """Light each text block at its time (seconds from now)."""
        start = time.monotonic()
        for at, ref in cues:
            await asyncio.sleep(max(0.0, start + at - time.monotonic()))
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
