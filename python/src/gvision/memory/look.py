"""The ``look`` tool (plan 9.2): Qwen examines recent frames directly.

"What just hit me?", "what was that?" and "what happened?" pick up to four
frames from the frame history (the newest plus the most eventful ones in
the window), add what the tracker saw appear and leave, and ask Qwen's
vision the player's question. Its answer is spoken as it is, which saves
the agent's second round trip. When the question names a part of the screen
("bottom right"), a full-resolution crop of it from the newest frame goes
along, so small HUD details like an ammo count stay readable. For "right
now" questions (seconds 0) the newest frame goes at 1600 px (Settings >
Vision > Image resolution) instead of the 640 px history copy, where an
inventory icon is only about a dozen pixels. With Settings > Vision >
Reasoning on (the default in the app), the vision model thinks briefly (a
200-token budget) before it answers. That is what reads small details: counts,
and with the sharp crop a button prompt like "□ OPEN" that it otherwise calls
"the OPEN button". Longer thinking was slower and talked itself into wrong counts.

``warm`` runs on push-to-talk press: it grabs that frame and has the vision
server read it while the player is still talking (llama-server keeps the
prompt cache), so a right-now look only pays for the question and answer.
The image goes before the text in the prompt so that prefix is shared.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import httpx

from gvision.agent.tools import ToolResult
from gvision.memory.crop import SCREEN_SIDE, crop_url, region_of
from gvision.memory.events import EventLog
from gvision.memory.evidence import Evidence
from gvision.memory.history import FrameHistory
from gvision.memory.narrator import image_part

log = logging.getLogger(__name__)

MAX_SECONDS = 60.0
FRAMES = 4
WARM_MAX_AGE = 30.0
"""Seconds a frame read at push-to-talk press stays usable for the question."""
THINK_BUDGET = 200
"""Thinking tokens before the server makes the model answer (about 130 words)."""
THINK_TOKENS = THINK_BUDGET + 160
"""Room for the thinking plus the short answer when reasoning is on."""

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "look",
        "description": (
            "Look at the last seconds of the game screen. Use for 'what just hit me', "
            "'what was that', 'what happened', 'did you see that' and any question about "
            "what something looks like. seconds: how far back to look, 10 for something that "
            "just happened, 0 for right now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The player's question."},
                "seconds": {"type": "number"},
            },
            "required": ["question"],
        },
    },
}

HINT = "- For 'what just hit me', 'what was that' or 'what happened' questions, call look."

PROMPT = """\
These are screenshots of a video game the player is playing, oldest first: {times}.
What the object tracker saw in that time: {events}
{crop}{evidence}The player asks: "{question}"
Answer in one or two short spoken sentences, at most 25 words, no markdown. \
Only say what you can see in the screenshots{or_know}; if you can't tell, say so."""

EVIDENCE_NOTE = """\
What G-VISION already knows about the screen (use it; the images win if they disagree):
{lines}
When the answer is text on screen, quote it exactly as written above.
"""


CROP_NOTE = (
    "The last image is a sharp full-resolution close-up of the {region} of the screen right now; "
    "use it for small details like numbers and icons.\n"
)


class LookTool:
    def __init__(self, qwen: Any, history: FrameHistory, events: EventLog | None = None, screen: Any = None,
                 screen_side: int = SCREEN_SIDE, reasoning: bool = False, evidence: Evidence | None = None) -> None:
        self.qwen = qwen
        self.evidence = evidence
        """OCR text, tracked objects and situation notes sent along with the screen."""
        self.history = history
        self.events = events
        self.screen = screen
        """``LatestFrame`` with the newest full-resolution frame, for crops."""
        self.screen_side = screen_side
        """Longest side of the "right now" frame; 0 keeps the 640 px history frame."""
        self.reasoning = reasoning
        self._warm: tuple[float, str, str, Any] | None = None
        """(time, data URL, size, image) of the frame read at push-to-talk press."""
        self._warming: asyncio.Task | None = None

    def warm(self) -> None:
        """Push-to-talk press: start reading the screen before the question arrives."""
        if self._warming and not self._warming.done():
            self._warming.cancel()
        self._warm = None
        if self.screen and self.screen_side:
            self._warming = asyncio.ensure_future(self._read_ahead())

    async def _read_ahead(self) -> None:
        sharp = await asyncio.to_thread(self._sharp)
        if sharp is None:
            return
        self._warm = (time.time(), *sharp)
        messages = [{"role": "user", "content": [image_part(sharp[0])]}]
        try:  # only the image's prompt cache matters
            prefill = getattr(self.qwen, "prefill", None)
            if not (prefill and await prefill(messages)):
                await self.qwen.chat(messages, max_tokens=1)
        except httpx.HTTPError as e:
            log.warning("look: reading ahead failed: %s", e)

    async def _warmed(self) -> tuple[str, str, Any] | None:
        """The frame read at press, once its prompt is cached, if still fresh."""
        if self._warming:
            with contextlib.suppress(asyncio.CancelledError):
                await self._warming
        if self._warm and time.time() - self._warm[0] <= WARM_MAX_AGE:
            return self._warm[1:]
        return None

    def _sharp(self) -> tuple[str, str, Any] | None:
        """The whole newest frame at up to ``screen_side`` px, with its size and the frame."""
        image = self.screen.grab() if self.screen and self.screen_side else None
        if image is None:
            return None
        try:
            url, (w, h) = crop_url(image, max_side=self.screen_side)
        except Exception as e:  # the 640 px history frame still answers
            log.warning("look: no sharp frame: %s", e)
            return None
        return url, f"{w}x{h}", image

    def _crop(self, question: str, image: Any = None) -> tuple[str, dict[str, Any]] | None:
        """A sharp crop of the region the question names, from ``image`` or the newest frame."""
        region = region_of(question)
        if region and image is None and self.screen:
            image = self.screen.grab()
        if region is None or image is None:
            return None
        name, box = region
        try:
            url, (w, h) = crop_url(image, box)
        except Exception as e:  # the overview frames still answer
            log.warning("look: no crop: %s", e)
            return None
        return url, {"region": name, "size": f"{w}x{h}"}

    async def __call__(self, question: str = "", seconds: float | int | str = 10) -> ToolResult:
        try:
            seconds = min(max(float(seconds), 0.0), MAX_SECONDS)
        except (TypeError, ValueError):
            seconds = 10.0
        now = time.time()
        frames = self.history.pick(seconds, k=FRAMES if seconds else 1, now=now)
        if not frames:
            return ToolResult({"error": "no frames captured yet"})
        times = ", ".join(f"{max(0.0, now - f.ts):.0f} s ago" for f in frames)
        events = "; ".join(self.events.describe(now - seconds, now=now)) if self.events else ""
        warmed = await self._warmed() if not seconds else None
        sharp = warmed or (await asyncio.to_thread(self._sharp) if not seconds else None)
        # Cut from the same frame as the overview, so both show one moment.
        crop = await asyncio.to_thread(self._crop, question, sharp[2] if sharp else None) if question else None
        known = await self.evidence.gather(question, seconds) if self.evidence else None
        text = PROMPT.format(times=times, events=events or "nothing", question=question or "What happened?",
                             crop=CROP_NOTE.format(region=crop[1]["region"]) if crop else "",
                             evidence=EVIDENCE_NOTE.format(lines=known.prompt) if known and known.prompt else "",
                             or_know=" or what G-VISION already knows" if known and known.prompt else "")
        urls = [sharp[0]] if sharp else [f.data_url() for f in frames]
        images = [image_part(u) for u in urls] + ([image_part(crop[0])] if crop else [])
        messages = [{"role": "user", "content": [*images, {"type": "text", "text": text}]}]
        thought = None
        try:
            if self.reasoning:
                reply = await self.qwen.chat(messages, max_tokens=THINK_TOKENS, think=True, think_budget=THINK_BUDGET)
                thought = len(((reply.raw or {}).get("reasoning_content") or "").split())
            else:
                reply = await self.qwen.chat(messages, max_tokens=80)
        except httpx.HTTPError as e:
            log.error("look: Qwen request failed: %s", e)
            return ToolResult({"error": "could not look at the screen"})
        answer = reply.content.strip()
        if not answer:
            return ToolResult({"error": "nothing seen"})
        content: dict[str, Any] = {"seen": answer, "frames": len(frames), "seconds": seconds}
        if sharp:
            content["screen"] = sharp[1]
        if warmed:
            content["read_ahead"] = True
        if thought is not None:
            content["thought"] = thought
        if crop:
            content["crop"] = crop[1]
        if known:
            content["evidence"] = known.counts
        return ToolResult(content, speak=answer, text_blocks=known.blocks if known else {})
