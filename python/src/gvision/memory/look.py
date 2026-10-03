"""The ``look`` tool (plan 9.2): Qwen examines recent frames directly.

"What just hit me?", "what was that?" and "what happened?" pick up to four
frames from the frame history (the newest plus the most eventful ones in
the window), add what the tracker saw appear and leave, and ask Qwen's
vision the player's question. Its answer is spoken as it is, which saves
the agent's second round trip.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from gvision.agent.tools import ToolResult
from gvision.memory.events import EventLog
from gvision.memory.history import FrameHistory
from gvision.memory.narrator import image_part

log = logging.getLogger(__name__)

MAX_SECONDS = 60.0
FRAMES = 4

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
The player asks: "{question}"
Answer in one or two short spoken sentences, at most 25 words, no markdown. \
Only say what you can see in the screenshots; if you can't tell, say so."""


class LookTool:
    def __init__(self, qwen: Any, history: FrameHistory, events: EventLog | None = None) -> None:
        self.qwen = qwen
        self.history = history
        self.events = events

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
        text = PROMPT.format(times=times, events=events or "nothing", question=question or "What happened?")
        messages = [{"role": "user", "content": [{"type": "text", "text": text}, *(image_part(f.data_url()) for f in frames)]}]
        try:
            reply = await self.qwen.chat(messages, max_tokens=80)
        except httpx.HTTPError as e:
            log.error("look: Qwen request failed: %s", e)
            return ToolResult({"error": "could not look at the screen"})
        answer = reply.content.strip()
        if not answer:
            return ToolResult({"error": "nothing seen"})
        return ToolResult({"seen": answer, "frames": len(frames), "seconds": seconds}, speak=answer)
