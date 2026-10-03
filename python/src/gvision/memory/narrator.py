"""Situation narrator (plan 7.3): Qwen summarizes the scene every ~25 s.

Input: the newest frame (plus the most eventful one since the last run),
what appeared or left, and the previous summary. Output, as constrained
JSON: the current situation, the player's state as far as visible, the
objective, and a short running summary of the session. It goes into the
world state, so every request's snapshot carries it and "what's going on?"
needs no extra call.

It skips a run when nothing changed on screen (a paused game, a menu left
open) and always yields to the player's requests (see ``SharedQwen``).
Its readings are approximate context, not precise values.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

import httpx

from gvision.memory.events import EventLog
from gvision.memory.history import FrameHistory
from gvision.memory.shared import Preempted, SharedQwen
from gvision.world import Situation, WorldState

log = logging.getLogger(__name__)

MIN_CHANGE = 0.02
"""Largest thumbnail change since the last run below which the scene counts as unchanged."""

PROMPT = """\
You watch a player's video game screen and keep notes for a voice assistant.
Previous notes: {previous}
What the object tracker saw since then: {events}
The images are screenshots, oldest first; the last one is now.
Reply with JSON:
- situation: what is happening right now, one sentence.
- player: the player's visible state (health, danger, what they hold), one short sentence or "unknown".
- objective: what the player seems to be doing or should do next, one short sentence or "unknown".
- summary: the whole session so far in at most 3 sentences: the previous summary updated with what is new.
Only describe what you can see. Health and other readings are rough estimates."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("situation", "player", "objective", "summary")},
    "required": ["situation", "player", "objective", "summary"],
}


def image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


class Narrator:
    def __init__(
        self, qwen: SharedQwen, history: FrameHistory, events: EventLog, world: WorldState,
        every_s: float = 25.0,
    ) -> None:
        self.qwen = qwen
        self.history = history
        self.events = events
        self.world = world
        self.every_s = every_s
        self._last_ts = 0.0
        """Capture time of the newest frame the last run looked at."""

    def due(self) -> bool:
        latest = self.history.latest()
        if latest is None or latest.ts <= self._last_ts:
            return False
        if self.world.situation is None:
            return True
        return self.history.max_change_since(self._last_ts) >= MIN_CHANGE or bool(self.events.since(self._last_ts))

    async def narrate(self) -> Situation | None:
        latest = self.history.latest()
        if latest is None:
            return None
        since = self._last_ts or latest.ts - self.every_s
        frames = [f for f in self.history.pick(latest.ts - since, k=2, now=latest.ts)]
        previous = self.world.situation
        text = PROMPT.format(
            previous=json.dumps(previous.notes()) if previous else "none yet",
            events="; ".join(self.events.describe(since, now=latest.ts)) or "nothing new",
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": text}, *(image_part(f.data_url()) for f in frames)]}]
        reply = await self.qwen.background_chat(
            messages, max_tokens=250,
            response_format={"type": "json_schema", "json_schema": {"name": "situation", "schema": SCHEMA}},
        )
        self._last_ts = latest.ts
        situation = parse(reply.content, latest.ts)
        if situation:
            self.world.set_situation(situation)
            log.info("narrator: %s", situation.situation)
        return situation

    async def run(self, stop: asyncio.Event) -> None:
        wait = self.every_s
        warned = False
        while not stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), wait)
            if stop.is_set():
                break
            wait = self.every_s
            if not self.due():
                continue
            try:
                await self.narrate()
                warned = False
            except Preempted:
                wait = 2.0  # the player asked something; try again once they're done
            except httpx.HTTPError as e:
                if not warned:
                    log.warning("narrator: Qwen request failed (%s); is the mmproj vision file loaded?", e)
                    warned = True


def parse(content: str, ts: float) -> Situation | None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"situation": content} if content else None
    if not isinstance(data, dict):
        return None
    fields = {k: str(data.get(k) or "").strip()[:300] for k in ("situation", "player", "objective", "summary")}
    if not fields["situation"]:
        return None
    return Situation(ts=ts, updated=time.monotonic(), **fields)
