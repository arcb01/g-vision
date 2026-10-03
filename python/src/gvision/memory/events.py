"""Event stream (plan section 8): an append-only log of what happened.

For now the only writer is the frame history: every stored frame, it diffs
the tracked objects in the world state and logs what appeared and what
left. Only labels YOLOE is prompted for are tracked, so this sees watched
targets and the startup prompts, not everything on screen.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from gvision.world import WorldState, where


@dataclass
class Event:
    ts: float
    text: str
    ref: str | None = None


class EventLog:
    def __init__(self, world: WorldState, maxlen: int = 200) -> None:
        self.world = world
        self.events: deque[Event] = deque(maxlen=maxlen)
        self._known: dict[str, str] = {}
        """ref -> label of confirmed objects already logged as appeared."""

    def add(self, text: str, ts: float | None = None, ref: str | None = None) -> None:
        self.events.append(Event(time.time() if ts is None else ts, text, ref))

    def update(self, ts: float | None = None) -> None:
        """Log objects that were confirmed or dropped since the last call."""
        ts = time.time() if ts is None else ts
        alive = {o.ref for o in self.world.objects}
        for o in self.world.confirmed():
            if o.ref not in self._known:
                self._known[o.ref] = o.label
                self.add(f"{o.label} appeared {where(o.box)}", ts, o.ref)
        # A lost track keeps its ref for a short grace period, so a brief
        # occlusion doesn't log "gone" and "appeared" again.
        for ref in [r for r in self._known if r not in alive]:
            self.add(f"{self._known.pop(ref)} gone", ts, ref)

    def since(self, ts: float) -> list[Event]:
        return [e for e in self.events if e.ts >= ts]

    def describe(self, ts: float, now: float | None = None, limit: int = 12) -> list[str]:
        """'3 s ago: zombie appeared on the left', newest last, for prompts."""
        now = time.time() if now is None else now
        return [f"{max(0, round(now - e.ts))} s ago: {e.text}" for e in self.since(ts)[-limit:]]
