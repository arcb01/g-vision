"""World state shared by the fast loop and the slow path (plan section 8).

Only the two sections "find X" needs exist so far:

- **tracked objects**, written by the live pipeline after every tracker step;
- **active watches**, written by the agent's ``set_watch`` / ``clear_watch``.

One writer per section; readers take what is there. Everything runs on one
asyncio loop, so no locking is needed.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from gvision.protocol import Box, ColorRole, TrackedObject

_ARTICLES = re.compile(r"^(the|a|an|my|some|any)\s+")


def normalize_target(text: str) -> str:
    """'The Creepers ' -> 'creepers'. Used as both the YOLOE prompt and the label."""
    return _ARTICLES.sub("", " ".join(text.lower().split())).strip(" .?!")


def where(box: Box) -> str:
    """Position in words for the snapshot and spoken answers (plan section 8)."""
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    col = "left" if cx < 0.36 else "right" if cx > 0.64 else "center"
    row = "top" if cy < 0.33 else "bottom" if cy > 0.67 else "middle"
    if col == "center" and row == "middle":
        return "in the center"
    if col == "center":
        return f"{row} center"
    if row == "middle":
        return f"on the {col}"
    return f"{row} {col}"


@dataclass
class Watch:
    target: str
    color_role: ColorRole = "target"
    started: float = field(default_factory=time.monotonic)


class WorldState:
    def __init__(self) -> None:
        self.objects: list[TrackedObject] = []
        self.frame_ts = 0.0
        self.watches: dict[str, Watch] = {}
        self.watch_version = 0
        """Bumped on every watch change, so the pipeline can react once."""
        self._objects_changed = asyncio.Event()

    # --- tracked objects (writer: live pipeline) ---------------------------

    def set_objects(self, frame_ts: float, objects: list[TrackedObject]) -> None:
        self.frame_ts = frame_ts
        self.objects = objects
        self._objects_changed.set()
        self._objects_changed = asyncio.Event()

    async def next_update(self, timeout: float) -> bool:
        """Wait for the next tracker step; False on timeout."""
        try:
            await asyncio.wait_for(self._objects_changed.wait(), timeout)
            return True
        except TimeoutError:
            return False

    def confirmed(self, label: str | None = None) -> list[TrackedObject]:
        return [o for o in self.objects if o.status == "confirmed" and (label is None or o.label == label)]

    # --- active watches (writer: agent) ------------------------------------

    def set_watch(self, target: str, color_role: ColorRole = "target") -> Watch:
        watch = Watch(normalize_target(target), color_role)
        self.watches[watch.target] = watch
        self.watch_version += 1
        return watch

    def clear_watch(self, target: str | None = None) -> list[str]:
        """Remove one watch, or all of them; returns the targets removed."""
        if target is None:
            removed = list(self.watches)
            self.watches.clear()
        else:
            key = normalize_target(target)
            removed = [key] if self.watches.pop(key, None) else []
        if removed:
            self.watch_version += 1
        return removed

    # --- snapshot for Qwen -------------------------------------------------

    def snapshot(self) -> dict:
        """Compact state for the prompt: a few hundred tokens at most."""
        counts: dict[str, int] = {}
        for o in self.confirmed():
            counts[o.label] = counts.get(o.label, 0) + 1
        return {
            "objects": [{"ref": o.ref, "label": o.label, "where": where(o.box)} for o in self.confirmed()][:20],
            "counts": counts,
            "watching": sorted(self.watches),
        }
