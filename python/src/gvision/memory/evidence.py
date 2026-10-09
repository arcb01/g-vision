"""What perception already knows, attached to every look (evidence-first).

In 287 logged questions, 84% ended in the vision model, and half of those
first went through ``read_text`` or ``query_state``, which found nothing
(``read_text`` matches a word from the question, like "label", against the
on-screen text). So instead of trying those tools first, the look tool gets
their results as evidence: the text the watcher reads (exact, so the answer
can quote it), the objects the tracker confirms, and the narrator's notes.
The vision model answers from the screen and the evidence in one call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gvision.memory.crop import region_of
from gvision.world import WorldState, where

if TYPE_CHECKING:
    from gvision.perception.text_watcher import TextBlock, TextWatcher

log = logging.getLogger(__name__)

MAX_TEXT = 8
"""On-screen text blocks sent, most useful first (the region the question names, then changing text)."""
MAX_GONE = 4
"""Text blocks already gone, for "what did that message say"."""
MAX_TEXT_CHARS = 120
MAX_LABELS = 8


@dataclass
class Gathered:
    prompt: str
    """Lines for the look prompt, empty when perception knows nothing yet."""
    counts: dict[str, int] = field(default_factory=dict)
    """How much went in, for the panel's step log."""
    blocks: dict[str, Any] = field(default_factory=dict)
    """The text blocks sent, by ref, so the ones the answer quotes can be outlined."""


class Evidence:
    def __init__(self, world: WorldState | None = None, text: TextWatcher | None = None) -> None:
        self.world = world
        self.text = text

    async def gather(self, question: str, seconds: float = 0.0) -> Gathered:
        lines: list[str] = []
        counts: dict[str, int] = {}
        blocks: dict[str, Any] = {}
        if self.text:
            try:
                await self.text.refresh()  # read what is on screen right now
            except Exception as e:  # stale text is still better than none
                log.warning("evidence: text refresh failed: %s", e)
            shown = rank(self.text.visible(), question)[:MAX_TEXT]
            if shown:
                lines.append("Text on screen now, read exactly by OCR: " + "; ".join(quote(b) for b in shown))
                blocks.update((b.ref, b) for b in shown)
            counts["text"] = len(shown)
            if seconds:
                now = time.time()
                gone = [b for b in self.text.recent(seconds, now) if b.gone is not None][:MAX_GONE]
                if gone:
                    lines.append(f"Text that was on screen in the last {seconds:.0f} s and is gone now: "
                                 + "; ".join(f"{quote(b)}, {max(0, round(now - b.gone))} s ago" for b in gone))
                counts["gone_text"] = len(gone)
        if self.world:
            objects = self.world.confirmed()
            if objects:
                lines.append("Objects the tracker sees now: " + describe_objects(objects))
            counts["objects"] = len(objects)
            if self.world.situation:
                notes = self.world.situation.notes()
                if notes:
                    lines.append("Rough notes on the situation from earlier: "
                                 + " ".join(f"{k}: {v}." for k, v in notes.items() if k != "summary"))
        return Gathered("\n".join(f"- {line}" for line in lines), counts, blocks)


def rank(blocks: list[TextBlock], question: str) -> list[TextBlock]:
    """Blocks in the part of the screen the question names first; otherwise the watcher's order."""
    region = region_of(question)
    if not region:
        return list(blocks)
    x0, y0, x1, y1 = region[1]

    def inside(b: TextBlock) -> bool:
        cx, cy = b.box.x + b.box.w / 2, b.box.y + b.box.h / 2
        return x0 <= cx <= x1 and y0 <= cy <= y1

    return sorted(blocks, key=lambda b: not inside(b))


def quote(block: TextBlock) -> str:
    text = " ".join(block.text.split())[:MAX_TEXT_CHARS].replace('"', "'")
    return f'"{text}" ({where(block.box)})'


def describe_objects(objects: list) -> str:
    """'2 zombie (on the left, top right); 1 cow (in the center)'."""
    by_label: dict[str, list[str]] = {}
    for o in objects:
        by_label.setdefault(o.label, []).append(where(o.box))
    parts = [f"{len(places)} {label} ({', '.join(dict.fromkeys(places))})"
             for label, places in list(by_label.items())[:MAX_LABELS]]
    return "; ".join(parts)
