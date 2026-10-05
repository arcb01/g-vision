"""Tools Qwen can call for "find X" (plan 9.2), and their executor.

The descriptions carry routing hints: on Arnau's PC they lifted Qwen3.5-2B
from 14/20 to 17/20 correct tool calls, and its typical mistake was sending
"where is X?" to ``query_state`` instead of ``set_watch``. Keep the list
short; tools from the plan that are not built yet (learn_label) are left out
so the model can't pick them, the text tools are only offered when the text
watcher runs, and optional features add theirs with ``ToolExecutor.register``
(scene memory adds ``look``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gvision.protocol import Step, StepKind
from gvision.world import WorldState, normalize_target, where

if TYPE_CHECKING:
    from gvision.perception.text_watcher import TextWatcher

FIND_TIMEOUT_S = 1.5
"""How long ``set_watch`` waits for a confirmed track before answering
"not visible yet" (plan 9.4 waits ~2 s before its fallback)."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_watch",
            "description": (
                "Highlight things on screen so the player can find them. "
                "Use this for EVERY 'where is X', 'find X', 'show me X', 'highlight X', "
                "'is there a X' or 'look out for X' request, even when X is not in the "
                "current objects list: it starts looking for X. "
                "Use color_role 'danger' for enemies and hazards, 'info' for text and signs, "
                "otherwise 'target'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short object names, singular, e.g. ['cow'] or ['chest', 'door'].",
                    },
                    "color_role": {"type": "string", "enum": ["target", "danger", "info"]},
                },
                "required": ["targets"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_watch",
            "description": (
                "Stop highlighting. Use when the player says stop, clear, never mind, "
                "or that they found it. Leave target out to clear everything."
            ),
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_state",
            "description": (
                "List the objects the detector is tracking (creatures, people, items in the world), "
                "with counts and positions. Only for counting or listing those, such as 'how many "
                "zombies are there' or 'what do you see'. For 'where is X' use set_watch instead. "
                "Numbers shown on the HUD (ammo, bullets, health, money, time) are text: use read_text."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# Plan 9.2's read_region and get_recent_text, named for what the player asks.
TEXT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": (
                "Read the text on screen: signs, quest and objective prompts, menus, dialogue, "
                "chat, item names, and HUD numbers like ammo, health or money. Use this for EVERY 'what does it say', 'read X', 'what is my "
                "quest' or 'what does the sign/menu/message say' request; never set_watch for text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "about": {"type": "string", "description": "A word from the player's question the text should contain. Optional."},
                    "where": {"type": "string", "description": "Part of the screen if the player said, e.g. 'top right'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_text",
            "description": (
                "Text seen in the last minute, including messages and subtitles that are gone. "
                "Use for 'what did that message say' or 'what did he just say'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "How far back, default 60."},
                    "about": {"type": "string", "description": "A word the text should contain. Optional."},
                },
            },
        },
    },
]

MAX_TEXT_BLOCKS = 6


@dataclass
class ToolResult:
    content: dict[str, Any]
    refs: list[str] = field(default_factory=list)
    """Elements the answer is about, for the spotlight."""
    text_refs: list[str] = field(default_factory=list)
    """Text blocks the answer may be about; highlighted once the answer says which."""
    text_blocks: dict[str, Any] = field(default_factory=dict)
    """Those blocks as they were read, so they can still be outlined if the
    watcher has re-read or lost them by the time the answer is spoken."""
    speak: str | None = None
    """A finished spoken answer (e.g. from Qwen's vision); spares the agent's second call."""

    def json(self) -> str:
        return json.dumps(self.content)


STEP_TITLES: dict[str, tuple[StepKind, str]] = {
    "set_watch": ("detector", "Detector: find and highlight"),
    "clear_watch": ("detector", "Stop highlighting"),
    "query_state": ("tool", "Check what is tracked"),
    "read_text": ("ocr", "OCR: read the screen"),
    "recent_text": ("ocr", "OCR: text seen recently"),
    "look": ("vision", "Qwen vision: look at recent frames"),
}


def brief(value: Any, limit: int = 400) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ToolExecutor:
    def __init__(
        self, world: WorldState, find_timeout: float = FIND_TIMEOUT_S, text: TextWatcher | None = None,
    ) -> None:
        self.world = world
        self.find_timeout = find_timeout
        self.text = text
        self.vision_model: str | None = None
        """The model the look tool uses, named in the panel's step log."""
        self._extra_specs: list[dict[str, Any]] = []
        self.hints: list[str] = []
        """System prompt lines for registered tools."""
        self._extra: dict[str, Callable[..., Awaitable[ToolResult]]] = {}

    @property
    def specs(self) -> list[dict[str, Any]]:
        """What Qwen is offered: the built-in tools, text tools and registered ones."""
        base = TOOLS + TEXT_TOOLS if self.text else TOOLS
        return base + self._extra_specs if self._extra_specs else base

    def register(self, schema: dict[str, Any], handler: Callable[..., Awaitable[ToolResult]], hint: str = "") -> None:
        self._extra_specs.append(schema)
        self._extra[schema["function"]["name"]] = handler
        if hint:
            self.hints.append(hint)

    def has(self, name: str) -> bool:
        return name in self._extra

    async def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "set_watch":
                return await self.set_watch(**args)
            if name == "clear_watch":
                return self.clear_watch(**args)
            if name == "query_state":
                return ToolResult(self.world.snapshot())
            if name == "read_text" and self.text:
                return await self.read_text(**args)
            if name == "recent_text" and self.text:
                return await self.recent_text(**args)
            if name in self._extra:
                return await self._extra[name](**args)
        except TypeError as e:  # wrong or missing arguments
            return ToolResult({"error": f"bad arguments for {name}: {e}"})
        return ToolResult({"error": f"unknown tool {name}"})

    def step(self, name: str, args: dict[str, Any], result: ToolResult, ms: float) -> Step:
        """The panel log's line for one tool call: what went in, what came back."""
        kind, title = STEP_TITLES.get(name, ("tool", name))
        detail = f"{name}({brief(args, 200)})\n→ {brief(result.content)}"
        if kind == "vision" and self.vision_model:
            title = f"{title} · {self.vision_model}"
            detail = f"Model: {self.vision_model}\n{detail}"
        if kind == "ocr" and self.text and self.text.timing_ms:
            detail += "\nRapidOCR " + ", ".join(f"{k} {v:.0f} ms" for k, v in self.text.timing_ms.items())
        screen = result.content.get("screen")
        if screen:
            detail += f"\nScreen sent at {screen} px"
            if result.content.get("read_ahead"):
                detail += ", read while you were talking"
        if "thought" in result.content:
            detail += f"\nReasoning on: thought for {result.content['thought']} words"
        crop = result.content.get("crop")
        if crop:
            detail += f"\nSharp crop of the {crop['region']}: {crop['size']} px at full resolution"
        return Step(kind=kind, title=title, detail=detail, ms=round(ms, 1), ok="error" not in result.content)

    async def set_watch(self, targets: list[str] | str, color_role: str = "target") -> ToolResult:
        if isinstance(targets, str):
            targets = [targets]
        if color_role not in ("target", "danger", "info"):
            color_role = "target"
        names = [n for n in (normalize_target(t) for t in targets) if n]
        if not names:
            return ToolResult({"error": "no target given"})
        for name in names:
            self.world.set_watch(name, color_role)  # type: ignore[arg-type]
        # Wait for the detector (now prompted with the new names) and the
        # tracker's confirmation, so the answer is grounded (plan principle 6).
        deadline = time.monotonic() + self.find_timeout
        while not all(self.world.confirmed(n) for n in names):
            left = deadline - time.monotonic()
            if left <= 0 or not await self.world.next_update(left):
                break
        found, refs = [], []
        for name in names:
            objs = self.world.confirmed(name)
            refs += [o.ref for o in objs]
            found.append({"target": name, "count": len(objs), "where": [where(o.box) for o in objs][:5]})
        return ToolResult({"found": found, "note": "highlighted; still watching for anything not found yet"}, refs)

    def clear_watch(self, target: str | None = None) -> ToolResult:
        return ToolResult({"cleared": self.world.clear_watch(target or None)})

    # --- text (plan 7.1 text watcher) ---------------------------------------

    async def read_text(self, about: str | None = None, where: str | None = None) -> ToolResult:
        from gvision.perception.text_watcher import matches_where

        await self.text.refresh()  # read what is on screen right now
        blocks = [b for b in self.text.visible() if matches_where(b.box, where)]
        content: dict[str, Any] = {}
        if about and blocks:
            wanted = text_words(about)
            matching = [b for b in blocks if wanted & text_words(b.text)]
            if matching:
                blocks = matching
            else:
                content["note"] = f"no text mentions '{about}'; this is all the text there"
        blocks = blocks[:MAX_TEXT_BLOCKS]
        content["text"] = [b.info() for b in blocks]
        if not blocks:
            content["note"] = "no readable text there"
        return ToolResult(content, text_refs=[b.ref for b in blocks], text_blocks={b.ref: b for b in blocks})

    async def recent_text(self, seconds: int | float = 60, about: str | None = None) -> ToolResult:
        await self.text.refresh()
        try:
            seconds = min(max(float(seconds), 1.0), 300.0)
        except (TypeError, ValueError):
            seconds = 60.0
        now = time.time()
        blocks = self.text.recent(seconds, now)
        if about:
            wanted = text_words(about)
            blocks = [b for b in blocks if wanted & text_words(b.text)] or blocks
        blocks = blocks[:MAX_TEXT_BLOCKS]
        content: dict[str, Any] = {"text": [b.info(now) for b in blocks]}
        if not blocks:
            content["note"] = "no text seen in that time"
        shown = [b for b in blocks if b.gone is None]
        return ToolResult(content, text_refs=[b.ref for b in shown], text_blocks={b.ref: b for b in shown})

    def text_cues(self, answer: str, results: list[ToolResult]) -> list[tuple[float, str]]:
        """The text blocks the answer quotes, each with where in the answer it
        starts being read (0..1 of its length), so its glow can follow the voice.

        Candidates are the blocks a text tool returned. When no text tool ran
        or vision answered instead (the reader found no block mentioning the
        player's word, such as "sign"), it is every block on screen, with a
        stricter match, so text vision quotes is still outlined."""
        if not self.text:
            return []
        known = self.text_known(results)
        candidates = [ref for r in results for ref in r.text_refs]
        strict = not candidates
        if strict:
            candidates = [b.ref for b in self.text.visible()]
        lowered = answer.lower()
        said = text_words(answer)
        cues: list[tuple[float, str]] = []
        for ref in dict.fromkeys(candidates):
            block = self.text.blocks.get(ref) or known.get(ref)
            if not block:
                continue
            have = text_words(block.text)
            shared = said & have
            need = min(2, len(have)) if strict else min(2, (len(have) + 1) // 2)
            if shared and len(shared) >= need:
                hits = (re.search(rf"\b{re.escape(w)}", lowered) for w in shared)
                start = min((m.start() for m in hits if m), default=0)
                cues.append((start / max(len(answer), 1), ref))
        # One block read and Qwen paraphrased it: that block is what it read.
        # Not when vision answered or the reader said nothing matched.
        settled = not any(r.speak or r.content.get("note") for r in results)
        if not cues and not strict and settled and len(candidates) == 1:
            if candidates[0] in self.text.blocks or candidates[0] in known:
                cues = [(0.0, candidates[0])]
        return sorted(cues)[:MAX_TEXT_BLOCKS]

    def text_known(self, results: list[ToolResult]) -> dict[str, Any]:
        """Text blocks as the tools read them, by ref."""
        return {ref: b for r in results for ref, b in r.text_blocks.items()}


_STOP = {"the", "and", "what", "does", "say", "says", "said", "that", "this", "with", "for", "you", "your",
         "text", "there", "its", "it's", "are", "was", "has", "have", "from", "screen"}


def text_words(text: str) -> set[str]:
    from gvision.perception.text_watcher import words

    return words(text) - _STOP
