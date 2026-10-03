"""Tools Qwen can call for "find X" (plan 9.2), and their executor.

The descriptions carry routing hints: on Arnau's PC they lifted Qwen3.5-2B
from 14/20 to 17/20 correct tool calls, and its typical mistake was sending
"where is X?" to ``query_state`` instead of ``set_watch``. Keep the list
short; tools from the plan that are not built yet (read_region, look,
get_recent_text, learn_label) are left out so the model can't pick them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from gvision.world import WorldState, normalize_target, where

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
                "List the objects currently tracked, with counts and positions. "
                "Only for counting or listing questions such as 'how many X are there' or "
                "'what do you see'. For 'where is X' use set_watch instead."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ToolResult:
    content: dict[str, Any]
    refs: list[str] = field(default_factory=list)
    """Elements the answer is about, for the spotlight."""

    def json(self) -> str:
        return json.dumps(self.content)


class ToolExecutor:
    def __init__(self, world: WorldState, find_timeout: float = FIND_TIMEOUT_S) -> None:
        self.world = world
        self.find_timeout = find_timeout

    async def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            if name == "set_watch":
                return await self.set_watch(**args)
            if name == "clear_watch":
                return self.clear_watch(**args)
            if name == "query_state":
                return ToolResult(self.world.snapshot())
        except TypeError as e:  # wrong or missing arguments
            return ToolResult({"error": f"bad arguments for {name}: {e}"})
        return ToolResult({"error": f"unknown tool {name}"})

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
