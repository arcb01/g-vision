"""The slow path's brain: one request -> Qwen -> tools -> a short answer (plan 9.2).

Every request goes to Qwen with a compact world-state snapshot and the tool
list. Tool results go back to Qwen once more for a short, speakable answer
grounded in what the tools found.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from gvision.agent.qwen import Reply
from gvision.agent.tools import ToolExecutor, ToolResult
from gvision.world import WorldState

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are G-VISION, a voice assistant that helps a player see what is on their game screen.
You can highlight objects on screen with tools.
- For any "where is", "find", "show me" or "is there" request, call set_watch.
- Answer in one short spoken sentence, at most 15 words. No markdown, no lists.
- Only say what the tool results or the state show. If something was not found, say you \
can't see it yet and that you're watching for it.
{abilities}Current state: {state}"""

CAN_READ = """\
- For any question about text on screen (signs, quests, menus, messages), call read_text, \
or recent_text for text that already went away. Quote the words that answer the question.
- You can't look back in time yet; say so briefly if asked.
"""
CANNOT_READ = "- You can't read text or look back in time yet; say so briefly if asked.\n"


class Chat(Protocol):
    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 200,
    ) -> Reply: ...


@dataclass
class Answer:
    text: str
    refs: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    """Tool names called, for logs and tests."""
    latency_ms: dict[str, float] = field(default_factory=dict)


def fallback_text(results: list[ToolResult]) -> str:
    """Spoken answer built from tool results when Qwen gives none."""
    for r in results:
        if "text" in r.content:
            blocks = r.content["text"]
            return f"It says: {blocks[0]['text']}" if blocks else "I can't see any text there."
        for f in r.content.get("found", []):
            if f["count"]:
                return f"The {f['target']} is {f['where'][0]}." if f["count"] == 1 else (
                    f"I see {f['count']} {f['target']}s, highlighted.")
            return f"I can't see a {f['target']} yet. I'll keep watching."
        if "cleared" in r.content:
            return "Cleared."
    return "Done."


class Agent:
    def __init__(self, qwen: Chat, world: WorldState, tools: ToolExecutor | None = None) -> None:
        self.qwen = qwen
        self.world = world
        self.tools = tools or ToolExecutor(world)

    async def handle(self, request: str) -> Answer:
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                abilities=CAN_READ if self.tools.text else CANNOT_READ, state=json.dumps(self.world.snapshot()))},
            {"role": "user", "content": request},
        ]
        reply = await self.qwen.chat(messages, tools=self.tools.specs)
        latency = {"llm_tool_call": (time.perf_counter() - t0) * 1000}
        if not reply.tool_calls:
            return Answer(reply.content or "Sorry, I didn't get that.", latency_ms=latency)

        results: list[ToolResult] = []
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": reply.raw.get("tool_calls")})
        t1 = time.perf_counter()
        for call in reply.tool_calls:
            log.info("tool call: %s(%s)", call.name, call.arguments)
            result = await self.tools.run(call.name, call.arguments)
            results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.json()})
        latency["tools"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        final = await self.qwen.chat(messages, max_tokens=60)
        latency["llm_answer"] = (time.perf_counter() - t2) * 1000
        text = final.content or fallback_text(results)
        refs = [ref for r in results for ref in r.refs] + self.tools.highlight_text(text, results)
        return Answer(text, refs, [c.name for c in reply.tool_calls], latency)
