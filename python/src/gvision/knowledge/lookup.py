"""lookup: answer a question about the game from its wiki, not from the screen.

"What does the mason want" isn't on screen, and the 2B and 4B models only
half know it. When a session has a game, questions that don't point at the
screen ("this", "that", "on screen", "left"...) try lookup first: the local
wiki index finds the sections the question names, and the vision model
answers from their text alone (no image, so it is quick). When the wiki
names nothing in the question, look answers as before.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from gvision.agent.tools import ToolResult
from gvision.knowledge.library import Knowledge

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": (
            "Look something up in the game's wiki: items, characters, recipes, trades, quests, "
            "how a mechanic works. Use for questions about the game itself, not about the screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The player's question."}},
            "required": ["question"],
        },
    },
}

HINT = "- For questions about the game itself (items, recipes, trades, quests), call lookup."

ABOUT_SCREEN = re.compile(
    r"\b(this|that(?!\s+(?:they|you|i|we|he|she|it|the|a|an|can|will|is needed)\b)|these|those|here|"
    r"screen|see|seeing|look|looks|looking|showing|shown|visible|left|right|top|bottom|center|centre|"
    r"middle|corner|color|colour|shape|icon|button|highlighted|just|happened|happening|hit|say|says|said|"
    r"read|written|am i|i'm)\b",
    re.I)
"""Words that tie a question to what is on screen: look answers those."""

PROMPT = """\
The player is playing {game}. They ask: "{question}"
From the {source}:
{snippets}
Answer from the wiki text above in one or two short spoken sentences, at most 30 words, no markdown, \
no tables. If the wiki text doesn't answer the question, say you couldn't find it in the wiki."""


def about_screen(question: str) -> bool:
    return bool(ABOUT_SCREEN.search(question))


class LookupTool:
    def __init__(self, qwen: Any, knowledge: Knowledge) -> None:
        self.qwen = qwen
        """The vision model's client: it reads the wiki text better than the 2B."""
        self.knowledge = knowledge

    @property
    def ready(self) -> bool:
        return self.knowledge.game is not None and self.knowledge.index is not None

    async def __call__(self, question: str = "") -> ToolResult:
        k = self.knowledge
        if not self.ready:
            return ToolResult({"error": "no game session, so no wiki"})
        snippets = await k.search(question)
        if not snippets:
            return ToolResult({"error": f"the {k.game.name} wiki names nothing in the question"})
        source = k.status().source or f"{k.game.name} wiki"
        text = PROMPT.format(
            game=k.game.name, question=question, source=source,
            snippets="\n\n".join(f"[{s.where}]\n{s.text}" for s in snippets))
        try:
            reply = await self.qwen.chat([{"role": "user", "content": text}], max_tokens=120)
        except httpx.HTTPError as e:
            log.error("lookup: Qwen request failed: %s", e)
            return ToolResult({"error": "could not read the wiki text"})
        answer = reply.content.strip()
        if not answer:
            return ToolResult({"error": "no answer from the wiki text"})
        return ToolResult({"answer": answer, "wiki": source, "pages": [s.where for s in snippets],
                           "chars": sum(len(s.text) for s in snippets)}, speak=answer)
