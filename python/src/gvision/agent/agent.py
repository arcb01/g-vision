"""The slow path's brain: one request -> Qwen -> tools -> a short answer (plan 9.2).

Every request goes to Qwen with a compact world-state snapshot and the tool
list. Tool results go back to Qwen once more for a short, speakable answer
grounded in what the tools found.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from gvision.agent.qwen import Reply
from gvision.agent.tools import ToolExecutor, ToolResult, brief
from gvision.knowledge.lookup import about_screen
from gvision.protocol import Step
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

ACTION_PROMPT = """\
You are G-VISION, a voice assistant that helps a player see what is on their game screen.
Decide whether the player wants an action:
- For any "where is", "find", "show me", "highlight", "is there a" or "watch for" request, call set_watch.
- When the player says stop, clear, never mind, or that they found it, call clear_watch.
- For anything else (questions about the screen, text, numbers, counts, what happened), call no tool \
and reply with the single word: look.
After a tool runs, answer in one short spoken sentence, at most 15 words, no markdown. Only say what the \
tool results show. If something was not found, say you can't see it yet and that you're watching for it.
Current state: {state}"""

ACTIONS = ("set_watch", "clear_watch")
"""What the 2B model still decides when look answers the questions."""

ASKS_ACTION = re.compile(
    r"\b(where|find|show me|highlight|locate|point (out|to|at)|watch for|keep an eye|stop|clear (it|that|the)"
    r"|never ?mind|found it)\b"
    r"|^\W*(is|are) there\b")
"""Requests that may want set_watch or clear_watch. Anything else goes to look
without asking Qwen: given only the action tools, the 2B called set_watch for
"what does the sign say" and "how many bullets", cancelling look (7 of 8
replayed questions)."""


def asks_action(request: str) -> bool:
    return bool(ASKS_ACTION.search(request.lower()))


PAST = re.compile(r"\b(just|was|were|happened|hit|did|earlier|before|ago|missed|that was)\b")


def look_back(request: str) -> int:
    """Seconds look should cover: 10 for something that just happened, 0 for right now."""
    return 10 if PAST.search(request.lower()) else 0


CAN_READ = """\
- For any question about text on screen (signs, quests, menus, messages, or HUD numbers \
like ammo, health and money), call read_text, \
or recent_text for text that already went away. Quote the words that answer the question.
"""
CANNOT_READ = "- You can't read text yet; say so briefly if asked.\n"
CANNOT_LOOK = "- You can't look back in time yet; say so briefly if asked.\n"


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
    text_cues: list[tuple[float, str]] = field(default_factory=list)
    """(where in the text it starts being read, 0..1; text block ref): the
    block lights up when the voice reaches it."""
    text_blocks: dict[str, Any] = field(default_factory=dict)
    """Blocks as the text tools read them, for outlining ones since lost."""
    steps: list[Step] = field(default_factory=list)
    """What it took, for the panel's log."""


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


def mentions_tracked(request: str, snapshot: dict[str, Any]) -> bool:
    """Whether the request names any of the objects being tracked."""
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", request.lower())}
    return any(w.rstrip("s") in words for label in snapshot.get("counts", {}) for w in label.lower().split())


QUESTION_WORDS = {"how", "what", "where", "which", "who", "is", "are", "do", "does", "can", "any", "whats"}
CANT_SEE = re.compile(
    r"\b(can't|cannot|can not|don't|do not|unable)\b.*\b(see|find|tell)\b|\bwatching for\b", re.IGNORECASE)


def asks_about_screen(request: str, reply: str) -> bool:
    """Whether a request Qwen answered without a tool still needed one: a
    question, or a reply that just says it can't see."""
    words = re.findall(r"[a-z]+", request.lower().replace("'", ""))
    return request.rstrip().endswith("?") or bool(words and words[0] in QUESTION_WORDS) or bool(
        CANT_SEE.search(reply or ""))


class Agent:
    def __init__(self, qwen: Chat, world: WorldState, tools: ToolExecutor | None = None,
                 vision_only: bool = False, route_first: bool = False) -> None:
        self.qwen = qwen
        self.world = world
        self.tools = tools or ToolExecutor(world)
        self.vision_only = vision_only
        """Testing: skip routing and let the vision model answer every request."""
        self.route_first = route_first
        """The old path: Qwen picks any tool first and look is the fallback. For comparing the two."""

    def abilities(self) -> str:
        lines = CAN_READ if self.tools.text else CANNOT_READ
        if not self.tools.has("look"):
            lines += CANNOT_LOOK
        return lines + "".join(h + "\n" for h in self.tools.hints)

    def missed(self, request: str, names: list[str], results: list[ToolResult]) -> tuple[str, str] | None:
        """(why, detail) when the tools came back without what was asked and
        vision could still answer, else None."""
        if not self.tools.has("look") or any(r.speak for r in results):
            return None
        for name, result in zip(names, results):
            if name == "query_state" and not mentions_tracked(request, result.content):
                return "Nothing tracked matches", f"tracked: {sorted(result.content.get('counts', {})) or 'nothing'}"
            if name == "read_text" and result.content.get("note", "").startswith(("no text mentions", "no readable")):
                return "No matching text", result.content["note"]
        return None

    async def look_only(self, request: str) -> Answer:
        steps = [Step(kind="llm", title="Routing skipped: vision only",
                      detail="Settings > Vision > Vision only is on; the question goes straight to look")]
        t = time.perf_counter()
        args = {"question": request, "seconds": 0}
        look = await self.tools.run("look", args)
        ms = (time.perf_counter() - t) * 1000
        steps.append(self.tools.step("look", args, look, ms))
        text = look.speak or look.content.get("error") or "I couldn't see anything."
        return self.with_text(Answer(text, look.refs, ["look"], {"tools": ms}, steps=steps), [look])

    async def look_first(self, request: str) -> Answer:
        """Evidence-first: look answers questions, with OCR text and tracked
        objects attached. Only a request that may want an action (find, where,
        show me, stop) also asks the 2B model, at the same time."""
        args = {"question": request, "seconds": look_back(request)}
        if not asks_action(request):
            wiki_steps: list[Step] = []
            if self.tools.has("lookup") and not about_screen(request):
                answer = await self._lookup(request, wiki_steps)
                if answer:
                    return answer
            steps = [Step(kind="llm", title="A question: look answers",
                          detail="no find, where, show or stop in it, so no action check; "
                                 "look answers with what OCR and the tracker know"), *wiki_steps]
            look, ms = await self._timed_look(args)
            steps.append(self.tools.step("look", args, look, ms))
            text = look.speak or "I couldn't see that right now."
            return self.with_text(Answer(text, look.refs, ["look"], {"tools": ms}, steps=steps), [look])
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ACTION_PROMPT.format(state=json.dumps(self.world.snapshot()))},
            {"role": "user", "content": request},
        ]
        router = asyncio.ensure_future(self.qwen.chat(messages, tools=self.tools.action_specs))
        looking = asyncio.ensure_future(self._timed_look(args))
        try:
            try:
                reply = await router
            except httpx.HTTPError as e:  # look still answers
                log.warning("action check failed: %s", e)
                reply = None
            latency = {"llm_tool_call": (time.perf_counter() - t0) * 1000}
            actions = [c for c in reply.tool_calls if c.name in ACTIONS] if reply else []
            if actions:
                looking.cancel()
                reply.tool_calls = actions
                calls = ", ".join(f"{c.name}({brief(c.arguments, 150)})" for c in actions)
                steps = [Step(kind="llm", title="Qwen checks for an action", ms=round(latency["llm_tool_call"], 1),
                              detail=f"called {calls}; the look started alongside was cancelled")]
                return await self._act(request, messages, reply, steps, latency)
            steps = [Step(kind="llm", title="Qwen checks for an action", ms=round(latency["llm_tool_call"], 1),
                          detail="no action: look answers, with what OCR and the tracker know" if reply
                          else "the action check failed; look answers", ok=reply is not None)]
            look, ms = await looking
        finally:
            for task in (router, looking):
                if not task.done():
                    task.cancel()
        steps.append(self.tools.step("look", args, look, ms))
        latency["tools"] = ms
        text = look.speak or "I couldn't see that right now."
        return self.with_text(Answer(text, look.refs, ["look"], latency, steps=steps), [look])

    async def _lookup(self, request: str, steps: list[Step]) -> Answer | None:
        """A question about the game, not the screen: the session game's wiki answers.
        None, with a step saying why, when the wiki names nothing in it."""
        args = {"question": request}
        t = time.perf_counter()
        result = await self.tools.run("lookup", args)
        ms = (time.perf_counter() - t) * 1000
        steps.append(self.tools.step("lookup", args, result, ms))
        if not result.speak:
            steps[-1].ok = True  # not a failure: look answers instead
            steps[-1].title += ": nothing found, look answers"
            return None
        return Answer(result.speak, [], ["lookup"], {"tools": ms}, steps=[
            Step(kind="llm", title="A question about the game: the wiki answers",
                 detail="nothing in it points at the screen, so the session game's wiki answers first"),
            *steps])

    async def _timed_look(self, args: dict[str, Any]) -> tuple[ToolResult, float]:
        t = time.perf_counter()
        result = await self.tools.run("look", args)
        return result, (time.perf_counter() - t) * 1000

    async def handle(self, request: str) -> Answer:
        if self.vision_only and self.tools.has("look"):
            return await self.look_only(request)
        if self.tools.has("look") and not self.route_first:
            return await self.look_first(request)
        t0 = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                abilities=self.abilities(), state=json.dumps(self.world.snapshot()))},
            {"role": "user", "content": request},
        ]
        reply = await self.qwen.chat(messages, tools=self.tools.specs)
        latency = {"llm_tool_call": (time.perf_counter() - t0) * 1000}
        calls = ", ".join(f"{c.name}({brief(c.arguments, 150)})" for c in reply.tool_calls)
        steps = [Step(
            kind="llm", title="Qwen chooses what to do", ms=round(latency["llm_tool_call"], 1),
            detail=f"called {calls}" if calls else f"answered without a tool: {reply.content!r}",
            ok=bool(calls or reply.content),
        )]
        if not reply.tool_calls:
            if self.tools.has("look") and asks_about_screen(request, reply.content):
                # The 2B sometimes answers a screen question from the state
                # alone ("I can't see any grenades"): let vision look.
                steps.append(Step(kind="tool", title="Answered without looking: look instead",
                                  detail=f"Qwen said {reply.content!r}"))
                t = time.perf_counter()
                args = {"question": request, "seconds": 0}
                look = await self.tools.run("look", args)
                steps.append(self.tools.step("look", args, look, (time.perf_counter() - t) * 1000))
                if look.speak:
                    latency["tools"] = (time.perf_counter() - t) * 1000
                    steps.append(Step(kind="llm", title="Answer taken from the tool",
                                      detail="no second Qwen call needed"))
                    return self.with_text(Answer(look.speak, look.refs, ["look"], latency, steps=steps), [look])
            return Answer(reply.content or "Sorry, I didn't get that.", latency_ms=latency, steps=steps)
        return await self._act(request, messages, reply, steps, latency)

    async def _act(self, request: str, messages: list[dict[str, Any]], reply: Reply, steps: list[Step],
                   latency: dict[str, float]) -> Answer:
        """Run the tools Qwen called, then have it say the answer (or take a tool's)."""
        results: list[ToolResult] = []
        messages.append({"role": "assistant", "content": reply.content or "", "tool_calls": reply.raw.get("tool_calls")})
        t1 = time.perf_counter()
        for call in reply.tool_calls:
            log.info("tool call: %s(%s)", call.name, call.arguments)
            t = time.perf_counter()
            result = await self.tools.run(call.name, call.arguments)
            steps.append(self.tools.step(call.name, call.arguments, result, (time.perf_counter() - t) * 1000))
            results.append(result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.json()})
        names = [c.name for c in reply.tool_calls]
        miss = self.missed(request, names, results)
        if miss:
            # Asked about something the detector doesn't track or the text
            # reader can't find (an ammo icon, a door's colour...): let vision
            # answer instead of "can't see".
            steps.append(Step(kind="tool", title=f"{miss[0]}: look instead", detail=miss[1]))
            t = time.perf_counter()
            args = {"question": request, "seconds": 0}
            look = await self.tools.run("look", args)
            steps.append(self.tools.step("look", args, look, (time.perf_counter() - t) * 1000))
            if look.speak:
                latency["tools"] = (time.perf_counter() - t1) * 1000
                steps.append(Step(kind="llm", title="Answer taken from the tool", detail="no second Qwen call needed"))
                return self.with_text(Answer(look.speak, look.refs, names + ["look"], latency, steps=steps),
                                      results + [look])
        latency["tools"] = (time.perf_counter() - t1) * 1000
        refs = [ref for r in results for ref in r.refs]
        if all(r.speak for r in results):
            steps.append(Step(kind="llm", title="Answer taken from the tool", detail="no second Qwen call needed"))
            return self.with_text(Answer(" ".join(r.speak for r in results), refs, [c.name for c in reply.tool_calls],
                                         latency, steps=steps), results)

        t2 = time.perf_counter()
        final = await self.qwen.chat(messages, max_tokens=60)
        latency["llm_answer"] = (time.perf_counter() - t2) * 1000
        text = final.content or fallback_text(results)
        steps.append(Step(
            kind="llm", title="Qwen writes the answer", ms=round(latency["llm_answer"], 1),
            detail=final.content if final.content else f"no answer; used the built-in fallback: {text!r}",
            ok=bool(final.content),
        ))
        return self.with_text(Answer(text, refs, [c.name for c in reply.tool_calls], latency, steps=steps), results)

    def with_text(self, answer: Answer, results: list[ToolResult]) -> Answer:
        """Attach the on-screen text the answer quotes, whichever tool answered,
        so its outline follows the voice."""
        answer.text_cues = self.tools.text_cues(answer.text, results)
        answer.text_blocks = self.tools.text_known(results)
        answer.refs += [ref for _, ref in answer.text_cues if ref not in answer.refs]
        return answer
