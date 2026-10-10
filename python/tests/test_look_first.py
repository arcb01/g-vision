"""Evidence-first: look answers questions with OCR text and tracked objects
attached, and Qwen 2B only checks for an action when the request asks for one."""

import asyncio
import time

from gvision.agent.agent import Agent, asks_action, look_back
from gvision.agent.qwen import Reply
from gvision.agent.tools import TOOLS, ToolExecutor, ToolResult
from gvision.memory import FrameHistory, LookTool
from gvision.memory.evidence import Evidence
from gvision.memory.look import HINT, SCHEMA
from gvision.protocol import Box
from gvision.world import Situation, WorldState
from test_agent import FakeQwen, obj, tool_reply
from test_memory import stored


class FakeText:
    """Just what Evidence reads from the text watcher."""

    def __init__(self, blocks, gone=()):
        self.blocks = {b.ref: b for b in blocks}
        self.log = list(gone)
        self.refreshed = 0

    async def refresh(self):
        self.refreshed += 1

    def visible(self):
        return list(self.blocks.values())

    def recent(self, seconds, now=None):
        return list(self.log) + self.visible()


def block(ref, text, x, y, gone=None):
    from gvision.perception.text_watcher import TextBlock

    now = time.time()
    return TextBlock(ref=ref, box=Box(x=x, y=y, w=0.1, h=0.04), text=text, confidence=0.9, kind="ui",
                     static=False, in_zone=False, appeared=now - 5, changed=now - 5, gone=gone)


def recording_look(seen, said="It says Old Death, 297.2m."):
    async def look(question="", seconds=10):
        seen.append((question, seconds))
        await asyncio.sleep(0.01)
        return ToolResult({"seen": said}, speak=said)

    return look


def test_a_question_goes_straight_to_look():
    world = WorldState()
    seen = []
    tools = ToolExecutor(world)
    tools.register(SCHEMA, recording_look(seen), HINT)
    qwen = FakeQwen()  # the 2B would call set_watch("label") here
    answer = asyncio.run(Agent(qwen, world, tools).handle("What does the label on the center say?"))

    assert answer.text == "It says Old Death, 297.2m." and answer.tool_calls == ["look"]
    assert seen == [("What does the label on the center say?", 0)]
    assert qwen.calls == []
    titles = [s.title for s in answer.steps]
    assert titles == ["A question: look answers", "Qwen vision: look at recent frames"]


def test_a_find_request_checks_for_an_action_alongside_look():
    world = WorldState()
    seen = []
    tools = ToolExecutor(world)
    tools.register(SCHEMA, recording_look(seen), HINT)
    qwen = FakeQwen(Reply("look"))
    answer = asyncio.run(Agent(qwen, world, tools).handle("Show me the exit"))

    assert answer.tool_calls == ["look"] and len(qwen.calls) == 1
    offered = {t["function"]["name"] for t in qwen.calls[0]["tools"]}
    assert offered == {"set_watch", "clear_watch"}
    assert [s.title for s in answer.steps] == ["Qwen checks for an action", "Qwen vision: look at recent frames"]


def test_only_find_where_show_and_stop_ask_for_an_action():
    # The questions replayed from the log, where the 2B wrongly called set_watch.
    for question in ["What does it say on the door sign?", "How many bullets do we have left?",
                     "What controls are shown?", "How many horse statues are there?",
                     "How many blue grenades are there?", "What did I just pick up?",
                     "What does the screen show?", "Is the area clear?"]:
        assert not asks_action(question), question
    for request in ["Where's the person?", "find the cow", "Show me the exit", "Highlight the chest",
                    "Is there a creeper?", "Are there any zombies?", "Stop", "never mind", "I found it",
                    "watch for skeletons", "clear that"]:
        assert asks_action(request), request


def test_an_action_still_runs_and_the_look_is_cancelled():
    async def run():
        world = WorldState()
        cancelled = []
        tools = ToolExecutor(world, find_timeout=0.05)

        async def look(question="", seconds=10):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(question)
                raise

        tools.register(SCHEMA, look, HINT)
        qwen = FakeQwen(tool_reply("set_watch", targets=["cow"]), Reply("I can't see a cow yet. I'll keep watching."))
        answer = await Agent(qwen, world, tools).handle("where is the cow?")
        await asyncio.sleep(0)
        return world, answer, cancelled

    world, answer, cancelled = asyncio.run(run())
    assert answer.tool_calls == ["set_watch"] and list(world.watches) == ["cow"]
    assert answer.text == "I can't see a cow yet. I'll keep watching."
    assert cancelled == ["where is the cow?"]


def test_route_first_keeps_the_old_path():
    world = WorldState()
    seen = []
    tools = ToolExecutor(world)
    tools.register(SCHEMA, recording_look(seen), HINT)
    qwen = FakeQwen(tool_reply("query_state"))
    answer = asyncio.run(Agent(qwen, world, tools, route_first=True).handle("How many bullets do I have left?"))
    assert answer.tool_calls == ["query_state", "look"]
    assert qwen.calls[0]["tools"] == TOOLS + [SCHEMA]


def test_look_back_covers_what_just_happened():
    assert look_back("What just hit me?") == 10
    assert look_back("what was that") == 10
    assert look_back("What does the sign say?") == 0
    assert look_back("How many grenades are there?") == 0


def test_evidence_puts_text_in_the_named_region_first():
    text = FakeText([block("text:1", "HP 20", 0.02, 0.02), block("text:2", "D Old Death 297.2m", 0.45, 0.48)])
    world = WorldState()
    world.set_objects(1.0, [obj("obj:1", "zombie", x=0.05), obj("obj:2", "zombie", x=0.85, y=0.05)])
    world.set_situation(Situation(ts=1.0, situation="Night in a forest.", player="unknown"))
    got = asyncio.run(Evidence(world, text).gather("What does the label on the center say?"))

    assert text.refreshed == 1
    first = got.prompt.splitlines()[0]
    assert first.index("Old Death") < first.index("HP 20")
    assert '"D Old Death 297.2m" (in the center)' in first
    assert "2 zombie (on the left, top right)" in got.prompt
    assert "situation: Night in a forest." in got.prompt and "unknown" not in got.prompt
    assert got.counts == {"text": 2, "objects": 2}
    assert set(got.blocks) == {"text:1", "text:2"}


def test_evidence_adds_text_that_is_gone_for_what_just_happened():
    gone = block("text:9", "Steve was slain by Zombie", 0.4, 0.8, gone=time.time() - 3)
    got = asyncio.run(Evidence(None, FakeText([], gone=[gone])).gather("what did that message say?", 10))
    assert "is gone now: \"Steve was slain by Zombie\" (bottom center), 3 s ago" in got.prompt
    assert got.counts == {"text": 0, "gone_text": 1}


def test_look_sends_the_evidence_with_the_screen():
    history = FrameHistory()
    history.add(stored(time.time()))
    text = FakeText([block("text:2", "D Old Death 297.2m", 0.45, 0.48)])
    qwen = FakeQwen(Reply('It says "D Old Death 297.2m".'))
    tools = ToolExecutor(WorldState())
    tools.register(SCHEMA, LookTool(qwen, history, evidence=Evidence(WorldState(), text)), HINT)
    result = asyncio.run(tools.run("look", {"question": "What does the label say?", "seconds": 0}))

    prompt = qwen.calls[0]["messages"][0]["content"][-1]["text"]
    assert "read exactly by OCR: \"D Old Death 297.2m\"" in prompt
    assert "quote it exactly" in prompt and "or what G-VISION already knows" in prompt
    assert result.content["evidence"] == {"text": 1, "objects": 0}
    assert "text:2" in result.text_blocks
    assert "Sent along: 1 text, 0 objects" in tools.step("look", {}, result, 1.0).detail


def test_look_without_evidence_keeps_its_prompt():
    history = FrameHistory()
    history.add(stored(time.time()))
    qwen = FakeQwen(Reply("A zombie."))
    asyncio.run(LookTool(qwen, history)("what is that?", seconds=0))
    prompt = qwen.calls[0]["messages"][0]["content"][-1]["text"]
    assert "already knows" not in prompt and "Only say what you can see in the screenshots;" in prompt


def recording_lookup(seen, said=None):
    from gvision.knowledge.lookup import SCHEMA as LOOKUP

    async def lookup(question=""):
        seen.append(question)
        if said is None:
            return ToolResult({"error": "the Minecraft wiki names nothing in the question"})
        return ToolResult({"answer": said, "wiki": "Minecraft Wiki", "pages": ["Trading > Mason"]}, speak=said)

    return LOOKUP, lookup


def test_a_game_question_goes_to_the_wiki_first():
    world = WorldState()
    looked, asked = [], []
    tools = ToolExecutor(world)
    tools.register(SCHEMA, recording_look(looked), HINT)
    tools.register(*recording_lookup(asked, "Masons want clay balls and stone for emeralds."))
    answer = asyncio.run(Agent(FakeQwen(), world, tools).handle("what does the mason villager want"))

    assert answer.text.startswith("Masons want") and answer.tool_calls == ["lookup"]
    assert asked == ["what does the mason villager want"] and looked == []
    assert "Trading > Mason" in answer.steps[-1].detail


def test_look_answers_when_the_wiki_has_nothing_or_the_question_is_about_the_screen():
    world = WorldState()
    looked, asked = [], []
    tools = ToolExecutor(world)
    tools.register(SCHEMA, recording_look(looked), HINT)
    tools.register(*recording_lookup(asked))
    agent = Agent(FakeQwen(), world, tools)
    answer = asyncio.run(agent.handle("how many hearts do zombies have"))
    assert answer.tool_calls == ["look"] and asked == ["how many hearts do zombies have"]
    assert [s.title for s in answer.steps][:2] == ["A question: look answers",
                                                   "Wiki lookup: nothing found, look answers"]
    asyncio.run(agent.handle("what does this sign say"))
    assert len(asked) == 1 and len(looked) == 2
