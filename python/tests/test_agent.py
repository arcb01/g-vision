import asyncio
import json

import httpx
import pytest

from gvision.agent.agent import Agent, fallback_text
from gvision.agent.qwen import QwenClient, Reply, ToolCall
from gvision.agent.tools import TOOLS, ToolExecutor
from gvision.protocol import Box, TrackedObject
from gvision.world import WorldState, normalize_target, where


def obj(ref, label, x=0.1, y=0.4, status="confirmed"):
    return TrackedObject(ref=ref, label=label, confidence=0.9, box=Box(x=x, y=y, w=0.1, h=0.1), status=status)


class FakeQwen:
    """Scripted replies; records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, tools=None, max_tokens=200):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0)


def tool_reply(name, **args):
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return Reply("", [ToolCall("c1", name, args)], {"role": "assistant", "tool_calls": [call]})


def test_normalize_and_where():
    assert normalize_target("  The Creeper? ") == "creeper"
    assert where(Box(x=0.0, y=0.4, w=0.1, h=0.1)) == "on the left"
    assert where(Box(x=0.8, y=0.0, w=0.1, h=0.1)) == "top right"
    assert where(Box(x=0.45, y=0.45, w=0.1, h=0.1)) == "in the center"


def test_tool_descriptions_keep_routing_hints():
    by_name = {t["function"]["name"]: t["function"]["description"] for t in TOOLS}
    assert set(by_name) == {"set_watch", "clear_watch", "query_state"}
    assert "where is X" in by_name["set_watch"]
    assert "use set_watch instead" in by_name["query_state"]


def test_find_x_end_to_end_with_fake_qwen():
    async def run():
        world = WorldState()
        qwen = FakeQwen(tool_reply("set_watch", targets=["the cow"]), Reply("The cow is on your left."))
        agent = Agent(qwen, world, ToolExecutor(world, find_timeout=1.0))

        async def perception():
            # The detector only finds cows once the watch has added the prompt.
            while "cow" not in world.watches:
                await asyncio.sleep(0.01)
            world.set_objects(1.0, [obj("obj:7", "cow"), obj("obj:8", "pig", x=0.8)])

        task = asyncio.create_task(perception())
        answer = await agent.handle("where is the cow?")
        await task
        return world, qwen, answer

    world, qwen, answer = asyncio.run(run())
    assert answer.text == "The cow is on your left."
    assert answer.refs == ["obj:7"]
    assert answer.tool_calls == ["set_watch"]
    assert list(world.watches) == ["cow"]
    # The first call offers the tools and a snapshot; the answer call gets the tool result.
    assert qwen.calls[0]["tools"] == TOOLS
    assert '"watching": []' in qwen.calls[0]["messages"][0]["content"]
    tool_msg = qwen.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["found"][0]["where"] == ["on the left"]


def test_not_found_is_grounded():
    async def run():
        world = WorldState()
        agent = Agent(FakeQwen(tool_reply("set_watch", targets=["creeper"]), Reply("")), world,
                      ToolExecutor(world, find_timeout=0.05))
        return await agent.handle("find the creeper")

    answer = asyncio.run(run())
    assert answer.refs == []
    assert answer.text == "I can't see a creeper yet. I'll keep watching."


def test_clear_and_query_tools():
    async def run():
        world = WorldState()
        world.set_objects(1.0, [obj("obj:1", "cow"), obj("obj:2", "cow", status="tentative")])
        tools = ToolExecutor(world)
        state = await tools.run("query_state", {})
        world.set_watch("cow")
        cleared = await tools.run("clear_watch", {})
        bad = await tools.run("set_watch", {"nope": 1})
        return state, cleared, bad, world

    state, cleared, bad, world = asyncio.run(run())
    assert state.content["counts"] == {"cow": 1}  # tentative tracks are not reported
    assert cleared.content == {"cleared": ["cow"]} and not world.watches
    assert "error" in bad.content


def test_reply_without_tools_is_spoken_as_is():
    async def run():
        return await Agent(FakeQwen(Reply("I can't read text yet.")), WorldState()).handle("what does the sign say")

    assert asyncio.run(run()).text == "I can't read text yet."


def test_fallback_text_counts():
    from gvision.agent.tools import ToolResult

    r = ToolResult({"found": [{"target": "cow", "count": 3, "where": ["on the left"]}]})
    assert fallback_text([r]) == "I see 3 cows, highlighted."


def test_qwen_client_parses_llama_server_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": "set_watch", "arguments": "{\"targets\": [\"cow\"]}"}}],
        }}]})

    async def run():
        client = QwenClient("http://llama")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await client.chat([{"role": "user", "content": "where is the cow"}], tools=TOOLS)

    reply = asyncio.run(run())
    assert reply.tool_calls == [ToolCall("x", "set_watch", {"targets": ["cow"]})]
    assert seen["tool_choice"] == "auto" and seen["temperature"] == 0
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("args", ["not json", "[1, 2]"])
def test_qwen_client_survives_malformed_arguments(args):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "query_state", "arguments": args}}]}}]})

    async def run():
        client = QwenClient("http://llama")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await client.chat([])

    assert asyncio.run(run()).tool_calls[0].arguments == {}
