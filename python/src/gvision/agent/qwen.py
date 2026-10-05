"""Client for Qwen3.5-2B behind a llama.cpp server (OpenAI-compatible API).

Start the server as tested on Windows (plan 9.2: non-thinking, 8k context):

    llama-server -m Qwen3.5-2B-Q4_K_M.gguf --mmproj mmproj-F16.gguf \
        -ngl 99 -c 8192 --jinja --reasoning off --port 8080
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8080"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    """The assistant message as returned, to append to the conversation."""


THINK_END = re.compile(r"</think>")
MEDIA = re.compile(r"<__media_?\w*__>")
"""Where llama-server's chat template put an image (the marker is random per server)."""


class QwenClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 200,
        response_format: dict[str, Any] | None = None, think: bool = False, think_budget: int | None = None,
    ) -> Reply:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Belt and braces with --reasoning off: the 2B model loops when thinking.
            # think=True is only for the vision server (Settings > Vision > Reasoning).
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if think and think_budget:
            # llama-server closes the thinking after this many tokens and the
            # model answers from what it has so far, in the same request.
            body["thinking_budget_tokens"] = think_budget
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if response_format:
            body["response_format"] = response_format
        r = await self._http.post(f"{self.url}/v1/chat/completions", json=body)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                log.warning("malformed tool arguments: %r", fn.get("arguments"))
                args = {}
            calls.append(ToolCall(c.get("id") or f"call_{i}", fn.get("name", ""), args if isinstance(args, dict) else {}))
        # A template that leaves the thinking in the answer: keep only what follows it.
        content = THINK_END.split(msg.get("content") or "")[-1].strip()
        return Reply(content, calls, msg)

    async def prefill(self, messages: list[dict[str, Any]]) -> bool:
        """Have the server read ``messages`` up to their last image, generating nothing.

        Qwen3.5 is partly recurrent, so llama-server can only reuse a cached
        prompt that ends exactly where the next one diverges. A chat request
        would add the end-of-turn tokens after the image; this stops right
        after it, so a later chat with the same images first skips them.
        """
        r = await self._http.post(f"{self.url}/apply-template", json={"messages": messages})
        r.raise_for_status()
        prompt = r.json()["prompt"]
        markers = list(MEDIA.finditer(prompt))
        images = [p["image_url"]["url"].split(",", 1)[1] for m in messages if isinstance(m.get("content"), list)
                  for p in m["content"] if p.get("type") == "image_url"]
        if not markers or len(markers) != len(images):
            return False
        body = {"prompt": {"prompt_string": prompt[:markers[-1].end()], "multimodal_data": images},
                "n_predict": 0, "cache_prompt": True}
        r = await self._http.post(f"{self.url}/completion", json=body)
        r.raise_for_status()
        return True

    async def health(self) -> bool:
        try:
            r = await self._http.get(f"{self.url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._http.aclose()
