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


class QwenClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 200,
        response_format: dict[str, Any] | None = None, think: bool = False,
    ) -> Reply:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Belt and braces with --reasoning off: the 2B model loops when thinking.
            # think=True is only for the vision server (Settings > Vision > Reasoning).
            "chat_template_kwargs": {"enable_thinking": think},
        }
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

    async def health(self) -> bool:
        try:
            r = await self._http.get(f"{self.url}/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._http.aclose()
