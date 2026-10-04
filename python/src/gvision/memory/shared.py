"""One llama-server for the player and the narrator, the player first (plan 7.3).

``chat`` is the player's path: it cancels any narrator call in flight
(closing its connection, so llama-server drops that generation) and runs at
once. ``background_chat`` is the narrator's: it waits until no player
request has touched Qwen for ``quiet_s``, so it never lands between the tool
call and the answer of the same request.

With a separate vision model (``vision``, a second llama-server picked in the
Settings tab), the narrator and the look tool (``SharedQwen.looking``) use it
and the player's routing and answers stay on the main Qwen. The narrator
still yields to the player, since both servers share the GPU.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class Preempted(Exception):
    """A player request took Qwen while a background call was running."""


class SharedQwen:
    def __init__(self, qwen: Any, quiet_s: float = 5.0, vision: Any = None) -> None:
        self.qwen = qwen
        self.vision = vision or qwen
        self.looking = _Looking(self)
        self.quiet_s = quiet_s
        self._active = 0
        self._last_used = float("-inf")
        self._background: set[asyncio.Task] = set()
        self._changed = asyncio.Event()

    @property
    def busy(self) -> bool:
        return self._active > 0 or time.monotonic() - self._last_used < self.quiet_s

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   max_tokens: int = 200, **kwargs: Any):
        return await self._foreground(self.qwen, messages, tools, max_tokens, **kwargs)

    async def _foreground(self, client: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                          max_tokens: int, **kwargs: Any):
        self._active += 1
        for task in self._background:
            task.cancel()
        try:
            return await client.chat(messages, tools=tools, max_tokens=max_tokens, **kwargs)
        finally:
            self._active -= 1
            self._last_used = time.monotonic()
            self._wake()

    async def background_chat(self, messages: list[dict[str, Any]], max_tokens: int = 200, **kwargs: Any):
        while self.busy:
            wait = max(0.05, self.quiet_s - (time.monotonic() - self._last_used))
            changed = self._changed
            try:
                await asyncio.wait_for(changed.wait(), wait)
            except TimeoutError:
                pass
        task = asyncio.ensure_future(self.vision.chat(messages, max_tokens=max_tokens, **kwargs))
        self._background.add(task)
        try:
            return await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and not (current and current.cancelling()):
                raise Preempted from None
            raise
        finally:
            self._background.discard(task)
            task.cancel()

    def _wake(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    async def health(self) -> bool:
        return await self.qwen.health()

    async def close(self) -> None:
        await self.qwen.close()
        if self.vision is not self.qwen:
            await self.vision.close()


class _Looking:
    """The look tool's way in: the player's priority, on the vision model."""

    def __init__(self, shared: SharedQwen) -> None:
        self.shared = shared

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   max_tokens: int = 200, **kwargs: Any):
        return await self.shared._foreground(self.shared.vision, messages, tools, max_tokens, **kwargs)
