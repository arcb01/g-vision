"""One llama-server for the player and the narrator, the player first (plan 7.3).

``chat`` is the player's path: it cancels any narrator call in flight
(closing its connection, so llama-server drops that generation) and runs at
once. ``background_chat`` is the narrator's: it waits until no player
request has touched Qwen for ``quiet_s``, so it never lands between the tool
call and the answer of the same request.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class Preempted(Exception):
    """A player request took Qwen while a background call was running."""


class SharedQwen:
    def __init__(self, qwen: Any, quiet_s: float = 5.0) -> None:
        self.qwen = qwen
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
        self._active += 1
        for task in self._background:
            task.cancel()
        try:
            return await self.qwen.chat(messages, tools=tools, max_tokens=max_tokens, **kwargs)
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
        task = asyncio.ensure_future(self.qwen.chat(messages, max_tokens=max_tokens, **kwargs))
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
