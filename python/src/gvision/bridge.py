"""Local WebSocket bridge between the Python processes and the Electron app.

The app connects as a client; every message is validated against the shared
protocol in both directions (plan section 12).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import websockets
from pydantic import ValidationError
from websockets.asyncio.server import ServerConnection, serve

from gvision.protocol import Message, dump, parse

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

Handler = Callable[[Message], Awaitable[None]]


class Bridge:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self._clients: set[ServerConnection] = set()
        self._handlers: list[Handler] = []
        self.edge: Callable[[ServerConnection], Awaitable[None]] | None = None
        """Serves the gaming PC on ``/edge`` in two-PC mode (``gvision.link``)."""

    def on_message(self, handler: Handler) -> None:
        self._handlers.append(handler)

    @property
    def connected(self) -> bool:
        return bool(self._clients)

    def send(self, msg: Message) -> None:
        """Broadcast to all connected clients without waiting."""
        if self._clients:
            websockets.broadcast(self._clients, dump(msg))

    async def _serve_client(self, ws: ServerConnection) -> None:
        if ws.request is not None and ws.request.path.startswith("/edge"):
            if self.edge is None:
                await ws.close(1008, "this backend does not take a gaming PC (start it with --source edge)")
                return
            await self.edge(ws)
            return
        log.info("app connected from %s", ws.remote_address)
        self._clients.add(ws)
        try:
            async for raw in ws:
                try:
                    msg = parse(raw)
                except ValidationError as e:
                    log.warning("invalid message from app: %s", e)
                    continue
                for handler in self._handlers:
                    await handler(msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info("app disconnected")

    async def run(self, stop: asyncio.Event) -> None:
        # Frames are a few hundred KB each; the default 1 MB cap is fine for
        # them, but full-screen JPEGs of busy scenes can go past it.
        async with serve(self._serve_client, self.host, self.port, max_size=16 * 2**20):
            log.info("bridge listening on ws://%s:%d", self.host, self.port)
            await stop.wait()
